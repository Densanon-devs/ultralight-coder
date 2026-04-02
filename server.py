#!/usr/bin/env python3
"""
Ultralight Code Assistant — REST API Server

Endpoints:
    POST /generate          — Generate a response
    POST /generate/stream   — Streaming generation via SSE
    POST /route             — Route a prompt (without generating)
    GET  /status            — System status
    GET  /modules           — List modules
    POST /memory/remember   — Store a fact
    POST /memory/search     — Search memory
    GET  /health            — Health check

Usage:
    python server.py
    python server.py --port 9000
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Suppress noisy third-party logs
for _name in ("httpx", "httpcore", "sentence_transformers", "transformers",
              "huggingface_hub", "filelock"):
    logging.getLogger(_name).setLevel(logging.WARNING)

# Suppress llama.cpp verbose output
os.environ.setdefault("LLAMA_LOG_LEVEL", "ERROR")

# Offline mode for sentence-transformers — skip HuggingFace HEAD requests
# on every startup. Only set if the model cache likely exists already.
_hf_cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
if _hf_cache.exists():
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

logger = logging.getLogger("UCA.server")

_engine = None


def get_engine():
    global _engine
    if _engine is None:
        from main import UltralightCodeAssistant
        _engine = UltralightCodeAssistant(dry_run=False)
        _engine.initialize()
        logger.info("Engine initialized for API server")
    return _engine


def create_app():
    try:
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import StreamingResponse, FileResponse
        from fastapi.staticfiles import StaticFiles
        from pydantic import BaseModel as PydanticModel
        from typing import Optional
    except ImportError:
        print("FastAPI not installed. Install with: pip install fastapi uvicorn")
        sys.exit(1)

    app = FastAPI(
        title="Ultralight Code Assistant API",
        description="REST API for the ultralight local coding assistant",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Serve static files (web UI)
    static_dir = PROJECT_ROOT / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    class GenerateRequest(PydanticModel):
        prompt: str
        code: Optional[str] = None
        max_tokens: Optional[int] = None
        temperature: Optional[float] = None

    class GenerateResponse(PydanticModel):
        response: str
        modules: list[str]
        generation_time: float

    class RouteRequest(PydanticModel):
        prompt: str

    class MemoryRequest(PydanticModel):
        content: str
        source: str = "api"
        importance: float = 0.7

    class SearchRequest(PydanticModel):
        query: str
        top_k: int = 5

    class ProjectIndexRequest(PydanticModel):
        path: str

    @app.on_event("startup")
    async def startup():
        get_engine()

    @app.get("/", include_in_schema=False)
    async def root():
        """Serve the web UI."""
        index = PROJECT_ROOT / "static" / "index.html"
        if index.exists():
            return FileResponse(str(index))
        return {"message": "Ultralight Code Assistant API. See /docs for endpoints."}

    @app.get("/models/available")
    async def list_available_models():
        """List all .gguf model files on disk with size and active status."""
        engine = get_engine()
        models_dir = PROJECT_ROOT / "models"
        if not models_dir.exists():
            return {"models": []}

        current_path = engine.config.base_model.path
        result = []
        for p in sorted(models_dir.glob("*.gguf"), key=lambda x: x.stat().st_size):
            size_mb = p.stat().st_size / (1024 * 1024)
            rel = str(p.relative_to(PROJECT_ROOT)).replace("\\", "/")
            result.append({
                "name": p.stem,
                "path": rel,
                "size_mb": round(size_mb),
                "active": rel == current_path.replace("\\", "/"),
            })
        return {"models": result}

    @app.post("/session/reset")
    async def session_reset():
        """Clear conversation history to start a new chat."""
        engine = get_engine()
        engine.memory.short_term.clear()
        return {"reset": True}

    @app.post("/generate", response_model=GenerateResponse)
    async def generate(req: GenerateRequest):
        engine = get_engine()
        start = time.monotonic()

        if req.max_tokens:
            engine.config.base_model.max_tokens = req.max_tokens
        if req.temperature:
            engine.config.base_model.temperature = req.temperature

        # If code is attached, prepend it to the prompt for review/debug
        prompt = req.prompt
        if req.code:
            prompt = f"Here is my code:\n```\n{req.code}\n```\n\n{req.prompt}"

        response = engine.process(prompt)
        elapsed = time.monotonic() - start

        last_perf = engine._perf_history[-1] if engine._perf_history else {}

        return GenerateResponse(
            response=response,
            modules=last_perf.get("modules", []),
            generation_time=round(elapsed, 3),
        )

    @app.post("/generate/stream")
    async def generate_stream(req: GenerateRequest):
        engine = get_engine()

        prompt = req.prompt
        if req.code:
            prompt = f"Here is my code:\n```\n{req.code}\n```\n\n{req.prompt}"

        async def event_stream():
            for token in engine.process_stream(prompt):
                data = json.dumps({"token": token})
                yield f"data: {data}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.post("/route")
    async def route(req: RouteRequest):
        engine = get_engine()
        routing = engine.router.route(
            user_prompt=req.prompt,
            available_modules=engine.modules.available_modules,
        )
        return {
            "modules": routing.selected_modules,
            "scores": routing.scores,
            "routing_mode": routing.routing_mode,
            "confidence": routing.classifier_confidence,
        }

    @app.get("/status")
    async def status():
        engine = get_engine()
        model_name = Path(engine.config.base_model.path).stem
        return {
            "system": {
                "name": engine.config.system.name,
                "version": engine.config.system.version,
            },
            "model": {
                "loaded": engine.base_model.is_loaded,
            },
            "model_name": model_name,
            "augmentors": engine._augmentors_enabled,
            "modules": engine.modules.available_modules,
            "memory": engine.memory.status(),
        }

    @app.get("/modules")
    async def list_modules():
        engine = get_engine()
        return {"modules": engine.modules.list_all()}

    @app.post("/memory/remember")
    async def remember(req: MemoryRequest):
        engine = get_engine()
        engine.memory.remember(req.content, source=req.source, importance=req.importance)
        return {"stored": True, "content": req.content}

    @app.post("/memory/search")
    async def search_memory(req: SearchRequest):
        engine = get_engine()
        if not engine.memory.long_term:
            return {"results": []}

        results = engine.memory.long_term.search(req.query, top_k=req.top_k)
        return {
            "results": [
                {"content": r.content, "source": r.source, "importance": r.importance}
                for r in results
            ],
        }

    @app.post("/project/index")
    async def project_index(req: ProjectIndexRequest):
        """Index a project directory for context-aware code generation."""
        engine = get_engine()
        result = engine.project_index.index_directory(req.path)
        return result

    @app.get("/project/status")
    async def project_status():
        """Get project index status."""
        engine = get_engine()
        return engine.project_index.status()

    @app.post("/project/clear")
    async def project_clear():
        """Clear the project index."""
        engine = get_engine()
        engine.project_index.clear()
        return {"cleared": True}

    @app.get("/health")
    async def health():
        return {"status": "ok", "version": "0.1.0"}

    return app


def main():
    parser = argparse.ArgumentParser(description="Ultralight Code Assistant — API Server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", "-p", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        import uvicorn
    except ImportError:
        print("uvicorn not installed. Install with: pip install uvicorn")
        sys.exit(1)

    print(f"\n  Ultralight Code Assistant")
    print(f"  Web UI: http://{args.host}:{args.port}")
    print(f"  API Docs: http://{args.host}:{args.port}/docs\n")

    uvicorn.run(
        "server:create_app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        factory=True,
    )


if __name__ == "__main__":
    main()
