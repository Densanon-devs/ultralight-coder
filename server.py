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
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

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
        from fastapi.responses import StreamingResponse
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

    class GenerateRequest(PydanticModel):
        prompt: str
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

    @app.on_event("startup")
    async def startup():
        get_engine()

    @app.post("/generate", response_model=GenerateResponse)
    async def generate(req: GenerateRequest):
        engine = get_engine()
        start = time.monotonic()

        if req.max_tokens:
            engine.config.base_model.max_tokens = req.max_tokens
        if req.temperature:
            engine.config.base_model.temperature = req.temperature

        response = engine.process(req.prompt)
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

        async def event_stream():
            for token in engine.process_stream(req.prompt):
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
        return {
            "system": {
                "name": engine.config.system.name,
                "version": engine.config.system.version,
            },
            "model": {
                "loaded": engine.base_model.is_loaded,
            },
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

    print(f"\n  Ultralight Code Assistant API")
    print(f"  http://{args.host}:{args.port}")
    print(f"  Docs: http://{args.host}:{args.port}/docs\n")

    uvicorn.run(
        "server:create_app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        factory=True,
    )


if __name__ == "__main__":
    main()
