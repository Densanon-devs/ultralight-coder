#!/usr/bin/env python3
"""
Ultralite Code Assistant — Desktop App

One click: starts the server in the background and opens a native window.
No browser, no localhost URL, no confusion.

Usage:
    python desktop.py
    python desktop.py --port 8000
"""

import argparse
import logging
import os
import sys
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Suppress noisy logs before anything imports
for _name in ("httpx", "httpcore", "sentence_transformers", "transformers",
              "huggingface_hub", "filelock"):
    logging.getLogger(_name).setLevel(logging.WARNING)
os.environ.setdefault("LLAMA_LOG_LEVEL", "ERROR")
_hf_cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
if _hf_cache.exists():
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")


def start_server(port):
    """Start the FastAPI server in a background thread."""
    import uvicorn
    from server import create_app
    uvicorn.run(
        create_app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
        factory=False,
    )


def wait_for_server(port, timeout=120):
    """Wait until the server is responding."""
    import time
    import urllib.request
    url = f"http://127.0.0.1:{port}/health"
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except Exception:
            time.sleep(0.5)
    return False


def main():
    parser = argparse.ArgumentParser(description="Ultralite Code Assistant — Desktop")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    # Check pywebview
    try:
        import webview
    except ImportError:
        print("pywebview not installed. Install with: pip install pywebview")
        print("Falling back to browser mode...")
        import webbrowser
        server_thread = threading.Thread(target=start_server, args=(args.port,), daemon=True)
        server_thread.start()
        print(f"Starting server on port {args.port}...")
        if wait_for_server(args.port):
            webbrowser.open(f"http://127.0.0.1:{args.port}")
            print("Opened in browser. Press Ctrl+C to stop.")
            try:
                server_thread.join()
            except KeyboardInterrupt:
                pass
        else:
            print("Server failed to start.")
        return

    # Start server in background
    server_thread = threading.Thread(target=start_server, args=(args.port,), daemon=True)
    server_thread.start()

    print("Starting Ultralite Code Assistant...")
    if not wait_for_server(args.port):
        print("Server failed to start within timeout.")
        sys.exit(1)

    # Open native window
    window = webview.create_window(
        "Ultralite Code Assistant",
        f"http://127.0.0.1:{args.port}",
        width=1100,
        height=750,
        min_size=(800, 500),
    )
    webview.start()


if __name__ == "__main__":
    main()
