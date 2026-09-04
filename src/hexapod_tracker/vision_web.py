"""Serve the guided local calibration studio without robot control."""
from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import signal
import threading
from typing import Sequence

from .paths import CONFIG_DIR
from .web_server import VisionRuntime, wrap_handler_with_vision


class _NotFoundHandler(BaseHTTPRequestHandler):
    def _not_found(self) -> None:
        body = json.dumps({"ok": False, "error": "not found"}).encode("utf-8")
        self.send_response(HTTPStatus.NOT_FOUND)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        self._not_found()

    def do_POST(self) -> None:
        self._not_found()

    def log_message(self, format: str, *args: object) -> None:
        del format, args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_DIR / "apriltag_pose_config_20260831.json",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8898)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runtime = VisionRuntime(args.config)
    runtime.start()
    handler = wrap_handler_with_vision(_NotFoundHandler, runtime)
    server = ThreadingHTTPServer((args.host, args.port), handler)

    def shutdown(_signal: int, _frame: object) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    print(f"Hexapod calibration studio: http://{args.host}:{args.port}/vision", flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        runtime.stop()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
