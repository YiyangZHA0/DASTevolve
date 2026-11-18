

from __future__ import annotations

import argparse
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from threading import Lock
from typing import Any, Dict, Optional

from astevolve.providers.service import (
    STRUCTURE_SERVICE_REQUEST_VERSION,
    STRUCTURE_SERVICE_RESPONSE_VERSION,
)


MAX_REQUEST_BYTES = 16 * 1024 * 1024


class _StructureHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, handler, *, provider: str, model: Any, token: str):
        super().__init__(address, handler)
        self.provider_name = provider
        self.model = model
        self.auth_token = token
        self.inference_lock = Lock()


class _Handler(BaseHTTPRequestHandler):
    server: _StructureHTTPServer

    def log_message(self, format: str, *args: Any) -> None:

        return

    def _write(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        expected = self.server.auth_token
        if not expected:
            return True
        supplied = self.headers.get("Authorization", "")
        prefix = "Bearer "
        return supplied.startswith(prefix) and hmac.compare_digest(
            supplied[len(prefix) :], expected
        )

    def do_GET(self) -> None:
        if self.path != "/health":
            self._write(404, {"error": "not_found"})
            return
        if not self._authorized():
            self._write(401, {"error": "unauthorized"})
            return
        self._write(
            200,
            {
                "schema_version": STRUCTURE_SERVICE_RESPONSE_VERSION,
                "status": "ok",
                "backend": self.server.provider_name,
                "persistent_process": True,
            },
        )

    def do_POST(self) -> None:
        operations = {
            "/v1/confidence/multichain": "confidence_multichain",
            "/v1/confidence/complex": "confidence_complex",
        }
        expected_operation = operations.get(self.path)
        if expected_operation is None:
            self._write(404, {"error": "not_found"})
            return
        if not self._authorized():
            self._write(401, {"error": "unauthorized"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0 or length > MAX_REQUEST_BYTES:
            self._write(413, {"error": "invalid_content_length"})
            return
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._write(400, {"error": "invalid_json"})
            return
        if not isinstance(payload, dict):
            self._write(400, {"error": "request_must_be_object"})
            return
        if payload.get("schema_version") != STRUCTURE_SERVICE_REQUEST_VERSION:
            self._write(400, {"error": "request_version_mismatch"})
            return
        if str(payload.get("backend") or "").strip().lower() != self.server.provider_name:
            self._write(409, {"error": "backend_mismatch"})
            return
        if payload.get("operation") != expected_operation:
            self._write(400, {"error": "operation_mismatch"})
            return
        arguments = payload.get("arguments")
        if not isinstance(arguments, dict):
            self._write(400, {"error": "arguments_must_be_object"})
            return
        method = getattr(self.server.model, expected_operation, None)
        if not callable(method):
            self._write(422, {"error": "backend_operation_unsupported"})
            return
        try:
            with self.server.inference_lock:
                result = method(**arguments)
            if not isinstance(result, dict):
                raise TypeError("backend result must be an object")
            self._write(
                200,
                {
                    "schema_version": STRUCTURE_SERVICE_RESPONSE_VERSION,
                    "backend": self.server.provider_name,
                    "result": result,
                },
            )
        except Exception as exc:
            self._write(
                500,
                {"error": type(exc).__name__, "message": str(exc)[:2000]},
            )


def create_structure_service(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    provider: str = "esmfold2",
    token: str = "",
    model: Optional[Any] = None,
) -> ThreadingHTTPServer:


    normalized_provider = str(provider or "").strip().lower()
    if normalized_provider in {"", "service", "structure_service", "remote_service"}:
        raise ValueError("service requires a non-service backend provider")
    if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
        raise ValueError("port must be an integer between 0 and 65535")
    if model is None:
        from astevolve.providers.registry import _structure_model

        model = _structure_model(normalized_provider)
    return _StructureHTTPServer(
        (str(host), port),
        _Handler,
        provider=normalized_provider,
        model=model,
        token=str(token or ""),
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Serve one persistent ASTevolve structure-model provider"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--provider", default="esmfold2")
    parser.add_argument(
        "--token",
        default=os.environ.get("ASTEVOLVE_STRUCTURE_SERVICE_TOKEN", ""),
    )
    args = parser.parse_args(argv)
    server = create_structure_service(
        host=args.host,
        port=args.port,
        provider=args.provider,
        token=args.token,
    )
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["MAX_REQUEST_BYTES", "create_structure_service", "main"]
