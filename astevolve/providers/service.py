

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple
from urllib import error, parse, request


STRUCTURE_SERVICE_REQUEST_VERSION = "astevolve.structure_service_request.v1"
STRUCTURE_SERVICE_RESPONSE_VERSION = "astevolve.structure_service_response.v1"


class StructureServiceError(RuntimeError):
    pass


class StructureServiceModel:


    name = "service"

    @staticmethod
    def _request(operation: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        call_args = dict(arguments)
        base_url = str(
            call_args.pop("service_url", None)
            or os.environ.get("ASTEVOLVE_STRUCTURE_SERVICE_URL")
            or ""
        ).strip().rstrip("/")
        if not base_url:
            raise StructureServiceError(
                "structure service URL is missing; set structure_service_url "
                "or ASTEVOLVE_STRUCTURE_SERVICE_URL"
            )
        backend = str(
            call_args.pop("service_backend", None)
            or os.environ.get("ASTEVOLVE_STRUCTURE_SERVICE_BACKEND")
            or "esmfold2"
        ).strip().lower()
        if backend in {"service", "structure_service", "remote_service"}:
            raise StructureServiceError("structure service backend cannot recurse into service")
        token = str(
            call_args.pop("service_token", None)
            or os.environ.get("ASTEVOLVE_STRUCTURE_SERVICE_TOKEN")
            or ""
        )
        timeout_value = call_args.pop("service_timeout", None)
        timeout = float(
            timeout_value
            if timeout_value is not None
            else os.environ.get("ASTEVOLVE_STRUCTURE_SERVICE_TIMEOUT", "7200")
        )
        if timeout <= 0:
            raise StructureServiceError("structure service timeout must be positive")

        payload = {
            "schema_version": STRUCTURE_SERVICE_REQUEST_VERSION,
            "backend": backend,
            "operation": operation,
            "arguments": call_args,
        }
        try:
            body = json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise StructureServiceError(
                f"structure service arguments are not exact JSON: {exc}"
            ) from exc
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        endpoint = {
            "confidence_multichain": "/v1/confidence/multichain",
            "confidence_complex": "/v1/confidence/complex",
        }.get(operation)
        if endpoint is None:
            raise StructureServiceError(f"unsupported service operation: {operation}")
        http_request = request.Request(
            base_url + endpoint,
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            hostname = (parse.urlsplit(base_url).hostname or "").lower()
            if hostname in {"127.0.0.1", "localhost", "::1"}:
                opener = request.build_opener(request.ProxyHandler({}))
                response_context = opener.open(http_request, timeout=timeout)
            else:
                response_context = request.urlopen(http_request, timeout=timeout)
            with response_context as response:
                response_body = response.read()
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise StructureServiceError(
                f"structure service returned HTTP {exc.code}: {detail[:1000]}"
            ) from exc
        except (error.URLError, TimeoutError, OSError) as exc:
            raise StructureServiceError(f"structure service request failed: {exc}") from exc
        try:
            envelope = json.loads(response_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise StructureServiceError("structure service returned invalid JSON") from exc
        if not isinstance(envelope, dict):
            raise StructureServiceError("structure service response must be an object")
        if envelope.get("schema_version") != STRUCTURE_SERVICE_RESPONSE_VERSION:
            raise StructureServiceError("structure service response version mismatch")
        if envelope.get("backend") != backend:
            raise StructureServiceError("structure service backend mismatch")
        result = envelope.get("result")
        if not isinstance(result, dict):
            raise StructureServiceError("structure service result must be an object")
        normalized = dict(result)
        normalized["service_transport"] = {
            "schema_version": STRUCTURE_SERVICE_RESPONSE_VERSION,
            "backend": backend,
            "endpoint": base_url,
            "persistent_process": True,
        }
        return normalized

    def confidence_multichain(
        self,
        pred_name: Optional[str] = None,
        chains: Optional[List[Tuple[str, str]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self._request(
            "confidence_multichain",
            {"pred_name": pred_name, "chains": chains or [], **kwargs},
        )

    def scalar_multichain(
        self,
        pred_name: Optional[str] = None,
        chains: Optional[List[Tuple[str, str]]] = None,
        metric: str = "plddt",
        **kwargs: Any,
    ) -> float:
        result = self.confidence_multichain(
            pred_name=pred_name,
            chains=chains,
            metric=metric,
            **kwargs,
        )
        return float(result.get("metrics", {}).get(metric, 0.0))

    def confidence_complex(
        self,
        pred_name: Optional[str] = None,
        entities: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self._request(
            "confidence_complex",
            {"pred_name": pred_name, "entities": entities or [], **kwargs},
        )

    def scalar_complex(
        self,
        pred_name: Optional[str] = None,
        entities: Optional[List[Dict[str, Any]]] = None,
        metric: str = "plddt",
        **kwargs: Any,
    ) -> float:
        result = self.confidence_complex(
            pred_name=pred_name,
            entities=entities,
            metric=metric,
            **kwargs,
        )
        return float(result.get("metrics", {}).get(metric, 0.0))


__all__ = [
    "STRUCTURE_SERVICE_REQUEST_VERSION",
    "STRUCTURE_SERVICE_RESPONSE_VERSION",
    "StructureServiceError",
    "StructureServiceModel",
]
