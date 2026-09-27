

from __future__ import annotations

import contextlib
import json
import sys
import traceback

from .esmfold import run_esmfold_confidence_multichain


def main() -> int:
    for line in sys.stdin:
        request_id = None
        try:
            request = json.loads(line)
            request_id = request.get("request_id")
            with contextlib.redirect_stdout(sys.stderr):
                result = run_esmfold_confidence_multichain(
                    pred_name=request.get("pred_name"),
                    chains=request.get("chains"),
                    metric=str(request.get("metric") or "plddt"),
                    model_name=request.get("model_name"),
                )
            response = {
                "request_id": request_id,
                "status": "ok",
                "result": result,
            }
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            response = {
                "request_id": request_id,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        sys.stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
