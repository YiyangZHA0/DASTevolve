"""Case-owned evaluator for the ASYN hotspot binder design program."""
from pathlib import Path
import sys

CASE_ROOT = Path(__file__).resolve().parent
if str(CASE_ROOT) not in sys.path:
    sys.path.insert(0, str(CASE_ROOT))

from asyn_binder_evaluator import register_asyn_hotspot_binder_plugin
register_asyn_hotspot_binder_plugin()

from astevolve.evaluation.outerloop import evaluate, validate_candidate

__all__ = ["evaluate", "validate_candidate"]
