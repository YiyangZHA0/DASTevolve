

from cases.demo_case.demo_evaluator import register_demo_gpu_plugin


register_demo_gpu_plugin()

from astevolve.evaluation.outerloop import evaluate, validate_candidate


__all__ = ["evaluate", "validate_candidate"]
