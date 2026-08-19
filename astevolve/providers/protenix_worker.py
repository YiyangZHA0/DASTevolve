

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


_SCRIPT_DIR = Path(__file__).resolve().parent


def _bool_arg(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _default_num_workers() -> int:
    try:
        return max(0, int(os.environ.get("ASTEVOLVE_PROTENIX_NUM_WORKERS", "1")))
    except (TypeError, ValueError):
        return 1


def _default_need_atom_confidence() -> bool:
    return _bool_arg(os.environ.get("ASTEVOLVE_PROTENIX_NEED_ATOM_CONFIDENCE", "1"))


def _prediction_available(out_dir: Path) -> bool:
    return bool(
        list(out_dir.rglob("*summary_confidence*.json"))
        or list(out_dir.rglob("*.cif"))
    )


def _print_error_files(out_dir: Path) -> None:
    err_dir = out_dir / "ERR"
    if not err_dir.exists():
        return
    for path in sorted(err_dir.rglob("*.txt"))[:5]:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        print(f"[protenix-worker] {path}:\n{text[-4000:]}", file=sys.stderr)


def _default_params(model_name: str, cycle: int, step: int, use_msa: bool) -> tuple[int, int, bool]:
    if model_name in {"protenix_base_default_v0.5.0", "protenix_base_constraint_v0.5.0"}:
        return 10, 200, use_msa
    if model_name in {
        "protenix_mini_esm_v0.5.0",
        "protenix_mini_ism_v0.5.0",
        "protenix_mini_default_v0.5.0",
        "protenix_tiny_default_v0.5.0",
    }:
        use_msa = False if model_name in {"protenix_mini_esm_v0.5.0", "protenix_mini_ism_v0.5.0"} else use_msa
        return 4, 5, use_msa
    raise RuntimeError(f"{model_name} is not supported for Protenix inference")


def main() -> int:


    parser = argparse.ArgumentParser(description="Run Protenix predict with ASTevolve-safe Windows defaults.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model-name", default="protenix_mini_esm_v0.5.0")
    parser.add_argument("--seeds", default="101")
    parser.add_argument("--cycle", type=int, default=10)
    parser.add_argument("--step", type=int, default=200)
    parser.add_argument("--sample", type=int, default=5)
    parser.add_argument("--use-msa", default="true")
    parser.add_argument("--use-default-params", default="true")
    parser.add_argument("--need-atom-confidence", default="true" if _default_need_atom_confidence() else "false")
    parser.add_argument("--num-workers", type=int, default=_default_num_workers())
    args = parser.parse_args()

    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

    sys.path = [
        entry for entry in sys.path
        if Path(entry or os.getcwd()).resolve() != _SCRIPT_DIR
    ]


    from runner.batch_inference import inference_jsons, inference_configs

    cycle = int(args.cycle)
    step = int(args.step)
    use_msa = _bool_arg(args.use_msa)
    if _bool_arg(args.use_default_params):
        cycle, step, use_msa = _default_params(args.model_name, cycle, step, use_msa)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    inference_configs["num_workers"] = max(0, int(args.num_workers))
    inference_configs["need_atom_confidence"] = _bool_arg(args.need_atom_confidence)

    print(
        "[protenix-worker] "
        f"model={args.model_name} cycle={cycle} step={step} sample={args.sample} "
        f"use_msa={use_msa} num_workers={inference_configs['num_workers']} "
        f"need_atom_confidence={inference_configs['need_atom_confidence']}"
    )
    inference_jsons(
        args.input,
        str(out_dir),
        use_msa=use_msa,
        seeds=tuple(int(seed) for seed in str(args.seeds).split(",") if str(seed).strip()),
        n_cycle=cycle,
        n_step=step,
        n_sample=int(args.sample),
        model_name=args.model_name,
    )

    if not _prediction_available(out_dir):
        print("[protenix-worker] no CIF or summary_confidence output was generated", file=sys.stderr)
        _print_error_files(out_dir)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
