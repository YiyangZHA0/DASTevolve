#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/quickstart.sh {asyn,cab,tiam1,ttgr} [options]
  --plan              Show settings without downloads, GPU work or API calls.
  --prepare-only      Install packages, fetch models and stage static inputs.
  --prepared          Reuse installed Python environments and local weights.
  --output PATH       Create a new external run directory.
  --iterations N      Accepted, consecutive outer revisions (default 2).
  --max-attempts N    Maximum proposal slots (default 3 per requested revision).
Each accepted program runs the case's native 10-round inner search.
EOF
}
if [[ $# -eq 0 || $1 == -h || $1 == --help ]]; then usage; exit 0; fi
example=$1
shift
case "$example" in asyn|cab|tiam1|ttgr) ;; *) usage >&2; exit 2 ;; esac

project=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
if [[ -n ${DASTEVOLVE_ENV_FILE:-} ]]; then
  # shellcheck disable=SC1090
  source "$DASTEVOLVE_ENV_FILE"
elif [[ -f "$project/.env.local" ]]; then
  source "$project/.env.local"
fi
export ASTEVOLVE_PROJECT_ROOT="$project"
export ASTEVOLVE_RUNTIME_ROOT=$(realpath -m "${ASTEVOLVE_RUNTIME_ROOT:-$project/../DASTevolve_runtime}")
export ASTEVOLVE_DATA_ROOT="${ASTEVOLVE_DATA_ROOT:-$ASTEVOLVE_RUNTIME_ROOT/datasets}"
export ASTEVOLVE_MODEL_ROOT="${ASTEVOLVE_MODEL_ROOT:-$ASTEVOLVE_RUNTIME_ROOT/models}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$project:$project/outerloop${PYTHONPATH:+:$PYTHONPATH}"
runtime="$ASTEVOLVE_RUNTIME_ROOT"
bootstrap_python="${ASTEVOLVE_BOOTSTRAP_PYTHON:-python3}"
prepared=0
plan=0
prepare_only=0
args=()
while (($#)); do
  case "$1" in
    --prepared) prepared=1; shift ;;
    --plan) plan=1; args+=("$1"); shift ;;
    --prepare-only) prepare_only=1; args+=("$1"); shift ;;
    --output|--iterations|--max-attempts)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
      args+=("$1" "$2"); shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done
# Resolve relative paths against the caller's working directory.
if ((plan)); then exec "$bootstrap_python" "$project/scripts/run_example.py" "$example" "${args[@]}"; fi
for path in "$runtime" "$ASTEVOLVE_MODEL_ROOT" "$ASTEVOLVE_DATA_ROOT" "${ASTEVOLVE_ARTIFACT_ROOT:-$runtime/runs}"; do
  resolved=$(realpath -m "$path")
  case "$resolved/" in "$project/"*) echo "Choose a directory outside the checkout: $resolved" >&2; exit 2 ;; esac
done
if ((!prepare_only)); then
  : "${ASTEVOLVE_LLM_API_BASE:?Set the base URL for your OpenAI-compatible endpoint}"
  : "${ASTEVOLVE_LLM_MODEL:?Set the model identifier for your endpoint}"
  : "${ASTEVOLVE_LLM_API_KEY:?Set your own API key}"
fi
driver="$runtime/envs/driver/bin/python"
if ((!prepared)); then
  mkdir -p "$runtime/envs"
  (
    flock -x 9
    if [[ ! -x $driver ]]; then
      "$bootstrap_python" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else "Use Python 3.12 or set ASTEVOLVE_BOOTSTRAP_PYTHON")'
      "$bootstrap_python" -m venv "$runtime/envs/driver"
    fi
    torch_source=(--index-url "${ASTEVOLVE_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}")
    if [[ -n ${ASTEVOLVE_TORCH_WHEELHOUSE:-} ]]; then
      torch_source=(--no-index --find-links "$ASTEVOLVE_TORCH_WHEELHOUSE")
    fi
    "$driver" -m pip install "torch==${ASTEVOLVE_TORCH_VERSION:-2.9.1}" "${torch_source[@]}"
    "$driver" -m pip install 'transformers==4.57.6' 'dacite>=1.8,<2' 'openai>=1,<3' 'PyYAML>=6,<7' 'numpy>=1.26,<3' scipy biopython
  ) 9>"$runtime/envs/.driver-install.lock"
  bash "$project/scripts/install_case_providers.sh" --case "$example" --runtime-root "$runtime" --python "$bootstrap_python"
  "$driver" "$project/scripts/setup_case_models.py" --case "$example" --runtime-root "$runtime"
else
  driver="${ASTEVOLVE_PYTHON_BIN:-$driver}"
  [[ -x $driver ]] || { echo "Prepared driver not found: $driver" >&2; exit 2; }
  "$driver" "$project/scripts/setup_case_models.py" --case "$example" --runtime-root "$runtime" --check-only
fi
export ASTEVOLVE_PYTHON_BIN="$driver"
export ASTEVOLVE_ESMFOLD_MODEL="${ASTEVOLVE_ESMFOLD_MODEL:-$ASTEVOLVE_MODEL_ROOT/esmfold_v1}"
export ASTEVOLVE_ESMFOLD_PYTHON="${ASTEVOLVE_ESMFOLD_PYTHON:-$runtime/envs/esmfold/bin/python}"
export ASTEVOLVE_PROTENIX_PYTHON="${ASTEVOLVE_PROTENIX_PYTHON:-$runtime/envs/protenix/bin/python}"
export ASTEVOLVE_PROTENIX_ROOT="${ASTEVOLVE_PROTENIX_ROOT:-$ASTEVOLVE_MODEL_ROOT/protenix}"
export PROTENIX_ROOT_DIR="$ASTEVOLVE_PROTENIX_ROOT"
export ASTEVOLVE_PROTENIX_NUM_WORKERS="${ASTEVOLVE_PROTENIX_NUM_WORKERS:-0}"
export ASTEVOLVE_PROTENIX_NEED_ATOM_CONFIDENCE=1
export LAYERNORM_TYPE="${LAYERNORM_TYPE:-torch}"
export ASTEVOLVE_PROGEN_MODEL_DIR="${ASTEVOLVE_PROGEN_MODEL_DIR:-$ASTEVOLVE_MODEL_ROOT/progen2-small}"
export ASTEVOLVE_MASKED_LM_MODEL_DIR="${ASTEVOLVE_MASKED_LM_MODEL_DIR:-$ASTEVOLVE_MODEL_ROOT/esm2_t6_8M_UR50D}"
export ASTEVOLVE_CAB_LYS3_ESM2_MODEL="${ASTEVOLVE_CAB_LYS3_ESM2_MODEL:-$ASTEVOLVE_MASKED_LM_MODEL_DIR}"
exec "$driver" "$project/scripts/run_example.py" "$example" "${args[@]}"
