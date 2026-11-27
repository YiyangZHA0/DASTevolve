#!/usr/bin/env bash
# Install isolated inference workers. Run once per case, outside Slurm if desired.
set -euo pipefail

usage() {
  echo "Usage: bash scripts/install_case_providers.sh --case {asyn,tiam1,cab,ttgr} --runtime-root DIR [--python /path/to/python3.12]"
  echo "GPU wheel overrides: ASTEVOLVE_TORCH_VERSION, ASTEVOLVE_TORCH_INDEX_URL, ASTEVOLVE_TORCH_WHEELHOUSE."
}
case_name=""
runtime=""
python=python3.12
while (($#)); do
  case "$1" in
    --case) case_name="${2:?missing case name}"; shift 2 ;;
    --runtime-root) runtime="${2:?missing runtime root}"; shift 2 ;;
    --python) python="${2:?missing Python executable}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done
case "$case_name" in
  asyn) providers=(esmfold) ;;
  tiam1) providers=(esmfold protenix) ;;
  cab|ttgr) providers=(protenix) ;;
  *) usage >&2; exit 2 ;;
esac
[[ -n "$runtime" ]] || { usage >&2; exit 2; }
runtime=$(realpath -m -- "$runtime")
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
case "$runtime/" in "$repo/"*) echo "Choose a runtime directory outside the checkout." >&2; exit 2 ;; esac
"$python" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) and sys.platform == "linux" else "Inference workers require Linux and Python 3.12")'
mkdir -p "$runtime/envs"
for provider in "${providers[@]}"; do
  (
    flock -x 9
    cd "$runtime"
    env_dir="$runtime/envs/$provider"
    [[ -x "$env_dir/bin/python" ]] || "$python" -m venv "$env_dir"
    py="$env_dir/bin/python"
    torch_source=(--index-url "${ASTEVOLVE_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}")
    if [[ -n "${ASTEVOLVE_TORCH_WHEELHOUSE:-}" ]]; then
      torch_source=(--no-index --find-links "$ASTEVOLVE_TORCH_WHEELHOUSE")
    fi
    "$py" -m pip install "torch==${ASTEVOLVE_TORCH_VERSION:-2.9.1}" "${torch_source[@]}"
    if [[ "$provider" == esmfold ]]; then
      # The full upstream distributions also declare packages for remote inference
      # and unrelated models. These are the local folding worker dependencies.
      "$py" -m pip install 'transformers==4.57.6' 'numpy>=2.2,<3' 'biotite>=1.4,<3' \
        'biopython>=1.85' 'scipy>=1.12' 'pandas>=2.2.3' 'pydantic>=2.10.6' 'matplotlib>=3.10' \
        'fire>=0.5' 'modal>=1.1,<2' 'deprecated>=1.2.14' 'rich>=14' pyyaml click
      "$py" -m pip install --no-deps \
        'boileroom @ git+https://github.com/softnanolab/boileroom.git@e343f185710505d43af28e77aadf99d93faa5d1b' \
        'biobagel @ git+https://github.com/softnanolab/bagel.git@a248750576a6e22d57660fdc2321224bdc9edcf6'
      "$py" -c 'from bagel import Chain, Residue; from bagel.oracles.folding.esmfold import ESMFold; from boileroom.models.esm.core import ESMFoldCore'
    else
      # Protenix's full distribution also pins training packages and optional
      # CUDA kernels. These workers use its torch inference kernels.
      "$py" -m pip install 'protenix==2.0.0' --no-deps
      "$py" -m pip install 'ml_collections==1.1.0' 'modelcif==1.4' 'fair-esm==2.0.0' \
        'icecream==2.1.7' 'optree==0.17.0' 'biotite==1.4.0' 'gemmi==0.6.7' 'py3Dmol==2.5.2' \
        'pdbeccdutils==1.0.0' 'ipdb==0.13.13' 'wandb==0.21.1' 'biopython==1.85' \
        'rdkit==2025.9.3' 'numpy>=2.2,<3' 'pandas>=2.2,<3' 'matplotlib>=3.10' \
        'scikit-learn>=1.7,<2' 'scipy>=1.12' 'networkx>=3.4' pyyaml pydantic tqdm einops dm-tree requests
      PROTENIX_ROOT_DIR="$runtime/models/protenix" LAYERNORM_TYPE=torch "$py" -c 'from runner.batch_inference import inference_jsons, inference_configs'
    fi
    echo "Ready: $py"
  ) 9>"$runtime/envs/.$provider-install.lock"
done
