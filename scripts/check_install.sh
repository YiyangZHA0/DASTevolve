#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd "${script_dir}/.." && pwd -P)"
python_bin="${ASTEVOLVE_PYTHON_BIN:-python}"

all_cases=0
while (($#)); do
  case "$1" in
    --python)
      [[ $# -ge 2 ]] || { echo "error: --python needs a path" >&2; exit 2; }
      python_bin="$2"
      shift 2
      ;;
    --all-cases) all_cases=1; shift ;;
    *) echo "Usage: bash scripts/check_install.sh [--python PATH] [--all-cases]" >&2; exit 2 ;;
  esac
done

set -a
if [[ -f "${project_root}/.env.local" ]]; then
  source "${project_root}/.env.local"
else
  source "${project_root}/configs/env.example"
fi
set +a

export ASTEVOLVE_PROJECT_ROOT="${project_root}"
export ASTEVOLVE_RUNTIME_ROOT="${ASTEVOLVE_RUNTIME_ROOT:-${project_root}/../DASTevolve_runtime}"
export PYTHONPATH="${project_root}:${project_root}/outerloop${PYTHONPATH:+:${PYTHONPATH}}"

demo_manifest="${project_root}/cases/demo_case/case.json"
tiam1_manifest="${project_root}/cases/tiam1_pdz_caspr4_vs_sdc1_selectivity/case.json"
manifests=(
  "${demo_manifest}"
)
case_names=(demo)
if [[ ${all_cases} -eq 1 ]]; then
  manifests=("${project_root}"/cases/*/case.json)
  case_names=(demo tiam1 asyn cab aph)
fi
for manifest in "${manifests[@]}"; do
  "${python_bin}" "${project_root}/scripts/check_assets.py" \
    --manifest "${manifest}"
  "${python_bin}" "${project_root}/scripts/preview_case.py" \
    --manifest "${manifest}" >/dev/null
done

for case_name in "${case_names[@]}"; do
  ASTEVOLVE_PYTHON_BIN="${python_bin}" \
    bash "${project_root}/scripts/run_case.sh" \
    "${case_name}" --iterations 1 --dry-run >/dev/null
done

if [[ ${all_cases} -eq 0 ]]; then
  echo "DASTevolve base install check passed: demo preview/assets/launcher."
  echo "Use --all-cases after configuring the external case inputs. No models were executed."
  exit 0
fi

check_tmp_root="${ASTEVOLVE_TMP_ROOT:-${ASTEVOLVE_RUNTIME_ROOT}/tmp}"
mkdir -p "${check_tmp_root}"
preflight_report="$(mktemp "${check_tmp_root}/tiam1-preflight.XXXXXX")"
trap 'rm -f -- "${preflight_report}"' EXIT

"${python_bin}" "${project_root}/scripts/case_preflight.py" run \
  --manifest "${tiam1_manifest}" \
  --report "${preflight_report}" >/dev/null
"${python_bin}" "${project_root}/scripts/case_preflight.py" verify \
  --manifest "${tiam1_manifest}" \
  --report "${preflight_report}" >/dev/null

echo "DASTevolve install check passed: five case previews/assets/launchers and Tiam1 strict preflight are ready"
echo "No sequence or structure model was executed by this check."
