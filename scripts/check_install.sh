#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd "${script_dir}/.." && pwd -P)"
python_bin="${ASTEVOLVE_PYTHON_BIN:-python}"

if [[ ${1:-} == "--python" ]]; then
  [[ $# -ge 2 ]] || { echo "error: --python needs a path" >&2; exit 2; }
  python_bin="$2"
  shift 2
fi
[[ $# -eq 0 ]] || { echo "error: unexpected arguments: $*" >&2; exit 2; }

export ASTEVOLVE_PROJECT_ROOT="${project_root}"
export ASTEVOLVE_RUNTIME_ROOT="${ASTEVOLVE_RUNTIME_ROOT:-${project_root}/../DASTevolve_runtime}"
export PYTHONPATH="${project_root}:${project_root}/outerloop${PYTHONPATH:+:${PYTHONPATH}}"

demo_manifest="${project_root}/cases/demo_case/case.json"
tiam1_manifest="${project_root}/cases/tiam1_pdz_caspr4_vs_sdc1_selectivity/case.json"
manifests=(
  "${demo_manifest}"
  "${tiam1_manifest}"
)
for manifest in "${manifests[@]}"; do
  "${python_bin}" "${project_root}/scripts/preview_case.py" \
    --manifest "${manifest}" >/dev/null
  "${python_bin}" "${project_root}/scripts/check_assets.py" \
    --manifest "${manifest}" >/dev/null
done

for case_name in demo tiam1; do
  ASTEVOLVE_PYTHON_BIN="${python_bin}" \
    bash "${project_root}/scripts/run_case.sh" \
    "${case_name}" --iterations 1 --dry-run >/dev/null
done

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

echo "DASTevolve install check passed: demo preview/assets/launcher and Tiam1 strict preflight are ready"
echo "No sequence or structure model was executed by this check."
