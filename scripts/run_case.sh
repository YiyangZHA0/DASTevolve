#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/run_case.sh CASE [OPTIONS]

CASE is one of: demo, tiam1 (or the full case directory name).

Options:
  --iterations N  Number of outer iterations (case config default if omitted).
  --output PATH   Output directory (runtime-derived unique path if omitted).
  --dry-run       Print the resolved OuterLoop command without executing it.
  -h, --help      Show this help.

The script sources .env.local when present, otherwise configs/env.example.
Model weights and third-party tools are never downloaded automatically.
EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd "${script_dir}/.." && pwd -P)"

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

case_name="$1"
shift
iterations=""
output=""
dry_run=0

while (($#)); do
  case "$1" in
    --iterations)
      [[ $# -ge 2 ]] || { echo "error: --iterations needs a value" >&2; exit 2; }
      iterations="$2"
      shift 2
      ;;
    --output)
      [[ $# -ge 2 ]] || { echo "error: --output needs a path" >&2; exit 2; }
      output="$2"
      shift 2
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "${case_name}" in
  tiam1|pdz|tiam1_pdz_caspr4_vs_sdc1|tiam1_pdz_caspr4_vs_sdc1_selectivity)
    case_slug="tiam1_pdz_caspr4_vs_sdc1_selectivity"
    ;;
  demo|demo_case)
    case_slug="demo_case"
    ;;
  *)
    echo "error: unknown case: ${case_name}" >&2
    usage >&2
    exit 2
    ;;
esac
case_dir="cases/${case_slug}"

set -a
if [[ -f "${project_root}/.env.local" ]]; then

  source "${project_root}/.env.local"
else

  source "${project_root}/configs/env.example"
fi
set +a

export ASTEVOLVE_PROJECT_ROOT="${project_root}"
export PYTHONPATH="${project_root}:${project_root}/outerloop${PYTHONPATH:+:${PYTHONPATH}}"

runtime_root="${ASTEVOLVE_RUNTIME_ROOT:-${project_root}/../DASTevolve_runtime}"
artifact_root="${ASTEVOLVE_ARTIFACT_ROOT:-${runtime_root}/runs}"
tmp_root="${ASTEVOLVE_TMP_ROOT:-${runtime_root}/tmp}"
mkdir -p "${artifact_root}" "${tmp_root}"

manifest="${project_root}/${case_dir}/case.json"
[[ -f "${manifest}" ]] || { echo "error: missing manifest: ${manifest}" >&2; exit 1; }

if [[ -z "${output}" ]]; then
  run_stamp="$(date -u +%Y%m%dT%H%M%SZ)_${BASHPID}"
  output="${artifact_root}/${case_slug}/${run_stamp}"
fi

args=(
  "${project_root}/scripts/run_outer.py"
  --manifest "${manifest}"
  --output "${output}"
)
if [[ -n "${iterations}" ]]; then
  args+=(--iterations "${iterations}")
fi
if [[ ${dry_run} -eq 1 ]]; then
  args+=(--dry-run)
fi

echo "CASE=${case_slug}"
echo "OUTPUT=${output}"
exec "${ASTEVOLVE_PYTHON_BIN:-python}" "${args[@]}"
