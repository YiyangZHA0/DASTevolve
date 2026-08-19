#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/bootstrap_linux.sh [OPTIONS]

Create a Python environment and a repository-external DASTevolve runtime.
This command installs Python packages only. It does not download models,
databases, containers, or third-party scientific tools.

Options:
  --runtime-root PATH  External runtime root (default: ../DASTevolve_runtime).
  --venv PATH          Virtual environment path (default: .venv).
  --python PATH        Python 3.10-3.13 interpreter (default: python3).
  --extras LIST        Optional extras (default: none; use gpu only on a
                       prepared CUDA machine).
  --skip-install       Reuse an already installed environment.
  --skip-checks        Do not run doctor and compile-only install checks.
  -h, --help           Show this help.
EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd "${script_dir}/.." && pwd -P)"
runtime_root="${project_root}/../DASTevolve_runtime"
venv_path="${project_root}/.venv"
python_bin="${ASTEVOLVE_BOOTSTRAP_PYTHON:-python3}"
extras="none"
skip_install=0
skip_checks=0

while (($#)); do
  case "$1" in
    --runtime-root)
      [[ $# -ge 2 ]] || { echo "error: --runtime-root needs a path" >&2; exit 2; }
      runtime_root="$2"
      shift 2
      ;;
    --venv)
      [[ $# -ge 2 ]] || { echo "error: --venv needs a path" >&2; exit 2; }
      venv_path="$2"
      shift 2
      ;;
    --python)
      [[ $# -ge 2 ]] || { echo "error: --python needs a path" >&2; exit 2; }
      python_bin="$2"
      shift 2
      ;;
    --extras)
      [[ $# -ge 2 ]] || { echo "error: --extras needs a value" >&2; exit 2; }
      extras="$2"
      shift 2
      ;;
    --skip-install)
      skip_install=1
      shift
      ;;
    --skip-checks)
      skip_checks=1
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

command -v "${python_bin}" >/dev/null 2>&1 || {
  echo "error: Python interpreter not found: ${python_bin}" >&2
  exit 1
}

"${python_bin}" - <<'PY'
import sys

if not ((3, 10) <= sys.version_info[:2] < (3, 14)):
    raise SystemExit(
        f"DASTevolve requires Python 3.10-3.13; found {sys.version.split()[0]}"
    )
PY

mkdir -p "${runtime_root}"
runtime_root="$(cd "${runtime_root}" && pwd -P)"
case "${runtime_root}/" in
  "${project_root}/"*)
    echo "error: --runtime-root must be outside the Git checkout" >&2
    exit 1
    ;;
esac

mkdir -p \
  "${runtime_root}/cases" \
  "${runtime_root}/models" \
  "${runtime_root}/tools" \
  "${runtime_root}/datasets" \
  "${runtime_root}/containers" \
  "${runtime_root}/runs" \
  "${runtime_root}/tmp"

if [[ ! -x "${venv_path}/bin/python" ]]; then
  "${python_bin}" -m venv "${venv_path}"
fi
venv_path="$(cd "${venv_path}" && pwd -P)"

source "${venv_path}/bin/activate"

if [[ ${skip_install} -eq 0 ]]; then
  python -m pip install --upgrade pip
  if [[ "${extras}" == "none" || -z "${extras}" ]]; then
    install_target="${project_root}"
  else
    install_target="${project_root}[${extras}]"
  fi
  python -m pip install -e "${install_target}"
  python -m pip check
fi

env_file="${runtime_root}/env.sh"
if [[ ! -e "${env_file}" ]]; then
  {
    printf 'export ASTEVOLVE_PROJECT_ROOT=%q\n' "${project_root}"
    printf 'export ASTEVOLVE_RUNTIME_ROOT=%q\n' "${runtime_root}"
    printf 'source %q\n' "${project_root}/configs/env.example"
  } >"${env_file}"
  chmod 600 "${env_file}"
fi

source "${env_file}"

if [[ ${skip_checks} -eq 0 ]]; then
  python "${project_root}/scripts/doctor.py" --strict
  bash "${project_root}/scripts/check_install.sh" --python "${venv_path}/bin/python"
fi

cat <<EOF

DASTevolve bootstrap complete.

Activate this deployment with:
  source "${venv_path}/bin/activate"
  source "${env_file}"

External models/tools were not downloaded. See:
  ${project_root}/resources/model-assets.example.json
  ${project_root}/resources/external-tools.lock.json
EOF
