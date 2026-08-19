# DASTevolve

DASTevolve is a Dual Abstract Syntax Tree (Dual-AST) protein-design framework with an LLM-guided outer loop and a MCTS inner search loop.
This repository contains the core runtime, a teaching demo, case studies, and a practical guide for configuring a new case.

## Requirements

- Linux
- Python 3.10-3.13
- A CUDA-capable GPU
- An OpenAI-compatible LLM endpoint
- External model installations required by the selected case

## Quick start

```bash
git clone https://github.com/YiyangZHA0/DASTevolve.git
cd DASTevolve
bash scripts/bootstrap_linux.sh --extras gpu
cp configs/env.example .env.local
```

Configure the LLM in `.env.local`:

```bash
export ASTEVOLVE_LLM_API_BASE="https://your-endpoint.example/v1"
export ASTEVOLVE_LLM_API_KEY="your-key"
export ASTEVOLVE_LLM_MODEL="your-model"
```

Configure external model and tool paths. Keep every installation outside the
repository:

```bash
export ASTEVOLVE_RUNTIME_ROOT="${PWD}/../DASTevolve_runtime"
export ASTEVOLVE_MODEL_ROOT="${ASTEVOLVE_RUNTIME_ROOT}/models"
export ASTEVOLVE_TOOL_ROOT="${ASTEVOLVE_RUNTIME_ROOT}/tools"
export ASTEVOLVE_PROGEN_MODEL_DIR="${ASTEVOLVE_MODEL_ROOT}/progen2-small"
export ASTEVOLVE_BAGEL_ROOT="${ASTEVOLVE_TOOL_ROOT}/bagel"
export ASTEVOLVE_ESMFOLD_PYTHON="${ASTEVOLVE_BAGEL_ROOT}/.venv-local/bin/python"
export ASTEVOLVE_ESMFOLD_MODEL="${ASTEVOLVE_MODEL_ROOT}/esmfold_v1"
export ASTEVOLVE_PROTENIX_ROOT="${ASTEVOLVE_MODEL_ROOT}/protenix"
export ASTEVOLVE_ENABLE_PROTENIX=1
```

Run the packaged-case validation policy without executing a model:

```bash
source .venv/bin/activate
bash scripts/check_install.sh
bash scripts/run_case.sh tiam1 --iterations 2 --dry-run
```

Run the teaching demo or the Tiam1 case used in our case study:

```bash
bash scripts/run_case.sh demo --iterations 1
bash scripts/run_case.sh tiam1 --iterations 100
```

Outputs are written outside the repository under
`${ASTEVOLVE_RUNTIME_ROOT}/runs` unless `--output PATH` is provided.

## How to create a new case

Copy `cases/demo_case/` to a new directory and update these files:

- `case.json`: set the case ID and the entry program, configuration, and
  evaluator paths.
- `case_sheet.json`: describe the scientific objective, success and failure
  signals, and hypotheses.
- `design_state.json`: define complete chain sequences, zero-based half-open
  mutable spans, fixed residues, the Functional AST, the Structural AST, and
  mapping edges.
- `initial_program.py`: define the LLM-editable Node and amino-acid policy and
  the locked GPU funnel.
- `*_evaluator.py`: define normalized metrics, weights, missing-data policies,
  and immutable hard gates.
- `memory.yaml`: provide read-only seed knowledge. Learned memory is written
  to the runtime output directory.
- `config.yaml`: configure the LLM, specialist islands, generation barrier,
  and memory policy.

Add an alias for the new directory in `scripts/run_case.sh`, then validate it:

```bash
python scripts/preview_case.py --manifest cases/your_case/case.json
python scripts/check_assets.py --manifest cases/your_case/case.json
python scripts/run_outer.py \
  --manifest cases/your_case/case.json \
  --iterations 2 \
  --output "${ASTEVOLVE_RUNTIME_ROOT}/runs/your_case_validation"
```

For a formal case that should meet the same readiness policy as Tiam1, also
declare `recorded_evidence_path` and `preflight` in `case.json`, then run:

```bash
python scripts/case_preflight.py run \
  --manifest cases/your_case/case.json \
  --report /tmp/your_case_preflight.json
python scripts/case_preflight.py verify \
  --manifest cases/your_case/case.json \
  --report /tmp/your_case_preflight.json
```

Case files must use relative paths only. Inject model locations, databases,
licensed tools, and credentials through environment variables.

PyRosetta/Rosetta, getContacts, ESMFold2, and AlphaFold 3 remain optional
backends. Check their availability with:

```bash
python scripts/check_physics_backends.py
```

Users can add and configure their own backends as needed.
