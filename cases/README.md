# Running the four case bundles

Use the Linux environment from the repository README and configure your GPU model environments and OpenAI-compatible endpoint. The four manifests below work with the repository's existing `scripts/run_outer.py` entry point. Paths in this guide are relative to the checkout. [Environment setup](../README.md) · [Environment variables](../configs/env.example)

| Case directory | Models used |
| --- | --- |
| `asyn_c_terminal_hotspot_binder` | Classic ESMFold v1 |
| `tiam1_pdz_caspr4_vs_sdc1_selectivity` | Classic ESMFold v1, Protenix Mini-ESM, ESM-2 and ProGen2-small |
| `ttgr_naltrexone_l1_blind` | Protenix Mini-ESM for the apo and ligand-bound states |
| `cab_lys3_hewl_cdr_recovery` | Protenix-v2 with frozen MSAs and ESM-2 for the VHH reference prior |

The original classic ESMFold worker uses the local `use_modal=False` interface from `biobagel==0.1.14` and `boileroom==0.2.2`. Set `ASTEVOLVE_BAGEL_ROOT` to the external BAGEL directory and `ASTEVOLVE_ESMFOLD_PYTHON` to its Python interpreter. This worker looks for the Hugging Face cache of `facebook/esmfold_v1` under `$ASTEVOLVE_BAGEL_ROOT/models`. [BAGEL interface](https://github.com/softnanolab/bagel/blob/6082512a24e9cbe9d8f7539b09742ad034324301/src/bagel/oracles/folding/esmfold.py) · [BoilerRoom model loading](https://github.com/softnanolab/boileroom/blob/1f37d8362af56469d452e8d1c005a8e1579ac849/boileroom/models/esm/esmfold.py)

For Protenix, set `CONDA_EXE` to the Conda executable, `ASTEVOLVE_PROTENIX_CONDA_ENV` to the name of its separate environment, and both `PROTENIX_ROOT_DIR` and `ASTEVOLVE_PROTENIX_ROOT` to its external model directory. Protenix 2.0.0 supplies the inference API for the mini and v2 checkpoints. TIAM1 also reads `ASTEVOLVE_PROGEN_MODEL_DIR` and `ASTEVOLVE_MASKED_LM_MODEL_DIR`. [Protenix package](https://pypi.org/project/protenix/2.0.0/) · [ProGen2-small](https://huggingface.co/hugohrban/progen2-small) · [ESM-2](https://huggingface.co/facebook/esm2_t6_8M_UR50D)

For cAb-Lys3, the expected checkpoint is `$PROTENIX_ROOT_DIR/checkpoint/protenix-v2.pt`. Its SHA-256 is `8f931f9774a396b67033d0e58628e1834f4a1448165e04254b40a780b0c0d599`. Supply it from a source you are authorized to use. The case includes the reference alignments and features. [Protenix model terms](https://github.com/bytedance/Protenix#license) · [cAb-Lys3 reference inputs](cab_lys3_hewl_cdr_recovery/INPUTS.md)

## Two outer proposals with ten inner iterations

Run the following in a single-GPU allocation after setting `ASTEVOLVE_LLM_API_BASE`, `ASTEVOLVE_LLM_API_KEY` and `ASTEVOLVE_LLM_MODEL`. Existing TtgR installations can export the values of `OPENAI_API_BASE` and `OPENAI_API_KEY` under the corresponding `ASTEVOLVE_LLM_*` names. Set `case_dir` to one of the four directory names above. The runtime, model weights and generated files belong outside the checkout.

```bash
source configs/env.example
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD:$PWD/outerloop${PYTHONPATH:+:$PYTHONPATH}"
export ASTEVOLVE_INNER_ITERATIONS=10 ASTEVOLVE_FORMAL_INNER_ITERATIONS=10
export ASTEVOLVE_OPENEVOLVE_SERIAL=1
case_dir=asyn_c_terminal_hotspot_binder
mkdir -p "$ASTEVOLVE_RUNTIME_ROOT/runs"
run=$(mktemp -d "$ASTEVOLVE_RUNTIME_ROOT/runs/$case_dir.XXXXXXXX")
export ASTEVOLVE_ARTIFACT_ROOT="$run/artifacts" ASTEVOLVE_RUN_ROOT="$run/artifacts"
export ASTEVOLVE_TMP_ROOT="$run/tmp" ASTEVOLVE_MCTS_OUTPUT_DIR="$run/inner"
export ASTEVOLVE_ESMFOLD_TMP="$run/structures/esmfold" ASTEVOLVE_PROTENIX_TMP="$run/structures/protenix"
python scripts/run_outer.py --manifest "cases/$case_dir/case.json" --iterations 2 --output "$run/outer"
```

TIAM1 also provides a reference-structure preflight. Within the GPU allocation, run the following before the outer search to generate its reference structure and check the resulting evidence. To use that checked reference, pass `"$run/tiam1-preflight/case.json"` as the manifest. [TIAM1 evidence sources](tiam1_pdz_caspr4_vs_sdc1_selectivity/DATA_SOURCES.md)

```bash
python -m cases.tiam1_pdz_caspr4_vs_sdc1_selectivity.prepare_inputs --output "$run"
```

The seed program has its own inner search. The outer budget counts proposal slots, so a rejected proposal can leave fewer evaluated children. Each case retains its population-based parent selection. Check the child's program parent and compare its inner root sequence with the selected parent's winning sequence when interpreting a run. Two accepted children can share a parent. A completed search can also return a candidate that misses the case's scientific criteria.
