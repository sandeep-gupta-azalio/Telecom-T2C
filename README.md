# Telecom-T2C-Trainer

Production fine-tuning framework for continue-LoRA (QLoRA) training of
`google/gemma-4-12B-it` on a telecom network-inventory NL query dataset.
Runs on Google Colab (A100 40GB recommended); the notebook only orchestrates —
all logic lives in `src/`.

**Adapters are never merged into the base model**, on any run (fresh init or
continuation). LoRA weights are always kept as a separate adapter.

---

## Project structure

```
Telecom-T2C/
  notebooks/Telecom_T2C_Trainer_v2.ipynb   # orchestration only, no business logic
  src/                                     # all real logic lives here
    config.py       # the only place YAML is parsed
    dataset.py       # DatasetLoader: load/validate train/val/golden splits
    statistics.py     # token/turn statistics, histograms, time estimates
    tokenizer.py       # tokenizer + HF token loading
    model.py           # 4-bit QLoRA base model + LoRA adapter (fresh or continue)
    trainer.py           # TRL SFTTrainer orchestration
    callbacks.py           # TrainerCallback subclasses (wandb wiring)
    evaluator.py             # validation/golden eval, PASS_0-4 metric interfaces
    benchmark.py               # post-training benchmark report
    inference.py                 # reload + generate (with a decode-bug workaround)
    wandb_logger.py                # no-op-safe Weights & Biases gateway
    checkpoint.py                    # checkpoint discovery/cleanup, Drive sync
    manifest.py                        # run/adapter provenance manifest.json
    utils.py                             # logging, GPU detection, JSONL I/O, git hash
  tests/                                # pytest smoke tests (CPU-safe, run locally)
  configs/experiment.yaml                # the ONLY file you should need to edit
  requirements.txt
  README.md (this file)
```

**Dependency layering** (bottom depends on nothing above it — no circular imports):

```
utils -> config -> {manifest, tokenizer -> statistics -> dataset, model,
checkpoint, wandb_logger, evaluator -> callbacks -> inference -> trainer -> benchmark}
```

---

## Dataset format

**Read this before you run anything — it differs from what "Text-to-Cypher"
might suggest.**

Each line of `dataset/phase1/train_sft_batched.jsonl` /
`val_sft_batched.jsonl` is one complete multi-turn conversation:

```json
{"messages": [
  {"role": "system", "content": "You are a GPON network inventory query compiler..."},
  {"role": "user", "content": "## Deployment context\n\nproduct_families:\n  OLT:\n..."},
  {"role": "user", "content": "## Query\nPull up device at 10.147.48.25"},
  {"role": "assistant", "content": "PASS_0\nNormalization\n(none)\n\nPASS_1\n...\n\nPASS_4\n{\"status\": \"SUCCESS\", ...}"},
  {"role": "user", "content": "## Query\nWhich subscribers are on ALABAMA-23"},
  {"role": "assistant", "content": "..."}
]}
```

Each assistant turn is a fixed **five-pass structured reply**, not a Cypher
query:

| Pass | Content |
|---|---|
| PASS_0 | Normalization — spelling/token fixes |
| PASS_1 | Lexical Detection — quoted verbatim phrases |
| PASS_2 | Intent — one canonical operation (LOOKUP/LIST/TRACE/COUNT/...) |
| PASS_3 | Semantic Resolution — YAML semantic record |
| PASS_4 | TIR envelope JSON (`status`, `operation`, `subject`, `qualifiers`, ...) |

**There is no literal Cypher text anywhere in this dataset.** `DatasetLoader`
feeds the `messages` array straight into `tokenizer.apply_chat_template()`
with no reformatting — conversations are never flattened or split.
`evaluator.cypher_exact_match()` keeps its name for spec consistency, but in
practice it compares the generated vs. gold assistant text (preferring a
structural comparison of the parsed PASS_4 envelope when both sides parse).

`data.golden_path` in `configs/experiment.yaml` is optional and unset by
default — `DatasetLoader.load_golden()` returns `None` and logs an info
message rather than failing when it's missing.

---

## Installation

### Local (for running `pytest tests/` only — no GPU needed)

```bash
pip install -r requirements.txt
pytest tests/ -v
```

GPU-dependent tests (actual model loading/training) are skipped
automatically via `pytest.importorskip` / `torch.cuda.is_available()` checks
when a GPU or the relevant package (`trl`, etc.) isn't present.

### Colab (for actually training)

1. Upload or `git clone` this repository into your Colab environment, e.g.:
   ```
   !git clone <your-repo-url> /content/Telecom-T2C
   %cd /content/Telecom-T2C
   ```
2. Open `notebooks/Telecom_T2C_Trainer_v2.ipynb` in Colab.
3. **Runtime -> Change runtime type -> A100 GPU** (Colab Pro/Pro+).
4. Edit `configs/experiment.yaml` for your data/adapter paths (see below).
5. Run all cells top to bottom.

---

## Colab setup

- **GPU**: A100 40GB recommended (the notebook's Section 1 "Runtime Check"
  raises immediately if no GPU is detected, and warns if it detects
  something other than A100). L4/T4 are supported with reduced defaults —
  see `src/model.py`'s `detect_gpu_profile()` — but a 12B model in 4-bit
  QLoRA may be marginal on a 16GB T4.
- **HF token** (optional, only needed for gated models): Colab **Secrets**
  panel -> add a secret named `HF_TOKEN` (or whatever `model.hf_token_env_var`
  is set to in the config). Falls back to anonymous download if unset.

---

## Google Drive setup

Set `drive.google_drive_directory` in `configs/experiment.yaml` (defaults to
`/content/drive/MyDrive/telecom_t2c`). The notebook's Configuration section
(3) auto-mounts Drive via `utils.mount_google_drive()` if this is set; the
Save section (11) auto-creates `<google_drive_directory>/<run_name>/` and
copies `adapter/`, `manifest.json`, `config.yaml`, `metrics/`, and
`predictions/` into it. Set `drive.google_drive_directory: null` to disable
Drive entirely — training will still work, you'll just need to download the
adapter zip manually.

If you intend to **continue** training from a prior adapter, that adapter
directory (with `adapter_config.json`, `adapter_model.safetensors`, and
ideally a `manifest.json`) needs to already exist at the path you put in
`model.continue_adapter` — this project does not upload one for you.

---

## Weights & Biases setup

Set `wandb.wandb_project` (and optionally `wandb.wandb_entity`) in the
config. Provide your API key via Colab **Secrets** as `WANDB_API_KEY` (or
set the `WANDB_API_KEY` environment variable directly). If no key is found,
`WandbLogger` automatically falls back to `wandb_mode="offline"` rather than
blocking; if the `wandb` package itself isn't installed or `wandb.init()`
fails for any reason, every other module keeps working — `WandbLogger` is
the single no-op-safe gateway all logging goes through, so a wandb outage
never aborts a training run.

Logged: train/eval loss, learning rate, grad norm, GPU utilization/memory,
examples/sec, tokens/sec, ETA, epoch, step, and run metadata (dataset
version, LoRA version, generator/validator version, git hash). Uploaded as
artifacts at the end of a run: manifest, adapter, predictions, metrics,
config.

---

## Running training

1. Edit `configs/experiment.yaml` — at minimum, check `data.train_path` /
   `data.val_path` point at real files (Drive paths in Colab).
2. Run `notebooks/Telecom_T2C_Trainer_v2.ipynb` top to bottom.
3. By default (`model.continue_adapter: null`), this is a **fresh LoRA
   init** run — there is no prior adapter for `google/gemma-4-12B-it` to
   continue from yet. To continue a later run from this one's output, set
   `model.continue_adapter` to the resulting `outputs/runs/<run>/adapter/`
   path (or its Drive-synced copy).

### Recommended first run

Because `google/gemma-4-12B-it` was released after this project's authoring,
several loading assumptions (exact `AutoModelForCausalLM` compatibility,
flash-attention support for its hybrid attention pattern, LoRA
`target_modules`) are unverified — see **Troubleshooting** below. Before
committing to a multi-hour run, set `data.max_train_samples: 50` in the
config and do one short pass purely to confirm the model loads, trains a
few steps, and saves/reloads correctly. Then set it back to `null` (or your
real cap) for the full run.

---

## Resuming

Set `reproducibility.resume_training: true` (the default). `config.py`'s
`resolve_run_dir()` automatically finds the most recent
`outputs/runs/run_*/` directory that already has a checkpoint under
`adapter/checkpoint-*/` and reuses it — training then resumes from that
checkpoint via `checkpoint.resolve_resume_path()`. To resume a *specific*
run instead of "the most recent one," set `reproducibility.run_id` to that
run's directory name (e.g. `run_20260721_140000`). Set
`reproducibility.resume_training: false` to always start a brand new run
directory regardless of what's already there.

---

## Evaluating

- **Loss-based validation**: runs automatically during training if
  `evaluation.run_eval: true` and a validation/golden split is available
  (`data.eval_source` selects which). Section 10 of the notebook also calls
  `evaluator.evaluate_validation()` explicitly after training finishes.
- **Golden generation-eval**: only runs if `data.golden_path` is set and the
  file exists — otherwise skipped gracefully. Produces
  `outputs/runs/<run>/predictions/golden_predictions.jsonl` and an
  `exact_match_rate` in the benchmark report.
- **PASS_0 - PASS_4 metrics**: `evaluator.PASS_METRIC_STUBS` are
  **interfaces only** — each raises `NotImplementedError` with a docstring
  naming the dataset block it corresponds to. `benchmark.py`'s report
  records their status as `"not_implemented"` for transparency. Implement
  real comparators here when ready; nothing else in the pipeline depends on
  them being implemented yet.
- Standalone re-benchmark of a saved adapter: call `benchmark.run_benchmark()`
  directly with a config, an adapter directory, and (optionally) a
  pre-loaded golden dataset.

---

## Troubleshooting

**Pulled a `requirements.txt` fix but Colab is still failing the same way.**
Editing this repo on GitHub does not change anything in an already-open
Colab session — Colab has its own copy of the files from whenever you last
cloned/uploaded them. Get the updated files into Colab first (re-run your
`git clone`/`git pull`, or re-upload), *then* re-run Section 2 (Install).
Also note: Section 2's pip-install cell runs `pip install --upgrade`
specifically so that re-running it actually applies a loosened version
constraint (e.g. `peft>=0.14.0`) instead of silently leaving an
already-installed version in place because it technically still "satisfies"
the constraint — if you're on an older copy of this notebook without
`--upgrade` in that cell, add it, or just re-clone.

**`ValueError: numpy.dtype size changed, may indicate binary incompatibility`
during Section 2 (Install), usually while importing `datasets`.**
Colab's base image ships numpy 2.x with pandas/pyarrow already compiled
against it. `requirements.txt` intentionally leaves `numpy` floor-only
(`numpy>=1.26,<3`, not exact-pinned) so `pip install` doesn't force a
downgrade that breaks those already-compiled binaries mid-kernel-session —
if you still hit this, it means pip changed numpy/pandas/pyarrow versions in
this already-running process. Fix: re-run Section 2's pip-install cell, then
**Runtime -> Restart session**, then re-run the notebook from Section 1. The
Section 2 version-check cell reports every package's import status (not
just the first failure) and prints this same hint automatically if anything
fails.

**`peft: FAILED — Could not import module 'X'. Are this object's
requirements defined correctly?`** (commonly `'BloomPreTrainedModel'` or
another per-architecture class), or **`RuntimeError: Detected that PyTorch
and TorchAudio were compiled with different CUDA versions`.**
This looks like a peft problem but usually isn't — it's peft's own import
chain (`from transformers import BloomPreTrainedModel` in
`peft/utils/constants.py`) transitively pulling in a `transformers`
audio-loss module that does `import torchaudio`, and *that* is what's
actually failing. Root cause: `torch` was reinstalled with a different
CUDA-toolkit build than Colab's pre-installed `torchaudio`/`torchvision`
expect. `requirements.txt` deliberately never lists `torch` at all — Colab's
pre-installed build is already correctly matched to its own driver and to
torchaudio/torchvision, and pinning/upgrading torch (even loosely) risks
exactly this mismatch. Section 1 (Runtime Check) now prints the active
`torch.__version__` / `torch.version.cuda` up front so this is visible
immediately rather than surfacing later as a confusing peft error. If you
still hit this, check whether anything (a stale `requirements.txt`, a manual
`%pip install torch==...` cell, or another notebook run earlier in the same
session) reinstalled torch — remove that, then **Runtime -> Restart
session**, then re-run from Section 1.

**Model fails to load with an unrecognized-architecture / `KeyError` /
`ValueError` on `model_type`.**
`google/gemma-4-12B-it` postdates this project's `transformers` floor pin.
Upgrade: `pip install -U transformers tokenizers huggingface_hub`, then
re-run Section 2 (Install) and Section 7 (Load Model). `model.py`'s
`load_base_model()` also attempts an `AutoModelForImageTextToText` fallback
automatically before raising.

**OOM during training (especially on L4/T4).**
Lower `training.batch_size` and raise `training.gradient_accumulation` to
keep the same effective batch size; check `model.detect_gpu_profile()`'s
recommendations (printed in Section 5/7 of the notebook). Also consider
lowering `data.max_seq_length`. As a last resort, try
`training.packing: false` first if generation quality looks corrupted at
conversation boundaries post-training (packing + Gemma 4's sliding-window
attention interaction is unverified — see below).

**wandb not logging / "No WANDB_API_KEY found".**
Training continues regardless — `WandbLogger` is designed to never block.
Add the `WANDB_API_KEY` Colab secret or set `wandb.wandb_mode: offline` /
`disabled` explicitly to silence the warning.

**`continue_adapter` path not found.**
`config.validate_config()` prints a warning at Configuration time (Section
3) if the path doesn't exist yet — this is expected if you haven't uploaded
that adapter to Drive. Either upload it, or leave `continue_adapter: null`
for a fresh LoRA init.

**Corrupted JSONL line in the dataset.**
`DatasetLoader` streams and validates the source file line-by-line; a
malformed JSON line, a record missing `messages`, or an invalid role
sequence is dropped (not fatal) and counted in the printed validation
report (`invalid_reasons` breakdown, first ~50 dropped line numbers).

**Generation looks garbled / repeats / never stops.**
`inference.generate()` tries a manual greedy-decode loop first (a
documented workaround for a known Gemma+PEFT+transformers `model.generate()`
bug) and falls back to `model.generate(do_sample=False)` on any exception,
logging a warning either way — check the logs to see which path ran.

**Checkpoint / GPU unavailable / missing dataset errors generally.**
Every module in `src/` raises actionable, specific exceptions (not bare
`Exception`) for these cases — read the message, it names the exact config
field or file path to fix.

### Known unverified risk areas (documented, not hidden)

`google/gemma-4-12B-it`'s exact `AutoModelForCausalLM` loading path,
flash-attention support for its hybrid sliding-window/global attention,
correct LoRA `target_modules`, and packing-vs-sliding-window interaction are
all unverifiable without a live run on real hardware. Defenses already
built in: a floor-pinned `transformers` + actionable load-failure errors, an
auto-detect fallback for `target_modules` (override via
`lora.lora_target_modules` if the auto-detected set is wrong), and an
`attn_implementation: auto` -> `sdpa` fallback. `training.learning_rate`
defaults to `1e-4`, carried over from the reference notebook's
*continue-training* value — for this project's default **fresh** LoRA init,
`2e-4` is more conventional and worth trying if `1e-4` converges too slowly.
`statistics.estimate_training_time()` is a rough heuristic (undocumented
tokens/sec table), not a benchmark — treat it as a ballpark only.

---

## Testing

```bash
pytest tests/ -v
```

Covers dataset-loader validation (`validate_json`/`validate_messages`/
`validate_roles`, corrupted-line handling), config loading/validation
(missing fields, resume-directory resolution), model-loading logic
(`resolve_target_modules`, GPU profile table, attention-implementation
resolution — not actual 4-bit weight downloads), trainer initialization
(`build_sft_config` field mapping — no real `.train()` call), and inference
(`build_prompt`, `generate()`'s greedy-decode-then-fallback logic against
fake model/tokenizer stand-ins). Tests requiring an unavailable package
(e.g. `trl` if not installed locally) or a GPU skip cleanly rather than
failing.
