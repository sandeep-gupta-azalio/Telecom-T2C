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

## Model backend

`model.backend` in `configs/experiment.yaml` selects how the base model and
LoRA adapter are loaded:

| Backend | Default? | What it is |
|---|---|---|
| `unsloth` | Yes | Custom kernels/patches for a curated set of architectures — confirmed to include the Gemma 4 family (`unsloth/gemma-4-12b-it` exists on the Hub) as of this project's authoring. Typically cuts VRAM usage substantially for QLoRA versus the plain path below. |
| `transformers` | Fallback | The original, proven-working path (plain `transformers` + `bitsandbytes` + `peft`) that this whole project was built and debugged against. |

**`unsloth` has not been validated on real hardware by this project** — there
was no GPU available during development, so this path was written against
Unsloth's documented API shape but never run end-to-end. Given how many
rounds of environment-specific breakage this project already hit getting the
*plain* stack working on Colab (see Troubleshooting below), treat the same
practice as load-bearing here too: **start with a small
`data.max_train_samples` smoke test** before a full run. If `unsloth` hits
its own compatibility issue, set `model.backend: transformers` in
`configs/experiment.yaml` and re-run from Section 3 (Configuration) — no
code changes needed, the plain path is fully preserved, not deleted.

Implementation notes, if you're reading the code:
- `model.load_base_model_for_backend()` / `model.attach_lora_for_backend()`
  dispatch on this field; `inference.load_model_for_inference_for_backend()`
  does the same for post-training reload.
- The `unsloth` backend returns a tokenizer from the model-loading call
  (Unsloth configures both together) — the notebook's Section 7 reassigns
  its `tokenizer` variable to that return value, so everything downstream
  (training, generation) uses the Unsloth-matched tokenizer, not the one
  loaded earlier in Section 4 for dataset statistics.
- `unsloth`'s fresh-LoRA-init path uses `use_gradient_checkpointing="unsloth"`
  (their own offloaded-checkpointing implementation) instead of
  transformers' generic gradient checkpointing — `trainer.build_sft_config()`
  knows not to also enable the latter on this backend, to avoid
  double-configuring checkpointing.
- Both backends still never call `merge_and_unload()` — the adapter stays
  separate from the base model either way.

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
set the `WANDB_API_KEY` environment variable directly, e.g. outside Colab).
`WandbLogger.init()` follows the login pattern wandb/Colab recommend for
notebooks — fetch the key and call `wandb.login(key=...)` explicitly, rather
than relying on `wandb.init()` to discover credentials implicitly:

```python
import wandb
from google.colab import userdata

wandb_key = userdata.get("WANDB_API_KEY")
wandb.login(key=wandb_key)
```

(see [wandb's Intro to Weights & Biases Colab](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/intro/Intro_to_Weights_%26_Biases.ipynb)
for the reference this follows — also the source for the `config=`,
`job_type=`, and `wandb.summary[...]` conventions below). If no key is
found, `WandbLogger` automatically falls back to `wandb_mode="offline"`
rather than blocking; if the `wandb` package itself isn't installed or
`wandb.init()`/`wandb.login()` fails for any reason, every other module
keeps working — `WandbLogger` is the single no-op-safe gateway all logging
goes through, so a wandb outage never aborts a training run.

**Config** (`wandb.init(config=...)`): the full set of hyperparameters
(learning rate, batch size, LoRA rank/alpha, packing, max_seq_length, base
model, backend) plus provenance (dataset/LoRA/generator/validator version,
git hash) — built in the notebook's Section 9, so every run is comparable
side-by-side in the wandb UI, not just tagged with metadata.

**During training** (via `TrainingCallback`/`EvaluationCallback`/`GPUCallback`):
train/eval loss, learning rate, grad norm, GPU utilization/memory,
examples/sec, tokens/sec, ETA, epoch, step — logged as a time series with
namespaced keys (`train/...`, `eval/...`, `gpu/...`, `system/...`).

**Final summary** (`wandb.summary[...]`, via `WandbLogger.set_summary()`):
final validation metrics and golden exact-match rate, set once at the end of
Section 10 (Evaluate) — these are what shows up as the run's headline stats
when comparing runs in the wandb UI, distinct from the time-series logs
above.

**Artifacts**, uploaded at the end of a run (Section 11): manifest, adapter,
predictions, metrics, config.

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

**`ValueError: numpy.dtype size changed, may indicate binary incompatibility`**
(usually while importing `datasets`/`pandas`), **or
`ImportError: cannot import name '_center' from 'numpy._core.umath'`**
(numpy's own pure-Python and compiled layers out of sync).
Both are the same underlying lesson, learned the hard way: `requirements.txt`
does not list `numpy` at all, on purpose. Colab's pre-installed numpy is
already correctly matched to the pandas/pyarrow wheels shipped alongside it;
even a *loosely* floor-pinned `numpy>=1.26,<3` still gets touched by
`pip install --upgrade` (required elsewhere so other floor-pin bumps
actually apply — see below) and that in-place upgrade can leave numpy in a
broken hybrid state on Colab specifically. Same reasoning as `torch` (also
never listed) — if you still hit either error, something reinstalled numpy
or torch in this session (a stale `requirements.txt`, a manual `%pip
install numpy==...`/`torch==...` cell, or an earlier run in the same
session); remove that, then **Runtime -> Restart session**, then re-run the
notebook from Section 1. The Section 2 version-check cell reports every
package's import status (not just the first failure) and prints a
remediation hint automatically if anything fails.

**`peft: FAILED — Could not import module 'X'. Are this object's
requirements defined correctly?`** (commonly `'BloomPreTrainedModel'` or
another per-architecture class), possibly with an underlying
`AttributeError: partially initialized module 'torchaudio' has no attribute
'lib' (circular import)` or `RuntimeError: Detected that PyTorch and
TorchAudio were compiled with different CUDA versions` deeper in the
traceback.
This looks like a peft problem but isn't — `peft/utils/constants.py` does
`from transformers import BloomPreTrainedModel`, which transitively imports
a `transformers` audio-loss module (`transformers/loss/loss_rnnt.py`) that's
guarded by `if is_torchaudio_available(): import torchaudio`. That guard
only checks whether the `torchaudio` *package is present*, not whether it
actually works — and some Colab images ship a `torchaudio` build with an
internal bug (a circular import inside its own CUDA-version check) that
crashes on that `import torchaudio`, taking down the entire chain that led
to it (including `import peft`, which has nothing to do with audio at all).

This project never uses `torchaudio` — Section 2 (Install) now runs
`pip uninstall -y torchaudio` before installing `requirements.txt`, which
makes `is_torchaudio_available()` correctly return `False` and skip that
code path entirely. If you're on an older copy of this notebook without
that uninstall step, pull the latest version, or run it manually:
```python
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchaudio"])
```
then **Runtime -> Restart session**, then re-run from Section 1.

(A related but distinct cause of the same symptom: `torch` itself getting
reinstalled with a different CUDA-toolkit build than Colab's pre-installed
torchaudio/torchvision expect. `requirements.txt` deliberately never lists
`torch` at all, and Section 1 prints `torch.__version__` /
`torch.version.cuda` up front so a future mismatch here is visible
immediately.)

**`TypeError: Accelerator.unwrap_model() got an unexpected keyword argument
'keep_torch_compile'`** during Section 9 (Train), inside
`transformers.Trainer._wrap_model()`.
A genuine version-skew bug, not a Colab environment artifact this time:
`transformers`' `Trainer` internals call
`self.accelerator.unwrap_model(model, keep_torch_compile=False)`, and that
`keep_torch_compile` parameter doesn't exist in older `accelerate` releases.
Since `transformers` is intentionally left floor-only to support Gemma 4,
`accelerate` needs to float with it too — it's now floor-only in
`requirements.txt` (`accelerate>=1.2.0`) alongside `peft`/`trl`, for the same
reason. Fix: re-run Section 2's pip-install cell, then
**Runtime -> Restart session**, then re-run from Section 1.

**Model fails to load with an unrecognized-architecture / `KeyError` /
`ValueError` on `model_type`.**
`google/gemma-4-12B-it` postdates this project's `transformers` floor pin.
Upgrade: `pip install -U transformers tokenizers huggingface_hub`, then
re-run Section 2 (Install) and Section 7 (Load Model). `model.py`'s
`load_base_model()` also attempts an `AutoModelForImageTextToText` fallback
automatically before raising.

**`OutOfMemoryError: CUDA out of memory` during Section 9 (Train), even on
A100 40GB.**
Two mitigations are already on by default: `attach_lora()` configures
non-reentrant gradient checkpointing (`use_reentrant: False` — generally
holds fewer saved tensors than the older reentrant default, one real lever
against backward-pass OOM) via `prepare_model_for_kbit_training`, matched by
the same `gradient_checkpointing_kwargs` on the `SFTConfig` side; and
`configure_cuda_visible_devices()` sets
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — exactly what PyTorch's
own OOM message suggests when "reserved but unallocated" memory is large
(reduces allocator fragmentation). If you still hit OOM after those:
1. Lower `training.batch_size` and raise `training.gradient_accumulation`
   proportionally to keep the same effective batch size (e.g. 4/4 ->
   2/8 -> 1/16). This is the biggest lever — activation memory scales with
   batch size directly.
2. Lower `data.max_seq_length` (e.g. 1536 -> 1024 or 768). With
   `training.packing: true`, every packed sequence is close to this length,
   so it directly sets the activation-memory floor per step.
3. Check `model.detect_gpu_profile()`'s recommendations (printed in
   Sections 5/7 of the notebook) — they're more conservative on L4/T4 than
   A100 by design.
4. As a last resort, try `training.packing: false` — non-packed batches can
   have shorter average sequence length than a fully-packed
   `max_seq_length`-sized block, at the cost of some padding waste. (This is
   also the first thing to try if generation quality looks corrupted at
   conversation boundaries post-training — packing + Gemma 4's
   sliding-window attention interaction is unverified, see below.)

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
(missing fields, resume-directory resolution, unsloth/transformers backend
validation), model-loading logic (`resolve_target_modules`, GPU profile
table, attention-implementation resolution, backend-dispatch error paths —
not actual 4-bit weight downloads for either backend), trainer
initialization (`build_sft_config` field mapping, including the
backend-aware gradient-checkpointing branch — no real `.train()` call), and
inference (`build_prompt`, `generate()`'s greedy-decode-then-fallback logic
against fake model/tokenizer stand-ins). Tests requiring an unavailable
package (e.g. `trl` if not installed locally) or a GPU skip cleanly rather
than failing. The `unsloth` backend's actual model loading is not covered by
these tests — it needs a GPU and is unverified by this project (see "Model
backend" above); the recommended validation is a small
`data.max_train_samples` smoke test on Colab.
