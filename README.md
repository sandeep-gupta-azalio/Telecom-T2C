# Telecom-T2C-Trainer

Production fine-tuning framework for continue-LoRA (QLoRA) training of
`google/gemma-4-12B-it` on a telecom network-inventory NL query dataset,
via [Unsloth](https://github.com/unslothai/unsloth). Runs on Google Colab
(A100 40GB recommended); the notebook only orchestrates — all logic lives
in `src/`.

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
    model.py           # 4-bit QLoRA base model + LoRA adapter via Unsloth (fresh or continue)
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

This project loads the base model and LoRA adapter exclusively via
[Unsloth](https://github.com/unslothai/unsloth)'s `FastModel` — custom
kernels/patches for a curated set of architectures, confirmed to include the
Gemma 4 family (`unsloth/gemma-4-12b-it` exists on the Hub), typically
cutting VRAM usage substantially for QLoRA versus a plain
transformers+bitsandbytes+peft path. An earlier version of this project also
supported that plain path as a config-selectable fallback; it was removed
once Unsloth was confirmed to be the working path, to keep the
implementation to one code path instead of two.

**Not validated end-to-end on real hardware by this project** (no GPU
available during development) — start with a small `data.max_train_samples`
smoke test before trusting a full run, same practice recommended throughout
this README.

**Known, currently-unresolved upstream risk**: Unsloth builds its LoRA
support on `peft`, and `peft` has previously had zero working PyPI release
for `transformers>=4.55` (see the `ImportError: cannot import name
'BloomPreTrainedModel'` entry in Troubleshooting). `requirements.txt` floors
`peft`/`accelerate`/`trl`/`datasets` just above validated versions rather
than exact-matching them, specifically so `pip install --upgrade` (Section
2) has room to pick up a compatibility fix without a `requirements.txt`
edit. Separately, Unsloth's own `exec()`-based monkeypatching of
transformers internals periodically breaks when transformers renames
something internal (see
[unslothai/unsloth#3415](https://github.com/unslothai/unsloth/issues/3415));
`unsloth`/`unsloth_zoo` are left fully unpinned for the same reason — see
`requirements.txt`'s comments for the full reasoning on each pin.

Implementation notes, if you're reading the code:
- `model.load_base_model()` returns `(model, tokenizer)` together — Unsloth
  configures both in lockstep — and the notebook's Section 7 reassigns its
  `tokenizer` variable to that return value, so everything downstream
  (training, generation) uses the Unsloth-matched tokenizer, not the one
  loaded earlier in Section 4 for dataset statistics.
- `model.attach_lora()` uses `use_gradient_checkpointing="unsloth"` (their
  own offloaded-checkpointing implementation) when
  `training.gradient_checkpointing` is true, else `False` —
  `trainer.build_sft_config()` always disables transformers' own
  gradient_checkpointing at the `SFTConfig` level, since this is the only
  place checkpointing gets configured; enabling both would conflict.
- Never calls `merge_and_unload()` on either the fresh-init or
  continue-adapter path — the adapter always stays separate from the base
  model.

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
model) plus provenance (dataset/LoRA/generator/validator version, git hash)
— built in the notebook's Section 9, so every run is comparable
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

**You've re-run Install + "Restart session" a couple of times already and
keep hitting a *different* missing-symbol `ImportError` each time** (not the
same one repeating) — e.g. `BloomPreTrainedModel`, then `auto_docstring`,
then `AutoProcessor` (a class that's existed in `transformers` since ~2022,
so its absence isn't a real version-gating issue — a strong sign of
something else). **Read this first, before chasing another individual pin:**
"Runtime -> Restart session" only restarts the Python process — it does
**not** reset installed packages. Every `pip install`/`--upgrade` run in this
conversation (and in your own session) is still sitting on disk, and repeated
installs/upgrades across a long session can leave `site-packages` in a
genuinely inconsistent state (partial overwrites, stale `.dist-info`
metadata) that produces different, seemingly-unrelated import errors on each
attempt. Fix: **Runtime -> Disconnect and delete runtime**, reconnect (a
truly fresh VM, not just a fresh Python process), then run Section 2
(Install) once from that clean slate before troubleshooting further — this
resets the ground you're debugging from, rather than layering another fix on
top of an increasingly muddled environment.

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
A genuine version-skew bug, not a Colab environment artifact: `transformers`'
`Trainer` internals call
`self.accelerator.unwrap_model(model, keep_torch_compile=False)`, and that
`keep_torch_compile` parameter doesn't exist in older `accelerate` releases.
`transformers` is exact-pinned (`==5.5.0`, required for Gemma 4's tokenizer —
see below) but `accelerate`, `peft`, `trl`, and `datasets` are deliberately
left floor-only (`accelerate>=1.8`, `peft>=0.19.1`, `trl>=0.15.0`,
`datasets>=3.2`) so pip has room to resolve versions that are actually
compatible with `transformers==5.5.0`, rather than this project guessing and
hand-pinning exact companions. Fix: re-run Section 2's pip-install cell
(it uses `pip install --upgrade` so a floor-pin bump actually applies), then
**Runtime -> Restart session**, then re-run from Section 1.

**`AttributeError: 'list' object has no attribute 'keys'`** inside
`transformers/tokenization_utils_base.py`'s
`_set_model_specific_special_tokens`, while loading the tokenizer (Section
4, `tokenizer.load_tokenizer()`, or Section 7's Unsloth model load, which
constructs its own tokenizer internally).
Confirmed, not a version-gating nuance: `google/gemma-4-12B-it`'s
`tokenizer_config.json` defines `extra_special_tokens` as a **list**
(transformers v5's format), but `transformers` v4.x's
`_set_model_specific_special_tokens` unconditionally calls `.keys()` on it
(a v4-only, dict-shaped assumption) — see
[huggingface/transformers#45376](https://github.com/huggingface/transformers/issues/45376)
and the
[google/gemma-4-E4B-it discussion](https://huggingface.co/google/gemma-4-E4B-it/discussions/17).
**Gemma 4 genuinely requires transformers v5** for this reason —
`requirements.txt` exact-pins `transformers==5.5.0` (a version confirmed to
handle this correctly). `tokenizer.py` also carries a defensive compat shim
(`patch_extra_special_tokens_list_format()`, applied automatically by
`load_tokenizer()` and by both `model.load_base_model()` and
`inference.load_model_for_inference()`, since Unsloth builds its own
tokenizer bypassing `tokenizer.py`) that converts the list to a dict only if
the installed transformers actually hits this exact `AttributeError` — a
no-op on the pinned v5.5.0, where the native list handling is used as-is.
If you still hit this, re-run Section 2 (Install) to make sure
`transformers` actually resolved to `5.5.0` (the version-check cell prints
it) rather than an older cached wheel.

**`ImportError: cannot import name 'BloomPreTrainedModel' from
'transformers'`** at `import peft` (in Section 2's version-check cell, or
anywhere else `peft` gets imported) — **or** the model failing to load with
an unrecognized-architecture / `KeyError` / `ValueError` on `model_type`.
Historically a real, upstream incompatibility, separate from the tokenizer
issue above: older `peft` releases had zero working PyPI release for
`transformers>=4.55` — see
[huggingface/peft#2754](https://github.com/huggingface/peft/issues/2754)
("No working peft version available in PyPI for transformers 4.55+").
`requirements.txt` floor-pins `peft>=0.19.1` specifically because that's the
first release confirmed to import cleanly against `transformers==5.5.0`
(the exact-pinned version this project uses — see the tokenizer entry
above). If you still hit this, `pip` likely resolved an older cached `peft`
wheel: re-run Section 2's pip-install cell (it uses `pip install --upgrade`
so this actually applies), then **Runtime -> Restart session**, then re-run
from Section 1. If it persists even on a genuinely fresh install, that's a
new regression in the `peft`/`transformers` compatibility matrix beyond what
this project has verified — check the issue above for its current state.

**`NameError: name 'auto_docstring' is not defined`** (or any other
non-`ImportError` exception) **while loading the model in Section 7.**
`unsloth/models/_utils.py` does `exec()`-based monkeypatching of
transformers internals at import time, and when its patches don't match the
installed transformers release, it can raise almost any exception type from
inside that `exec()` call — not a clean `ImportError`. `model.load_base_model()`
catches this broadly (not just `ImportError`) and re-raises an actionable
`RuntimeError`.

**Root cause, confirmed by reading unsloth's own `pyproject.toml` directly**
(both `unslothai/unsloth` and `unslothai/unsloth-zoo`): they declare
`transformers>=4.51.3,...,!=5.0.0,!=5.1.0,<=5.5.0` — i.e. **`5.5.0` is the
highest transformers release unsloth has actually declared support for.**
An earlier version of this project's `requirements.txt` exact-pinned
`transformers==5.12.1`, which is *above* that ceiling; pip still installed
it without complaint (that PyPI release's metadata didn't hard-block it),
but unsloth's `_utils.py` patch code — written against transformers source
up to `5.5.0` — didn't know how to handle whatever changed structurally in
later releases (heavier use of the `@auto_docstring` decorator, evidently),
raising this `NameError` at import time. `requirements.txt` now exact-pins
`transformers==5.5.0` specifically to stay within unsloth's own declared
range. If you're on an older clone of this repo still pinning `5.12.1` (or
anything else above `5.5.0`), `git pull` to get the corrected pin, then
re-run Section 2 (Install), then **Runtime -> Restart session**, then
re-run from Section 1. `unsloth`/`unsloth_zoo` themselves stay fully
unpinned (per Unsloth's own recommendation) so a fresh install still picks
up the latest patch set *for* `transformers==5.5.0` — see
[unslothai/unsloth#3415](https://github.com/unslothai/unsloth/issues/3415)
for the general class of bug. Do not raise the `transformers` pin above
`5.5.0` without first re-checking unsloth's `pyproject.toml` to confirm its
ceiling has actually moved.

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

`google/gemma-4-12B-it`'s exact loading path through Unsloth's `FastModel`,
flash-attention support for its hybrid sliding-window/global attention,
correct LoRA `target_modules`, and packing-vs-sliding-window interaction are
all unverifiable without a live run on real hardware. Defenses already
built in: an exact-pinned `transformers==5.5.0` (the version confirmed to
load Gemma 4's tokenizer) + actionable load-failure errors, and Unsloth's own
`target_modules` auto-detection inside `FastModel.get_peft_model()`
(override via `lora.lora_target_modules` if it picks the wrong set).
`training.learning_rate`
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
(missing fields, resume-directory resolution), the GPU profile table
(`detect_gpu_profile` — override handling, unknown-override errors, T4
marginal-capacity warning), trainer initialization (`build_sft_config` field
mapping, including asserting that `SFTConfig.gradient_checkpointing` stays
unconditionally `False` regardless of `training.gradient_checkpointing`,
since Unsloth's `FastModel.get_peft_model(use_gradient_checkpointing=...)`
owns that setting instead — no real `.train()` call), inference
(`build_prompt`, `generate()`'s greedy-decode-then-fallback logic against
fake model/tokenizer stand-ins), and the tokenizer v4/v5
`extra_special_tokens` compat shim (`patch_extra_special_tokens_list_format`
against fake buggy/fixed method stand-ins — the exact real-world
`AttributeError` this guards against is covered by the shim's own logic
tests, not by loading real Gemma 4 weights). Tests requiring an unavailable
package (e.g. `trl`/`torch` if not installed locally) skip cleanly rather
than failing. Unsloth's actual model loading (`model.load_base_model`,
`model.attach_lora`) is not covered by these tests — it needs a GPU and is
unverified by this project (see "Model backend" above); the recommended
validation is a small `data.max_train_samples` smoke test on Colab.
