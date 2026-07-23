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
keeps only the raw `messages` column — conversations are never flattened or
split, and `trainer.train()` passes that column straight to `SFTTrainer`,
which applies the chat template itself (see "Model backend" below for why
that matters for loss masking). `evaluator.cypher_exact_match()` keeps its
name for spec consistency, but in practice it compares the generated vs.
gold assistant text (preferring a structural comparison of the parsed
PASS_4 envelope when both sides parse).

**Each conversation row batches multiple query→response turns** (5 in the
real `train_sft_batched.jsonl`/`val_sft_batched.jsonl`, confirmed by
counting assistant turns directly against the file) — so a
`data.max_train_samples` cap of, say, 10,000 rows is really ~50,000
distinct supervised exchanges, not 10,000. Training only computes loss on
the assistant turns' own tokens (`assistant_only_loss=True`, see "Model
backend"), not the repeated system-prompt/deployment-context boilerplate
that precedes every turn — so most of each row's *token count* is still
that shared context, even though the *loss signal* is concentrated on the
5 responses.

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

**Confirmed, worked-around upstream constraint**: `unsloth`/`unsloth_zoo`'s
own PyPI-published metadata hard-caps `transformers<=5.5.0` (confirmed by
downloading and unzipping their actual wheel METADATA, not just reading
GitHub source), but `google/gemma-4-12B-it`'s `gemma4_unified` architecture
is only recognized starting at `transformers==5.10.0`. Section 2 (Install)
does not resolve these two together — it installs the correlated Unsloth
stack (`unsloth`, `unsloth_zoo`, `bitsandbytes`, `accelerate`, `peft`,
`trl`, `triton`, `xformers`) with `--no-deps` in one phase, then
`transformers`/`tokenizers` with `--no-deps` in a separate, later phase —
mirroring
[Unsloth's own official Colab recipe for a newer Gemma 4 variant](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma4_(26B_A4B)-Vision.ipynb)
rather than inventing a workaround from scratch. See `requirements.txt`'s
top comment and the notebook's Section 2 markdown/code for the full,
empirically-verified reasoning (including what breaks if you resolve them
together instead: pip backtracks `unsloth`/`unsloth_zoo` down to an
ancient, pre-Gemma-4 release).

Separately, Unsloth's own `exec()`-based monkeypatching of transformers
internals periodically breaks when transformers renames something internal
(see
[unslothai/unsloth#3415](https://github.com/unslothai/unsloth/issues/3415))
— confirmed directly during this project's development at
`transformers==5.12.1` (`NameError: name 'auto_docstring' is not defined`),
which is why `transformers==5.10.2` is exact-pinned rather than left to
float upward. `peft` has also previously had zero working PyPI release for
`transformers>=4.55` (see the `ImportError: cannot import name
'BloomPreTrainedModel'` entry in Troubleshooting) — the Install cell's
`peft>=0.19.1` floor is specifically the first release confirmed past that.

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
- `utils.disable_unused_transformers_backends()` is called before any
  transformers/unsloth import in `tokenizer.load_tokenizer()` and
  `model.load_base_model()` (and once more, as early as possible, at the end
  of the notebook's Install cell) — it permanently neutralizes transformers'
  own torchaudio/torchao-mediated import paths regardless of whether either
  package is actually installed or working, since this project never uses
  either directly (text-only; always bitsandbytes 4-bit, never TorchAO
  quantization). This is a second line of defense on top of, not a
  replacement for, Section 2's own torchaudio uninstall / torchao version
  floor — see Troubleshooting for the full incident history behind both.
- `evaluator.evaluate_validation()` wraps `trainer.evaluate()` in
  `torch.compiler.set_stance("force_eager")`, forcing eager (non-compiled)
  execution only for that call — training keeps Unsloth's full
  `torch.compile`-based speed, since `gemma4_unified`'s compiled
  attention/RMSNorm modules were confirmed stable during training but not
  during `evaluate()` specifically; see Troubleshooting's
  `InternalTorchDynamoError` entry for the reproduced crash this avoids
  (and why an earlier version of this project instead disabled compilation
  globally via `UNSLOTH_COMPILE_DISABLE`, unnecessarily costing training
  speed too).
- `model.load_base_model()` calls
  `tokenizer.patch_chat_template_for_assistant_masking()` on the returned
  tokenizer, and `trainer.build_sft_config()` sets `assistant_only_loss=True`
  — together these make training compute loss only on the assistant's own
  PASS_0-4 response tokens, not the entire conversation (system prompt +
  deployment-context blob + all prior turns), which is what the plain
  `dataset_text_field="text"` approach this project used before did.
  `google/gemma-4-12B-it`'s own chat template has no `{% generation %}`
  marker (confirmed by downloading and testing it directly), which is what
  TRL's `assistant_only_loss` needs to build its per-token mask — the patch
  inserts that marker around exactly the assistant-content span, verified
  locally (byte-identical rendered text, correct per-turn token spans)
  before being wired in. `trainer.train()` correspondingly passes
  `train_ds`/`eval_ds` to `SFTTrainer` with their native `messages` column
  intact (no more pre-flattening to a `text` field) — required for TRL's
  conversational-format auto-detection to kick in at all. See
  Troubleshooting if the patch ever raises (e.g. Google revises the
  template upstream).
- `trainer.train()` passes `getattr(tokenizer, "tokenizer", tokenizer)` —
  Unsloth's returned `tokenizer` is actually a `Gemma4UnifiedProcessor`
  (Gemma 4 is nominally multimodal — see `inference.py`'s docstring for a
  different bug from the same root fact) — as `processing_class` to
  `SFTTrainer`, not the full processor. Confirmed directly by reading TRL's
  source: it sets `self._is_vlm = True` whenever `isinstance(processing_class,
  ProcessorMixin)`, *unconditionally* (regardless of whether the dataset
  actually contains any images/audio/video), and hard-blocks `packing`,
  `padding_free`, **and** `assistant_only_loss` for VLM mode with a
  `ValueError`. This project never trains on anything but text, so passing
  the processor's own inner tokenizer (confirmed via `AutoProcessor`
  locally: `processor.tokenizer` exists, is a plain `PreTrainedTokenizerBase`
  subclass, not a `ProcessorMixin`) avoids VLM mode entirely — verified
  locally end-to-end (real downloaded `Gemma4UnifiedProcessor`, correct
  `_is_vlm`-relevant `isinstance` result, non-zero `assistant_masks` from
  the inner tokenizer's own `apply_chat_template`). Also confirmed the
  outer processor and its inner tokenizer carry **separate** `chat_template`
  strings (`proc.chat_template is not proc.tokenizer.chat_template`), which
  is why `patch_chat_template_for_assistant_masking()` patches both
  independently rather than assuming one covers the other.
- `trainer.build_sft_config()` sets `packing_strategy="wrapped"` (not
  `SFTConfig`'s default `"bfd"`) — `"bfd"` unconditionally forces
  `padding_free=True` whenever `packing=True` (confirmed in TRL's source),
  and `padding_free` requires FlashAttention 2/3, which this project
  doesn't use (Unsloth's xformers-based kernels instead — confirmed via
  Section 7's load banner). See Troubleshooting for the exact crash this
  avoids.
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

This project never uses `torchaudio` — Section 2 (Install) uninstalls it
unconditionally (`pip uninstall -y -q torchaudio`, run as the last install
step, after all four install phases), which makes
`is_torchaudio_available()` correctly return `False` and skip that code
path entirely. If you're on an older copy of this notebook without that
uninstall step, pull the latest version, or run it manually:
```python
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchaudio"])
```
then **Runtime -> Restart session**, then re-run from Section 1.

(A related but distinct cause of the same symptom: `torch` itself getting
reinstalled with a different CUDA-toolkit build than Colab's pre-installed
torchaudio/torchvision expect. Nothing in this project's install commands
ever lists `torch` at all, and Section 1 prints `torch.__version__` /
`torch.version.cuda` up front so a future mismatch here is visible
immediately.)

**`unsloth`/`unsloth_zoo`: FAILED to import**, with `AttributeError:
'_OpNamespace' '_c10d_functional' object has no attribute
'_wrap_tensor_autograd'` deep in a traceback through
`torchao/dtypes/nf4tensor.py` (via `transformers/quantizers/quantizer_torchao.py`).
The exact same bug class as the `torchaudio` entry above, this time via
`torchao`: any `Auto*` class (`AutoProcessor`, `AutoTokenizer`,
`AutoModelForCausalLM`, ...) transitively imports
`transformers/modeling_utils.py`, which unconditionally imports
`transformers/quantizers/auto.py` (needed for `AutoHfQuantizer`, regardless
of which quantization backend you actually use — this project only ever
uses bitsandbytes 4-bit, never TorchAO directly). `quantizers/auto.py`
unconditionally imports `quantizer_torchao.py`, which itself only imports
`torchao.prototype.safetensors.safetensors_support` when
`is_torchao_available()` is `True` — but, same as the `torchaudio` case,
that check only confirms `torchao` is *present*, not that importing it
actually works. If the installed `torchao` build expects a torch op
signature the installed torch build doesn't register under that name
(`torch.ops._c10d_functional._wrap_tensor_autograd`), that reference raises
`AttributeError`, uncaught.

Unlike `torchaudio`, this project can't just uninstall `torchao` —
`unsloth_zoo`'s own metadata declares `torchao>=0.13.0` as a genuine
dependency of its own code (not just something transformers' quantizer
machinery incidentally imports), and an earlier version of this notebook
that uninstalled it outright risked breaking whatever unsloth_zoo itself
uses it for. Two things fix this together, mirroring
[Unsloth's own official Colab recipe](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma4_(26B_A4B)-Vision.ipynb):
- Section 2's Phase 3 installs `torchao>=0.16.0` explicitly (`--no-deps
  --upgrade`) rather than letting an unconstrained `pip install` pick
  whatever's latest — the theory (matching Unsloth's own choice of floor)
  being that an under-constrained torchao version was the actual cause of
  the mismatch above, not torchao categorically.
- `utils.disable_unused_transformers_backends()` (called at the end of
  Section 2, and defensively again at the top of `tokenizer.load_tokenizer()`
  and `model.load_base_model()`) directly monkeypatches
  `is_torchaudio_available`/`is_torchao_available` in `transformers.utils`
  to unconditionally return `False` — see its docstring in `src/utils.py`.
  This means transformers' own quantizer-chain crash above is neutralized
  *regardless* of whether Phase 3's floor actually works on a given Colab
  image, since this project never routes through TorchAO quantization
  directly anyway. If you're on an older copy of this notebook without
  either fix, pull the latest version, then **Runtime -> Restart session**,
  then re-run from Section 1. (The monkeypatch itself needs no restart if
  you just want to apply it manually right now:
  `from src import utils; utils.disable_unused_transformers_backends()`.)

**`TypeError: Accelerator.unwrap_model() got an unexpected keyword argument
'keep_torch_compile'`** during Section 9 (Train), inside
`transformers.Trainer._wrap_model()`.
A genuine version-skew bug, not a Colab environment artifact: `transformers`'
`Trainer` internals call
`self.accelerator.unwrap_model(model, keep_torch_compile=False)`, and that
`keep_torch_compile` parameter doesn't exist in older `accelerate` releases.
`transformers` is exact-pinned (`==5.10.2`, required for Gemma 4 — see
below), installed via Section 2's Phase 4, while `accelerate`, `peft`, and
`trl` (Phase 2) are deliberately left floor-only (`accelerate>=1.8`,
`peft>=0.19.1`, `trl>=0.15.0`) so re-running Install can still pick up a
newer compatible release without a code edit. Fix: re-run Section 2's
Install cell (every phase uses `--upgrade` so a floor-pin bump actually
applies), then **Runtime -> Restart session**, then re-run from Section 1.

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
**Gemma 4 genuinely requires transformers v5** for this reason — Section 2's
Phase 4 exact-pins `transformers==5.10.2` (a version confirmed to handle
this correctly). `tokenizer.py` also carries a defensive compat shim
(`patch_extra_special_tokens_list_format()`, applied automatically by
`load_tokenizer()` and by both `model.load_base_model()` and
`inference.load_model_for_inference()`, since Unsloth builds its own
tokenizer bypassing `tokenizer.py`) that converts the list to a dict only if
the installed transformers actually hits this exact `AttributeError` — a
no-op on the pinned v5.10.2, where the native list handling is used as-is.
If you still hit this, re-run Section 2 (Install) to make sure
`transformers` actually resolved to `5.10.2` (the version-check cell prints
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
Section 2's Phase 2 floor-pins `peft>=0.19.1` specifically because that's
the first release confirmed to import cleanly against
`transformers==5.10.2` (installed in Phase 4 — see the tokenizer entry
above). If you still hit this, `pip` likely resolved an older cached `peft`
wheel: re-run Section 2's Install cell (every phase uses `--upgrade` so
this actually applies), then **Runtime -> Restart session**, then re-run
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

**Root cause, confirmed empirically (not just by reading GitHub source,
which lags behind what's published)**: every `unsloth`/`unsloth_zoo` PyPI
release from `2026.6.1` through `2026.7.4` (the latest at time of writing)
was downloaded and unzipped directly to inspect its wheel METADATA — all of
them declare `transformers>=4.51.3,...,!=5.0.0,!=5.1.0,<=5.5.0` as a real,
pip-enforced dependency, not an optional extra. This crash has been hit
from **both directions** of that ceiling:
- **Too new** (an earlier version of this project's `requirements.txt`
  exact-pinned `transformers==5.12.1`, above the ceiling): pip installed it
  without complaint at the time (a sign the ceiling had drifted, or that
  specific release's resolution didn't hard-block it), but unsloth's
  `_utils.py` patch code — written against transformers source up to
  around `5.5.0` — didn't know how to handle whatever changed structurally
  in later releases (heavier use of the `@auto_docstring` decorator,
  evidently), raising this exact `NameError`.
- **Too old** (a later attempt resolved `unsloth`/`unsloth_zoo` and an
  exact transformers pin *above* 5.5.0 in the same plain
  `pip install -r requirements.txt` call): since no 2026.x
  `unsloth`/`unsloth_zoo` release's metadata allows a transformers version
  above 5.5.0, pip's resolver backtracked all the way down to an ancient
  release from **September 2025** to find *something* that didn't conflict
  — reproduced as `unsloth: FAILED to import (pip-installed version:
  2025.9.5)`, with this exact same `auto_docstring` `NameError`, because
  that ancient release predates Gemma 4 (and the `auto_docstring` pattern)
  entirely.

The actual fix (Section 2, Phases 2-4) is not a version-pin tweak at all —
it's installing the correlated Unsloth stack (`unsloth`, `unsloth_zoo`,
`bitsandbytes`, `accelerate`, `peft`, `trl`, `triton`, `xformers`) together
with `--no-deps` in one phase, then `transformers`/`tokenizers` together
with `--no-deps` in a separate, later phase, so pip's resolver never
attempts to satisfy unsloth's declared ceiling against this project's
actual transformers version at all. This isn't a workaround invented for
this project — it's copied directly from
[Unsloth's own official Colab notebook for a newer Gemma 4 variant](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma4_(26B_A4B)-Vision.ipynb),
adapted here for `google/gemma-4-12B-it` (that notebook pins
`transformers==5.5.0`, correct for *its* model but too old for
`gemma4_unified` — see the next entry). If you're on an older copy of this
notebook using a single flat `pip install -r requirements.txt` for
everything, pull the latest version, then **Runtime -> Restart session**,
then re-run from Section 1. If this crash recurs even with the phased
install, `unsloth`/`unsloth_zoo` themselves are left fully unpinned in
Phase 2 specifically so a fresh install picks up whatever the latest
release actually is — check
[unslothai/unsloth#3415](https://github.com/unslothai/unsloth/issues/3415)
for the general class of bug, and re-verify with `pip download <pkg>==<ver>
--no-deps` + unzipping the wheel's `METADATA` file (the method used to
confirm the above) rather than trusting `pyproject.toml` on GitHub alone.

**`ValueError: The checkpoint you are trying to load has model type
'gemma4_unified' but Transformers does not recognize this architecture`**
(commonly wrapped by Unsloth into `` `google/gemma-4-12B-it` is not
supported yet in `transformers==X.Y.Z`. Please update transformers... ``)
while loading the model in Section 7.
The opposite problem from the entry above: transformers only registered the
`gemma4_unified` architecture (Gemma 4 12B's actual model type) starting at
`transformers==5.10.0` — confirmed directly against
`transformers/models/auto/auto_mappings.py` at each tag (absent at
5.6.0-5.9.0, present at 5.10.0/5.10.1/5.10.2/5.11.0; note this data moved
out of the older `configuration_auto.py` file in transformers' own v5
refactor, so searching the wrong file gives a false "not found"). This is
an exact match for
[unslothai/unsloth#5985](https://github.com/unslothai/unsloth/issues/5985)
("unsloth-zoo pins transformers<=5.5.0 but Gemma 4 12B needs a newer
version"), fixed by unsloth's maintainer in
[unslothai/unsloth#6054](https://github.com/unslothai/unsloth/pull/6054) by
pairing this exact model with `transformers==5.10.2` inside Unsloth
Studio's per-model "sidecar" environments — a mechanism that lives in their
separate desktop app, not in the plain `pip install unsloth` package this
project uses, which is why Section 2's Phase 4 reimplements the *version
pairing* (transformers==5.10.2) via its own `--no-deps` install rather than
reusing Unsloth's installer directly. `google/gemma-4-12B-it`'s own HF repo
ships no `trust_remote_code`/`auto_map` custom modeling code either, so
there's no way to sidestep transformers' built-in architecture
registration — the transformers version genuinely has to be new enough.
If you're on an older copy of this notebook pinning `5.5.0` (or resolving
transformers as part of a single flat `pip install -r requirements.txt`),
pull the latest version, then **Runtime -> Restart session**, then re-run
from Section 1.

**`InternalTorchDynamoError: AcceleratorError: CUDA error: an illegal
memory access was encountered`** during Section 10 (Evaluate), deep inside
`torch._dynamo`'s tracing of Unsloth's compiled
`Gemma4UnifiedTextAttention`/`RMSNorm` forward
(`unsloth_compiled_cache/unsloth_compiled_module_gemma4_unified.py`).
Not a bug in this project's code — `evaluator.evaluate_validation()` is a
one-line wrapper around `trainer.evaluate()`. Two confirmed, related facts
about Unsloth's own bleeding-edge Gemma 4 support:
1. `gemma4_unified` is an extremely recent addition to Unsloth (weeks old
   at time of writing — see the entries above), and its custom-compiled
   attention/RMSNorm modules go through `torch.compile`/dynamo tracing,
   which is exactly where this crash surfaces.
2. Gemma 4's architecture shares KV state across a subset of layers
   (`num_kv_shared_layers`), and there is a separate, confirmed upstream bug
   class where `use_cache=False` — which training with gradient
   checkpointing forces — causes those KV-shared layers to recompute
   incorrectly instead of reusing cached state; serious enough that Unsloth
   shipped a full re-release over it rather than a patch.

`evaluator.evaluate_validation()` wraps `trainer.evaluate()` in
`torch.compiler.set_stance("force_eager")` (a stable PyTorch API,
confirmed to work as a context manager — unlike `torch.compiler.disable()`,
which has a documented bug making it unreliable as one:
[pytorch/pytorch#123771](https://github.com/pytorch/pytorch/issues/123771)),
forcing eager execution for just that call. Training completed successfully
with compilation enabled *before* this crash was ever hit, so the
instability appears scoped to eval mode specifically, not compilation in
general — this fix is deliberately narrow: it costs nothing during
training, where Unsloth's compiled kernels are most of its advertised "2x
faster" speedup and matter most given multi-hour run times. (An earlier
version of this project instead disabled compilation globally via
`UNSLOTH_COMPILE_DISABLE=1` in `model.load_base_model()` — fixing eval
stability the same way, but at the cost of roughly halving training
throughput for the *entire* run just to make brief periodic eval passes
safe. If you're chasing an unexpectedly slow training run, check you're on
a commit with the scoped fix, not the blanket one.)

If you're on an older copy of this repo without this fix, `git pull`. If
you still hit this crash even with the scoped fix: **once "illegal memory
access" occurs, the CUDA context for the rest of that kernel process should
be considered corrupted** — `Runtime -> Restart session` (not just
re-running the cell) before retrying anything, since the error is
asynchronously reported and the actual fault may have occurred earlier
(e.g. during Section 9's training loop, only surfacing here). If it recurs
after a genuine restart, this is an active upstream Unsloth correctness
issue for this specific model, not something to chase further in this
project's code — check
[unslothai/unsloth discussions on Gemma 4](https://github.com/unslothai/unsloth/discussions/4800)
for the current state, and consider setting `evaluation.run_eval: false`
temporarily to let training/saving complete while waiting for an upstream
fix (the adapter still saves in Section 11 regardless of whether Section 10
ran).

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
`disabled` explicitly to silence the warning. If you see `wandb.login()
failed (API key must have 40+ characters, has N.)`, the Colab secret isn't
a real W&B API key (they're always 40 characters) — check
[wandb.ai/authorize](https://wandb.ai/authorize) for your actual key. This
isn't fatal either way; it just falls back to `wandb_mode='offline'`.

**`TypeError: '<' not supported between instances of 'str' and 'float'`**
during Section 9 (Train), inside Unsloth's compiled
`unsloth_compiled_cache/UnslothSFTTrainer.py` (e.g. comparing
`learning_rate < 1e-7`).
Not an Unsloth bug — a YAML gotcha in whatever numeric field you last
edited in `configs/experiment.yaml`. PyYAML's `SafeLoader` does not
recognize bare scientific notation as a float: `learning_rate: 1e-4` (no
decimal point) parses as the **string** `"1e-4"`, not the float `0.0001` —
only `1.0e-4` or a plain decimal like `0.0001` gets recognized. Python
dataclasses don't validate/coerce field types at construction time, so
that string used to flow silently all the way to Unsloth's compiled
`SFTConfig`, crashing on the comparison there instead of at config-load
time. `config.py`'s `_coerce_numeric_fields()` now catches this at
`load_config()` time (Section 3) for every `int`/`float` field across all
config sections, either coercing the numeric-looking string or raising an
actionable `ConfigError` naming the exact field and value if it's not
numeric at all. If you're on an older copy of this repo without this
check, either `git pull`, or just rewrite the offending YAML value with an
explicit decimal point (`1.0e-4`) or plain decimal (`0.0001`) instead of
bare scientific notation.

**`RuntimeError: Could not find the expected assistant-content anchor...`**
from `tokenizer.patch_chat_template_for_assistant_masking()`, during
Section 7 (Load Model).
This means `google/gemma-4-12B-it`'s `chat_template.jinja` has been revised
upstream since this patch was written (its docstring/comment cites the
template's own `Published: 2026-07-09` header — Google does update it,
per that same header's changelog). The patch intentionally raises loudly
here rather than silently leaving `assistant_only_loss=True` broken (which
would otherwise surface later as a much more confusing TRL
`RuntimeError: ...at least one example has no assistant tokens...` from
deep inside `SFTTrainer`'s dataset preparation). Fix: fetch the current
template (`https://huggingface.co/google/gemma-4-12B-it/raw/main/chat_template.jinja`),
find wherever it now renders an assistant/model turn's actual text content,
and update `_GENERATION_MARKER_ANCHOR`/`_GENERATION_MARKER_REPLACEMENT` in
`tokenizer.py` to match the new structure — then re-verify the same way
this was originally verified: load the tokenizer locally, patch it,
confirm `apply_chat_template(..., tokenize=False)` renders byte-identical
text before and after the patch, and confirm
`apply_chat_template(..., return_assistant_tokens_mask=True)` produces
non-zero, correctly-positioned spans. As a temporary unblock, remove
`assistant_only_loss=True` from `trainer.build_sft_config()` and the
`patch_chat_template_for_assistant_masking()` call in
`model.load_base_model()` to fall back to loss-over-the-whole-conversation
(the previous behavior) until the patch is updated.

**`ValueError: Assistant-only loss is not yet supported for
vision-language models. Please set 'assistant_only_loss=False' in the
'SFTConfig'.`** during Section 9 (Train), inside `SFTTrainer.__init__`.
Confirmed by reading TRL's source directly: `SFTTrainer` sets
`self._is_vlm = True` whenever `isinstance(processing_class, ProcessorMixin)`
— unconditionally, regardless of whether the dataset actually has any
images/audio/video — and hard-blocks `assistant_only_loss` (and separately,
`packing` and `padding_free`) in that mode. Unsloth's returned `tokenizer`
for Gemma 4 is a genuine `Gemma4UnifiedProcessor` (`ProcessorMixin`
subclass), since the model is nominally multimodal, which is exactly what
trips this. Fixed in `trainer.train()`: it now passes
`getattr(tokenizer, "tokenizer", tokenizer)` — the processor's own inner,
plain-text tokenizer — as `processing_class` instead of the full processor,
which this project's data (always text-only) never actually needs. If
you're on an older clone still passing the full `tokenizer` directly to
`SFTTrainer`, `git pull`. Note `patch_chat_template_for_assistant_masking()`
patches *both* the outer processor and its inner tokenizer independently
(confirmed locally: they carry separate `chat_template` strings, not a
shared reference) specifically so this fix doesn't silently lose the
generation-marker patch by switching which object gets passed to
`SFTTrainer`.

**`ValueError: When padding_free=True without packing, max_length is not
enforced. Either enable packing..., provide already truncated inputs, or
set max_length=None.`** during Section 9 (Train), inside
`SFTTrainer.__init__` (surfaces through Unsloth's compiled
`UnslothSFTTrainer.__init__`).
Confirmed directly in `trl/trainer/sft_trainer.py`: TRL unconditionally
computes `self.padding_free = args.padding_free or (args.packing and
args.packing_strategy == "bfd")` — meaning `packing=True` with
`SFTConfig`'s *default* `packing_strategy="bfd"` force-enables
`padding_free` regardless of what this project sets for `padding_free`
itself. `padding_free` only works with FlashAttention 2/3, per TRL's own
docs — but this project runs on Unsloth's xformers-based attention
kernels (confirmed via Section 7's load banner printing `FA2 = False`),
not FA2. `trainer.build_sft_config()` now sets `packing_strategy="wrapped"`
instead of the default `"bfd"` — reproduced locally: constructing a real
`SFTConfig` with `packing_strategy="bfd"` computes `padding_free=True` via
TRL's own formula above; `"wrapped"` computes `False`. Tradeoff: `"wrapped"`
packing can occasionally cut an example across a pack boundary (vs.
`"bfd"`'s more careful bin-packing) — a minor quality cost given most
conversations here are well under `max_seq_length`, versus a hard crash.
If you're on an older clone still hitting this, `git pull`.

**`WARNING:trl.trainer.sft_trainer:[RANK 0] The chat template does not
include the assistant turn's end-of-turn token in the loss mask; the model
may not learn to stop.`** during Section 9 (Train) — not an error, just a
warning, and an already-documented, deliberate tradeoff: see "Known
unverified risk areas" below.
`patch_chat_template_for_assistant_masking()`'s generation-marker span
covers exactly the assistant's response text but excludes the turn-closing
`<turn|>` token (see its docstring/comment in `tokenizer.py`) — TRL is
correctly flagging exactly this. If generation runs past a natural stopping
point more than expected once training completes, extending the marker
span to include `<turn|>` for `role == 'model'` is the fix to revisit (a
larger change than the current one-line-anchor patch, since `<turn|>`'s
rendering is shared across all roles in the template, not model-specific
— see `chat_template.jinja`'s `continues_into_next`/closing-tag logic).

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

**`ValueError: Incorrect image source. Must be a valid URL starting with
`http://` or `https://`, a valid path to an image file, or a base64
encoded string. Got <bos><|turn>system...`** during Section 12 (Smoke
Test), inside `transformers/image_utils.py`'s `load_image_as_tensor`.
A real, confirmed bug that was in this project's own `inference.py`, not
an environment issue: Gemma 4 is nominally multimodal, so Unsloth loads
`tokenizer` as a `Gemma4UnifiedProcessor`, whose `__call__` signature is
`(self, images=None, text=None, videos=None, audio=None, **kwargs)` —
`images` comes *first*. `inference.generate()` used to call
`tokenizer(prompt, return_tensors="pt")` with the prompt string
**positional**, which silently bound it to `images` instead of `text`; the
processor then tried to interpret the entire formatted chat prompt as an
image URL/path/base64 string, failing exactly as shown. Fixed by calling
`tokenizer(text=prompt, return_tensors="pt")` with `text` as an explicit
keyword — correct for both a plain `AutoTokenizer` (whose `__call__` also
names its first parameter `text`) and a multimodal processor. If you're on
an older clone with this bug, `git pull`.

**Checkpoint / GPU unavailable / missing dataset errors generally.**
Every module in `src/` raises actionable, specific exceptions (not bare
`Exception`) for these cases — read the message, it names the exact config
field or file path to fix.

### Known unverified risk areas (documented, not hidden)

`google/gemma-4-12B-it`'s exact loading path through Unsloth's `FastModel`,
flash-attention support for its hybrid sliding-window/global attention,
correct LoRA `target_modules`, and packing-vs-sliding-window interaction are
all unverifiable without a live run on real hardware. Defenses already
built in: an exact-pinned `transformers==5.10.2` (the version confirmed to
both recognize Gemma 4's `gemma4_unified` architecture and load its
tokenizer correctly) + actionable load-failure errors, and Unsloth's own
`target_modules` auto-detection inside `FastModel.get_peft_model()`
(override via `lora.lora_target_modules` if it picks the wrong set).
`training.learning_rate`
defaults to `1e-4`, carried over from the reference notebook's
*continue-training* value — for this project's default **fresh** LoRA init,
`2e-4` is more conventional and worth trying if `1e-4` converges too slowly.
`statistics.estimate_training_time()` is a rough heuristic (undocumented
tokens/sec table), not a benchmark — treat it as a ballpark only.
`patch_chat_template_for_assistant_masking()`'s generation-marker span
covers exactly the assistant's response text (verified: decodes back to
precisely the PASS_0-4 content), but deliberately *excludes* the turn's
closing `<turn|>` token — meaning the model gets no direct gradient signal
on learning when to stop each response via this mechanism specifically.
This is a reasonable simplification (Gemma 4 already knows generic
turn-closing conventions from pretraining; this LoRA only needs to relearn
the PASS_0-4 content distribution) but is unverified end-to-end on real
hardware — if generation runs past a natural stopping point more than
before, this is the first place to look.

---

## Testing

```bash
pytest tests/ -v
```

Covers dataset-loader validation (`validate_json`/`validate_messages`/
`validate_roles`, corrupted-line handling), config loading/validation
(missing fields, resume-directory resolution, and `_coerce_numeric_fields`
turning YAML's bare-scientific-notation gotcha, e.g. `learning_rate: 1e-4`
parsing as the string `"1e-4"`, into either a correctly-coerced float or an
actionable `ConfigError` — see Troubleshooting), the GPU profile table
(`detect_gpu_profile` — override handling, unknown-override errors, T4
marginal-capacity warning), trainer initialization (`build_sft_config` field
mapping, including asserting that `SFTConfig.gradient_checkpointing` stays
unconditionally `False` regardless of `training.gradient_checkpointing`,
since Unsloth's `FastModel.get_peft_model(use_gradient_checkpointing=...)`
owns that setting instead — no real `.train()` call), inference
(`build_prompt`, `generate()`'s greedy-decode-then-fallback logic against
fake model/tokenizer stand-ins), the tokenizer v4/v5 `extra_special_tokens`
compat shim (`patch_extra_special_tokens_list_format` against fake
buggy/fixed method stand-ins — the exact real-world `AttributeError` this
guards against is covered by the shim's own logic tests, not by loading
real Gemma 4 weights), and `utils.disable_unused_transformers_backends()`
(asserts it forces the real, installed transformers'
`is_torchaudio_available`/`is_torchao_available` to return `False`
regardless of actual package presence, is idempotent, and tolerates the
extra positional/keyword args `quantizer_torchao.py` actually calls it
with — see "Model backend" above for why this patch exists), and
`patch_chat_template_for_assistant_masking()` (asserts the `{% generation %}`
marker gets correctly inserted around a fake template's assistant-content
anchor, is a no-op when the marker is already present or no template is
set, raises `RuntimeError` when the anchor is missing, and — mirroring the
real `google/gemma-4-12B-it` processor structure confirmed by loading it
locally — patches an outer processor-like object and its separate
`.tokenizer` independently rather than assuming one covers the other;
these test the string-replacement logic in isolation. The actual Jinja
rendering correctness against the real `google/gemma-4-12B-it` template,
and the fact that passing its inner tokenizer as `processing_class` avoids
TRL's VLM detection, were both verified separately, offline, before this
was wired into training: byte-identical rendered text before/after
patching, and `return_assistant_tokens_mask=True` producing per-turn spans
that decode back to exactly the PASS_0-4 assistant
content). Tests requiring an unavailable package (e.g. `trl`/`torch` if not
installed locally) skip cleanly rather than failing. Unsloth's actual model
loading (`model.load_base_model`, `model.attach_lora`) is not covered by
these tests — it needs a GPU and is unverified by this project (see "Model
backend" above); the recommended validation is a small
`data.max_train_samples` smoke test on Colab.
