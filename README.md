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
  notebooks/
    Telecom_T2C_Trainer_v2.ipynb           # training orchestration only, no business logic
    Telecom_T2C_Benchmark.ipynb            # re-run PASS_0-4/exact-match eval against an existing adapter
    Telecom_T2C_Inference_Server.ipynb     # Drive adapter -> ngrok tunnel or GGUF export, for local testing
    Telecom_T2C_Inference_Server_Kaggle.ipynb  # same idea, hosted on Kaggle (Kaggle Dataset instead of Drive)
  src/                                     # all real logic lives here
    config.py       # the only place YAML is parsed
    dataset.py       # DatasetLoader: load/validate train/val/golden splits
    statistics.py     # token/turn statistics, histograms, time estimates
    tokenizer.py       # tokenizer + HF token loading
    model.py           # 4-bit QLoRA base model + LoRA adapter via Unsloth (fresh or continue)
    trainer.py           # TRL SFTTrainer orchestration
    callbacks.py           # TrainerCallback subclasses (wandb wiring)
    evaluator.py             # validation/golden eval, PASS_0-4 parsing + accuracy scoring
    benchmark.py               # post-training benchmark report
    inference.py                 # reload + generate (with a decode-bug workaround)
    server.py                      # FastAPI /generate endpoint for the inference-server notebook
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

Each line of `dataset/phase1/train_sft.jsonl` / `val_sft.jsonl` is one
complete conversation (potentially multi-turn):

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
keeps only the raw `messages` column, and `trainer.train()` flattens each
conversation into a single `text` field via `tokenizer.apply_chat_template()`
(no reformatting of the messages themselves) before handing it to
`SFTTrainer` — see "Model backend" below for why it's flattened rather than
passed through natively. `evaluator.cypher_exact_match()` keeps its name
for spec consistency, but in practice it compares the generated vs. gold
assistant text (preferring a structural comparison of the parsed PASS_4
envelope when both sides parse).

**A conversation row may batch multiple query→response turns** — an earlier
version of this dataset (`train_sft_batched.jsonl`/`val_sft_batched.jsonl`)
packed 5 query/assistant exchanges per row (confirmed by counting assistant
turns directly against that file), so a `data.max_train_samples` cap of,
say, 10,000 rows was really ~50,000 distinct supervised exchanges, not
10,000. The current dataset files (`train_sft.jsonl`/`val_sft.jsonl`, built
unbatched, with far fewer total rows) may have a different rows-to-turns
ratio — re-confirm by counting assistant turns if this matters for your
`max_train_samples` planning; `max_train_samples: null` (this project's
current default) trains on every row regardless. Training currently computes loss
over the *entire* flattened conversation (system prompt + repeated
deployment-context boilerplate + all turns), not just the assistant
responses — an assistant-only-loss masking mode was built and verified to
work correctly in isolation, but reverted before use because combining it
with `packing=True` hit a confirmed, reproduced crash in Unsloth's compiled
`SFTTrainer` that couldn't be resolved without GPU access; see "Model
backend" below for the full story and what's kept in place for a future
retry.

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
- `trainer.train()` flattens each conversation's `messages` into a single
  `text` field via `tokenizer.apply_chat_template()` before handing the
  dataset to `SFTTrainer` (`SFTConfig(dataset_text_field="text")`) — loss is
  computed over the entire flattened conversation (system prompt +
  deployment-context blob + all turns), not just the assistant responses.
  **This project tried and reverted a stricter alternative**: passing
  `train_ds`/`eval_ds` to `SFTTrainer` with their native `messages` column
  intact (letting TRL's conversational-format auto-detection apply the
  chat template itself) plus `SFTConfig(assistant_only_loss=True)`, so loss
  would count only the assistant's own PASS_0-4 response tokens. That part
  worked and was verified correct in isolation — `google/gemma-4-12B-it`'s
  chat template has no `{% generation %}` marker that
  `assistant_only_loss` needs (confirmed by downloading and testing the
  real template), so `tokenizer.patch_chat_template_for_assistant_masking()`
  inserts one around exactly the assistant-content span (byte-identical
  rendered text, correct per-turn token spans, verified locally against the
  real tokenizer both before and after switching which object gets passed
  to `SFTTrainer`). Combining it with `packing=True` hit a
  `ValueError: When padding_free=True without packing, max_length is not
  enforced...` crash, which was initially (incorrectly) attributed to the
  assistant-only-loss + conversational-dataset combination specifically —
  it was reverted on that assumption. **That diagnosis was wrong**: the
  same crash recurred afterward with `assistant_only_loss` fully reverted
  (flattened `text` dataset, no conversational format at all), proving the
  real cause was unrelated to it — see the `padding_free` entry in
  Troubleshooting and the bullet below for the actual root cause and fix
  (an explicit `padding_free=False`, since Unsloth's own wrapper injects a
  default of `True` regardless of dataset format). `assistant_only_loss`
  is not re-enabled by this discovery — it stays off for now simply
  because it hasn't been re-tried since the real fix landed, not because
  of any remaining known incompatibility. `patch_chat_template_for_assistant_masking()`
  is left in place, tested, and ready for that retry; see
  `trainer.py`'s module docstring for where to pick it back up.
- `trainer.train()` passes `getattr(tokenizer, "tokenizer", tokenizer)` —
  Unsloth's returned `tokenizer` is actually a `Gemma4UnifiedProcessor`
  (Gemma 4 is nominally multimodal — see `inference.py`'s docstring for a
  different bug from the same root fact) — as `processing_class` to
  `SFTTrainer`, not the full processor. Confirmed directly by reading TRL's
  source: it sets `self._is_vlm = True` whenever `isinstance(processing_class,
  ProcessorMixin)`, *unconditionally* (regardless of whether the dataset
  actually contains any images/audio/video), and hard-blocks `packing`
  (and, separately, `assistant_only_loss`) for VLM mode with a `ValueError`.
  This project never trains on anything but text, so passing the
  processor's own inner tokenizer (confirmed via `AutoProcessor` locally:
  `processor.tokenizer` exists, is a plain `PreTrainedTokenizerBase`
  subclass, not a `ProcessorMixin`) avoids VLM mode entirely — kept even
  after reverting `assistant_only_loss`, since it's independently correct
  and harmless either way.
- `trainer.build_sft_config()` sets `packing_strategy="wrapped"` (not
  `SFTConfig`'s default `"bfd"`) and explicitly `padding_free=False`.
  Confirmed directly against the actual generated
  `unsloth_compiled_cache/UnslothSFTTrainer.py` (not just the plain `trl`
  package — see Troubleshooting for why that distinction mattered here):
  `"bfd"`/`"bfd_split"` packing without a supported FlashAttention variant
  risks real cross-contamination between packed examples (this project
  uses Unsloth's xformers-based kernels, not FA2/3 — confirmed via
  Section 7's load banner), which `"wrapped"` avoids. Separately, Unsloth's
  own `new_init` wrapper evidently injects `padding_free=True` onto the
  config by default regardless of `packing_strategy` — reproduced with
  `packing_strategy="wrapped"` alone still raising
  `ValueError: When padding_free=True without packing, max_length is not
  enforced...`; setting `padding_free=False` explicitly overrides it. See
  Troubleshooting for the full incident, including an earlier, incorrect
  diagnosis that blamed this on `assistant_only_loss` specifically.
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

1. Open `notebooks/Telecom_T2C_Trainer_v2.ipynb` in Colab — no manual
   `git clone` needed. **Runtime -> Change runtime type -> A100 GPU**
   (Colab Pro/Pro+).
2. Run Section 0 ("Sync Code + Mount Google Drive") first — it clones this
   repo into `/content/Telecom-T2C` on a fresh runtime (or `git pull`s the
   latest commit if the runtime already has it, stashing/restoring any
   local edit to `configs/experiment.yaml` around the pull) and mounts
   Google Drive at `/content/drive`.
3. Edit `configs/experiment.yaml` for your data/adapter paths (see below).
4. Run the rest of the cells top to bottom.

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
`/content/drive/MyDrive/telecom_t2c`). Section 0 mounts Drive at
`/content/drive` up front (before `data.train_path`/`val_path`, which point
under there, are ever read); the Configuration section (3) mounts it again
defensively via `utils.mount_google_drive()` for anyone who skips Section
0 — a no-op if already mounted. The Save section (11) auto-creates
`<google_drive_directory>/<run_name>/` and
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
- **Generation-eval dataset**: uses `data.golden_path` if set and the file
  exists; otherwise **falls back to the validation set** (`val_ds`) so this
  actually produces results with the default config (`golden_path: null`).
  `report.eval_dataset_source` records which one was used (`"golden"` or
  `"val"`). Produces `outputs/runs/<run>/predictions/<source>_predictions.jsonl`
  and an `exact_match_rate` in the benchmark report.
- **PASS_0 - PASS_4 accuracy** (`evaluator.evaluate_passes()`): each pass is
  parsed out of the assistant turn independently —
  `parse_pass0_normalizations` (src/dst pairs, or `[]` for `"(none)"`),
  `parse_pass1_lexemes` (ordered quoted-lexeme list), `parse_pass2_intent`
  (the single canonical operation string), `parse_pass3_semantic` (YAML ->
  dict), `parse_pass4_envelope` (brace-matched JSON, reused from
  `cypher_exact_match`) — then compared for exact equality against the same
  parse of the gold reply. `benchmark_report.json`'s `pass_metrics` gives,
  per pass: `accuracy`, `num_scored`, `num_gold_unparseable` (gold itself
  didn't parse — excluded from the denominator; should be ~0 on real data),
  and `num_prediction_unparseable` (the model produced no parseable section
  for that pass at all, as opposed to a parseable-but-wrong value) — the
  last two distinguish "model output is garbled" from "model output is
  well-formed but incorrect," which a single blended score can't tell apart.
  `evaluator.PASS_METRICS` also exposes each pass as a single
  `(prediction, gold) -> Optional[float]` comparator, for ad-hoc scoring.
- Standalone re-benchmark of a saved adapter: call `benchmark.run_benchmark()`
  directly with a config, an adapter directory, and a golden and/or
  fallback (val) dataset — or use `notebooks/Telecom_T2C_Benchmark.ipynb`
  (below), which does exactly this without re-running training.

## Re-benchmarking without retraining

`notebooks/Telecom_T2C_Benchmark.ipynb` is a minimal-steps notebook for
re-running generation-based eval (`exact_match_rate` + `pass_metrics`)
against an **already-trained** adapter on Google Drive — no dataset
statistics, no fresh model/adapter load beyond what `run_benchmark()` needs
internally, no training. Useful for checking whether a code fix (e.g. an
`inference.py`/prompt-construction change) actually improved generation
quality, without a multi-hour re-run of the full trainer notebook. Sections:
Sync Code + Mount Drive, Runtime Check, Install, Configuration, Locate
Adapter (auto-detects the latest Drive-synced `run_*/adapter/`, or set
`ADAPTER_RUN_OVERRIDE` to a specific run name), Load Validation/Golden
Dataset, Run Benchmark. Writes `benchmark_report.json` and
`<source>_predictions.jsonl` back into the *same* Drive run directory the
adapter came from, alongside `adapter/`.

### Lookup-level robustness check

Training so far only covers phase-1 simple-lookup queries — the aggregate
`pass_metrics` above can't show whether the model stays correct as the
*same* lookup gets phrased with progressively less explicit, more natural
language. Section 7 of `Telecom_T2C_Benchmark.ipynb`
(`evaluator.run_lookup_level_benchmark()`) tests exactly that, across 4
levels: explicit entity + explicit identifier, synonyms, implicit entity
inferred from the identifier's shape, and natural operator language.

**Deliberately does not score against a hand-authored "gold" answer** —
there's no independently-verified ground truth for arbitrary hand-written
queries, and guessing the correct entity/qualifier resolution risks
silently encoding a wrong answer as truth. Instead, within each group (the
same underlying entity+identifier phrased at all 4 levels), **Level 1's
own answer becomes that group's reference**; every other level is checked
against it via `parse_pass4_envelope()`, not an external gold. A group
whose Level 1 doesn't parse leaves the rest of that group unscored (not
silently marked right or wrong). `evaluator.summarize_lookup_level_results()`
reports, per level, how often it matched its own group's Level 1 —
directly answering whether correctness degrades as phrasing gets less
explicit.

## Speeding up generation-based evaluation

Generation-eval (val/golden `exact_match_rate` + `pass_metrics`) was slow
enough to be impractical for iterating on a fix — 256 examples run
sequentially through an uncached, one-at-a-time greedy-decode loop. Four
independent levers, all on by default now:

- **The model is already 4-bit quantized** (`load_in_4bit=True` via
  Unsloth, unchanged since the first training run) — there's no further
  weight-quantization lever to pull; the slowness was elsewhere.
- **`evaluation.fast_decode: true`**: `inference.greedy_decode` keeps a KV
  cache and feeds only the newest token each step, instead of
  re-forwarding the whole sequence every step (`use_cache=False`, the
  original notebook's Gemma+PEFT `model.generate()` workaround, which is
  O(n²) — 512 new tokens over a ~500-token prompt costs ~380x the compute
  of cached decoding). Mathematically identical output (same argmax over
  the same logits), just far less compute. Set `false` to restore the
  original uncached loop if this ever needs debugging in isolation.
- **`evaluation.max_new_tokens_eval: 400`**: lowered from `512` after
  measuring real gold PASS_0-4 response lengths (median ~190 tokens, p99
  ~325, max ~330, from a real `val_predictions.jsonl`) — `512` was nearly
  2x oversized for the actual task and only mattered when generation
  didn't hit EOS early.
- **`evaluation.generation_batch_size: 8`**: `evaluator.generate_predictions()`
  now decodes this many examples per forward-pass batch
  (`inference.generate_batch`/`greedy_decode_batch`) instead of one at a
  time — the single biggest lever, since 256 sequential decode loops
  become 32 batched ones. Prompts are left-padded (required so every row's
  "next token to generate" lines up at the same trailing column) with
  `position_ids` computed explicitly from the attention mask (this
  project's manual decode loop bypasses transformers'
  `prepare_inputs_for_generation`, so nothing else derives them — getting
  this wrong for a left-padded row silently shifts its positions rather
  than crashing, so it's computed the same way transformers' own
  generation utilities do it). A row that reaches EOS is frozen at
  `pad_token_id` for the rest of that batch's steps rather than removed
  from the batch. Lower this if it OOMs on a smaller GPU (a 16GB T4 in
  particular); raise it on more VRAM. Set to `1` to fall back to the
  original one-at-a-time path. `notebooks/Telecom_T2C_Benchmark.ipynb`'s
  `MAX_EVAL_SAMPLES_OVERRIDE` (Section 5) is also worth setting low (e.g.
  `32`) while iterating on a fix, independent of these four — no need to
  wait through a full 256-example run just to check whether a change
  helped.

---

## Testing the fine-tuned adapter from your own PC

### Running inference on your own PC directly (no Colab/Kaggle/ngrok)

If you fine-tuned against a smaller base model (e.g.
`unsloth/gemma-4-E2B-it-qat-q4_0-unquantized` rather than the 12B model the
rest of this doc assumes) and it actually fits your local GPU's VRAM, you
don't need any of the Colab/Kaggle/ngrok machinery below —
`scripts/run_local_inference.py` runs the adapter directly on this machine
via Unsloth:

```bash
python scripts/run_local_inference.py --adapter-dir /path/to/your/adapter
```

This calls `inference.load_model_for_inference()` exactly the way every
other path in this project does — the base model is read from the
adapter's own `adapter_config.json` (`base_model_name_or_path`, recorded
automatically at training time), **not** from `configs/experiment.yaml`'s
`model.base_model` field, so no config edit is needed regardless of which
base model you actually trained against. `--query "..."` runs your own
query instead of the default smoke-test one.

Add `--serve` to also start `src/server.py`'s FastAPI app locally (no
ngrok tunnel needed — both ends are the same machine) so the rest of this
project's tooling works against it unmodified:

```bash
python scripts/run_local_inference.py --adapter-dir /path/to/your/adapter --serve
# prints a bearer token, then serves on http://127.0.0.1:8000
python scripts/run_remote_lookup_benchmark.py --provider openai \
  --base-url http://localhost:8000 --api-token "<printed token>"
```

Unsloth's install is far less proven on native Windows than on Colab/Linux
— expect the same kind of multi-round debugging `requirements.txt`'s
extensive comments document for the Colab path (this project has never
independently verified the exact pip/wheel set that works on Windows).
Known rough edges if `import unsloth` fails locally: plain `triton` doesn't
support Windows at all (`triton-windows` is the fork that does),
`bitsandbytes` needs a recent version for official Windows wheels
(`pip install -U bitsandbytes`), and a native extension build may want
Visual Studio's C++ Build Tools installed. `model.py`'s
`resolve_attn_implementation()` already falls back to `"sdpa"` when
flash-attention isn't importable, which is the common case on Windows.

### Hosting on Colab/Kaggle instead

A 12B model needs far more VRAM than most laptop/desktop GPUs have — even
4-bit QLoRA weights alone are roughly 6GB+, before KV cache/activations.
`notebooks/Telecom_T2C_Inference_Server.ipynb` works around this by keeping
the GPU work on Colab: it mounts Google Drive, auto-detects the most
recently synced `run_*/adapter/` (via
`checkpoint.find_latest_synced_run()`), reloads it through
`inference.load_model_for_inference()` (same code path as the trainer
notebook's Section 12 Smoke Test), and serves a `POST /generate` endpoint
(`src/server.py`, FastAPI) tunneled out through
[ngrok](https://ngrok.com) so a plain `requests.post(...)` from your own
machine reaches it.

- Needs a free ngrok account + authtoken (Colab secret `NGROK_AUTHTOKEN`,
  or pasted in when prompted).
- `/generate` requires a bearer token that's generated fresh each run and
  only ever printed in the notebook output — never written to disk. Anyone
  with both the printed ngrok URL and this token can reach your model, so
  treat both as sensitive for the lifetime of the tunnel.
- `fastapi`, `uvicorn`, and `pyngrok` are installed directly in that
  notebook's Install cell, not in `requirements.txt` — they're
  serving-only, not needed for training.
- The tunnel stays up only as long as the Colab runtime is connected;
  closing the tab or letting Colab idle-disconnect kills it.

**Colab runtime type (T4 vs. A100)** is a Colab Runtime menu setting
(Runtime → Change runtime type → Hardware accelerator/GPU type), not
anything this notebook's code controls — pick A100 there before running
Section 1's Runtime Check if you have Colab Pro/Pay-as-you-go access to it.
`model.py`'s `detect_gpu_profile()` picks sensible batch-size/sample-count
defaults per GPU class automatically either way.

**Cloudflare tunnel instead of ngrok**: Section 8b
(`server.start_server_cloudflare()`) is an alternative to Section 8a's
ngrok cell — same server, tunneled via Cloudflare's free "quick tunnel"
(`cloudflared tunnel --url ...`) instead. **No account or authtoken
needed at all**, unlike ngrok. Run one of 8a/8b, not both (they'd fight
over the same port); Section 10's Stop Server cell tears down whichever
one you used automatically. The resulting URL looks like
`https://<random-words>.trycloudflare.com` and works with every tool in
this doc that takes `--base-url`/`api_base` exactly like an ngrok URL does
— see "Running the lookup-level benchmark locally" below.

> **Note on Unsloth's own "Unsloth Studio" Colab notebook**: Unsloth
> publishes a separate, fuller-featured hosting notebook
> ([`studio/Unsloth_Studio_Colab.ipynb`](https://colab.research.google.com/github/unslothai/unsloth/blob/main/studio/Unsloth_Studio_Colab.ipynb))
> that also serves an OpenAI-compatible `/v1/chat/completions` endpoint over
> a Cloudflare tunnel, with `sk-unsloth-...`-style API keys. It launches a
> full separate web application (its own admin UI behind the tunnel), not
> something driven by editing notebook cells — this project has **not**
> verified its exact workflow for importing a custom-trained LoRA adapter
> (its docs pages for that were unavailable/404 as of this writing), so
> it isn't documented here beyond noting it exists. If you're already
> using it successfully, everything in "Running the lookup-level benchmark
> locally" below still applies — `--provider openai --base-url
> https://<its-url>/v1` talks to it the same way it talks to this
> project's own server (see the `/v1` note in that section).

### Same thing, hosted on Kaggle

`notebooks/Telecom_T2C_Inference_Server_Kaggle.ipynb` is the same ngrok-tunneled
LoRA-adapter server, adapted for Kaggle instead of Colab — useful when Colab
quota/availability is the blocker, or you'd simply rather run this on
Kaggle. Kaggle has no Google Drive mount, so the adapter has to arrive as a
**Kaggle Dataset** you attach to the notebook (upload the adapter folder or
the zip the trainer notebook's Section 11 already produces, then attach it
via the notebook sidebar's **+ Add Input**) — the notebook's own Section 4
has the exact steps. `utils.resolve_secret()` already checks Kaggle Secrets
alongside Colab's and a plain environment variable, so the ngrok-authtoken
and HF-token cells need no Kaggle-specific changes. Turn on **Internet** and
select a **GPU** in the notebook's settings before running.

### Pointing `t2c` at this server directly (LoRA adapter, no GGUF/Ollama)

`src/server.py` exposes **two** generation routes side by side:
`POST /generate` (this project's own request shape) and
`POST /chat/completions` (OpenAI's shape), both behind the same bearer-token
auth. The second one exists specifically so the sibling `t2c` project's
`--llm-provider openai` can talk to this server **unmodified** — no Ollama,
no GGUF, no merged model, just the LoRA adapter served straight from the
notebook:

```bash
export OPENAI_API_KEY="<bearer token printed by the notebook's Start Server cell>"
python -m t2c.cli benchmark gpon-xlsx --run-llm-l1 \
  --llm-provider openai --llm-model t2c-gemma4 \
  --llm-api-base https://<your-ngrok-subdomain>.ngrok-free.dev
```

`OPENAI_API_KEY` is not a real OpenAI key here — `execute_openai()` sends
whatever's in that env var as `Authorization: Bearer <value>`, which is
exactly the header this server's `_check_auth()` expects, so setting it to
the printed bearer token authenticates directly against your own server.
`--llm-model` can be any string; this server always answers with whatever
model is actually loaded regardless of what's requested. Don't set
`--llm-provider ollama` against this URL — Ollama's provider sends a
completely different request shape (`/api/generate` with
`{"model", "prompt", "system"}`), which this server doesn't understand.

### Running the lookup-level benchmark locally

`scripts/run_remote_lookup_benchmark.py` runs the exact same query set as
Section 7 of `Telecom_T2C_Benchmark.ipynb` (both import from
`src/lookup_level_queries.py`, so they can't drift apart), but from your own
PC against a hosted server instead of inside the notebook — no
GPU/model/Unsloth needed locally, only the standard library. Two providers,
via `--provider`:

- **`openai`** (default) — the Colab/Kaggle ngrok server above:

  ```bash
  export OPENAI_API_KEY="<bearer token printed by the notebook's Start Server cell>"
  python scripts/run_remote_lookup_benchmark.py \
    --base-url https://<your-ngrok-subdomain>.ngrok-free.dev
  ```

  (`--api-token` also works directly if you'd rather not reuse
  `OPENAI_API_KEY`; `T2C_API_TOKEN` is checked as a second fallback env var.)

  `--base-url` must point at wherever `/chat/completions` actually lives.
  This project's own `src/server.py` exposes it bare (no `/v1`), matching
  the example above — but any OTHER genuine OpenAI-compatible server
  (Ollama's own `/v1/chat/completions` compat mode, a Cloudflare-tunneled
  Unsloth Studio instance, etc.) puts it under `/v1`, so pass `--base-url
  https://<host>/v1` for those instead — `generate_remote()` always appends
  `/chat/completions` to exactly what you give it, nothing more.

- **`ollama`** — a model already `ollama create`'d locally (see "Running the
  fine-tuned model in Ollama" below), no bearer token needed:

  ```bash
  python scripts/run_remote_lookup_benchmark.py --provider ollama
  ```

  Defaults to `http://localhost:11434`; pass `--base-url` if Ollama is on a
  different host/port. `--model` defaults to `t2c-gemma4`, matching that
  section's `ollama create t2c-gemma4 -f Modelfile` example — pass
  `--model <your-tag>` if you named it something else.

Both talk to `src/remote_client.py`'s two client functions
(`generate_remote` for the ngrok server's OpenAI-compatible
`/chat/completions`, `generate_ollama` for Ollama's own native
`/api/chat`) — same `RemoteGenerationMetrics`/log-file/summary handling
either way, just a different wire format and a different source for the
tokens/sec number: `generate_remote` estimates it from streamed-chunk
wall-clock timing (the ngrok server has no other way to report it),
`generate_ollama` reads Ollama's own exact `eval_count`/`eval_duration`
instead of estimating, since Ollama already measures this itself.

Streaming lets the client measure **real** per-query performance, not just
pass/fail correctness:

- **TTFT** (time to first token) — wall-clock time from request sent to the
  first non-empty SSE content chunk. Dominated by prompt processing, not
  decode speed.
- **Tokens/sec** — steady-state decode throughput,
  `(tokens_generated - 1) / (total_seconds - ttft_seconds)`, i.e. excluding
  the TTFT interval, since that's prompt-processing time, not decoding.
- **Checks covered** — the same `evaluator.summarize_lookup_level_results()`
  consistency-with-Level-1 breakdown described above, printed per level
  alongside the performance numbers.

Every per-query result (generated text, parsed PASS_4 envelope,
`matches_level1`, and its TTFT/total/tokens-per-sec) is appended to a JSONL
log file **as it arrives** (default `logs/lookup_level_benchmark_<timestamp>.jsonl`,
override with `--log-file`) — a run interrupted partway through (Ctrl-C, a
dropped tunnel, a timeout) still leaves a usable partial record on disk
rather than losing everything held only in memory.

**Testing a single ad-hoc query** instead of the full 12-query suite — for
a quick "is this deployment even working" check, phase-1 exploration, or
reproducing something you saw manually — pass `--query`:

```bash
python scripts/run_remote_lookup_benchmark.py --provider ollama \
  --query "Show ONU 48575443EC9D3DB0"
```

Same provider/base-url/log-file handling as the full suite; `matches_level1`
is meaningless for a lone query (nothing else in its group to compare
against), but the generated text, parsed PASS_4 envelope, and TTFT/tokens-
per-sec all print and log exactly as normal.

## Running the fine-tuned model in Ollama

`notebooks/Telecom_T2C_Inference_Server.ipynb`'s Section 6 (Export to
GGUF) is an alternative to the ngrok server above: it exports the loaded
model + adapter as a single merged, quantized `.gguf` file via Unsloth's
`model.save_pretrained_gguf(...)`, for running with
[Ollama](https://ollama.com) or llama.cpp instead of Colab. **This is the
one place in this project that merges the adapter into the base model** —
disposable, for local serving only; the Drive-synced training artifact
(`adapter/`) is never touched. Copies the export to
`<google_drive_directory>/<run_name>/gguf/` since Colab's local disk is
ephemeral.

**Unverified**: `gemma4_unified` is a very new architecture — whether
Unsloth's GGUF export / llama.cpp actually support it is confirmed only by
that cell succeeding, not guaranteed by this project.

Ollama has no knowledge of this project's custom chat template or the
`<turn|>` stop-token fix already covered above
(`inference.build_prompt`/`_resolve_stop_token_ids`) — a `Modelfile` needs
both declared explicitly, or Ollama will reproduce the exact same
never-stops-generating bug in its own serving stack:

```
FROM ./gemma4_t2c_gguf/your-model-file.gguf
TEMPLATE """{{ if .System }}<|turn>system
{{ .System }}<turn|>
{{ end }}<|turn>user
{{ .Prompt }}<turn|>
<|turn>model
{{ .Response }}<turn|>
"""
PARAMETER stop "<turn|>"
PARAMETER temperature 0
```

Test it standalone before wiring it into anything else: `ollama create
t2c-gemma4 -f Modelfile`, then `ollama run t2c-gemma4` with a raw prompt,
and confirm it produces clean `PASS_0...PASS_4` output that actually stops.
The sibling `t2c` project's `--llm-provider ollama --llm-model t2c-gemma4`
(optionally `--llm-api-base` if Ollama isn't on the default
`http://localhost:11434`) then points its `--l1-mode llm`/`--run-llm-l1`
path at it — see that project's own docs for the prompt-alignment work
that makes this a valid comparison against the training data in the first
place (`t2c.tir.l1.build_llm_l1_prompt`/`extract_pass4_envelope`).

Once it's running, `python scripts/run_remote_lookup_benchmark.py
--provider ollama` (see "Running the lookup-level benchmark locally" above)
runs this project's own lookup-level robustness check against it directly —
useful for comparing the GGUF-exported/Ollama-served model's behavior
against the same adapter served straight from Colab/Kaggle via ngrok.

---

## Troubleshooting

**`scripts/run_remote_lookup_benchmark.py --provider ollama` logs
`tokens_generated` always exactly equal to `--max-tokens`, and `generated`
in the JSONL log is empty or cut off mid-`PASS_1`/`PASS_2`.** Ollama has
auto-detected the model as thinking-capable and is routing reasoning
tokens into a separate `message.thinking` field instead of
`message.content` — the entire token budget gets spent there, with little
or nothing left for the actual trained `PASS_0...PASS_4` answer. This is
the exact same `google/gemma-4-12B-it` thinking-channel behavior
`inference.build_prompt()` already documents and avoids for the
local/adapter path (training never demonstrates continuing from that
channel). Fixed by default — `generate_ollama()` sends `"think": false` in
every request — so if you're seeing this, confirm you're on a version of
this script that includes it (`git log -1 -- src/remote_client.py` should
show the `think=false` commit); `--think` (CLI) / `think=True` (Python)
exists only to deliberately reproduce the broken behavior for comparison.

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

**Re-enabling `assistant_only_loss=True` (not the current default —
see "Model backend"): two errors/a warning to expect, in the order
you'll likely hit them.** This project tried
`SFTConfig(assistant_only_loss=True)` and reverted it, initially believing
it was incompatible with `packing=True` — **that diagnosis turned out to
be wrong** (see the standalone `padding_free` entry below for the actual,
unrelated root cause and fix, which is now in place regardless of
`assistant_only_loss`). It hasn't been re-tried since that fix landed, so
it's still off by default, but there's no known remaining reason it
wouldn't work now. The entries below are kept for whoever retries it:
1. **`RuntimeError: Could not find the expected assistant-content
   anchor...`** from `tokenizer.patch_chat_template_for_assistant_masking()`,
   during Section 7 (Load Model). Means `google/gemma-4-12B-it`'s
   `chat_template.jinja` has been revised upstream since this patch was
   written (its docstring/comment cites the template's own
   `Published: 2026-07-09` header — Google does update it). Fix: fetch the
   current template
   (`https://huggingface.co/google/gemma-4-12B-it/raw/main/chat_template.jinja`),
   find wherever it now renders an assistant/model turn's actual text
   content, and update `_GENERATION_MARKER_ANCHOR`/
   `_GENERATION_MARKER_REPLACEMENT` in `tokenizer.py` to match — then
   re-verify the same way this was originally verified: load the tokenizer
   locally, patch it, confirm `apply_chat_template(..., tokenize=False)`
   renders byte-identical text before/after, and confirm
   `apply_chat_template(..., return_assistant_tokens_mask=True)` produces
   non-zero, correctly-positioned spans.
2. **`ValueError: Assistant-only loss is not yet supported for
   vision-language models...`** during Section 9 (Train), inside
   `SFTTrainer.__init__`. Confirmed by reading TRL's source directly:
   `SFTTrainer` sets `self._is_vlm = True` whenever
   `isinstance(processing_class, ProcessorMixin)` — unconditionally,
   regardless of whether the dataset actually has any images/audio/video —
   and hard-blocks `assistant_only_loss` (and separately, `packing`) in
   that mode. Unsloth's returned `tokenizer` for Gemma 4 is a genuine
   `Gemma4UnifiedProcessor` (`ProcessorMixin` subclass), since the model is
   nominally multimodal, which trips this. Already handled regardless of
   whether `assistant_only_loss` is on: `trainer.train()` passes
   `getattr(tokenizer, "tokenizer", tokenizer)` — the processor's own
   inner, plain-text tokenizer — as `processing_class`, which avoids VLM
   mode. `patch_chat_template_for_assistant_masking()` patches *both* the
   outer processor and its inner tokenizer independently (confirmed
   locally: they carry separate `chat_template` strings, not a shared
   reference), so re-enabling `assistant_only_loss` won't silently lose the
   generation-marker patch.

**`ValueError: When padding_free=True without packing, max_length is not
enforced. Either enable packing..., provide already truncated inputs, or
set max_length=None.`** during Section 9 (Train), inside
`SFTTrainer.__init__` (surfaces through Unsloth's compiled
`UnslothSFTTrainer.__init__`).
**This was initially misdiagnosed** as specific to combining
`assistant_only_loss=True` with `packing=True` on a conversational
(`messages`-column) dataset, and `assistant_only_loss` was reverted on
that basis — but the identical crash then recurred with
`assistant_only_loss` fully reverted (flattened `text` dataset, no
conversational format at all), proving the real cause was something else
entirely. The actual mechanism, confirmed by asking for and reading the
literal contents of `unsloth_compiled_cache/UnslothSFTTrainer.py` (not the
plain pip-installed `trl` package, which computes this differently —
see below) around its `self.padding_free = args.padding_free or
(args.packing and args.packing_strategy in {"bfd", "bfd_split"})` line:
even with `packing_strategy="wrapped"` (which makes that `or`'s second
term `False`), the crash still reproduced — meaning `args.padding_free`
itself was already truthy *before* this line ever ran. Unsloth's own
`new_init` wrapper (`unsloth/trainer.py`) evidently injects a default of
`padding_free=True` onto the config regardless of `packing_strategy` or
dataset format. `trainer.build_sft_config()` now sets `padding_free=False`
explicitly, which overrides whatever default gets injected — confirmed by
constructing a real `SFTConfig` locally with this exact combination and
reading back `.padding_free`.

(The plain, pip-installed `trl` package's own `sft_trainer.py` computes
`self.padding_free = args.padding_free or (args.packing and
args.packing_strategy == "bfd")` — a single string check, no `"bfd_split"`,
and no wrapper-injected default. That formula alone was used to (correctly,
as far as it went) verify the `packing_strategy="wrapped"` fix in isolation,
but it doesn't fully describe what Unsloth's own generated, per-run
compiled trainer actually does — which is why that first fix didn't hold
and why `padding_free=False` had to be set explicitly instead of relying
on `packing_strategy` alone.)

Separately, `"bfd"`/`"bfd_split"` packing without a supported FlashAttention
variant risks real cross-contamination between packed examples (confirmed
in that same compiled file's own warning text) — this project uses
Unsloth's xformers-based attention kernels, not FA2/3 (confirmed via
Section 7's load banner printing `FA2 = False`), so `packing_strategy="wrapped"`
(not the default `"bfd"`) is kept regardless of the `padding_free` fix
above. Tradeoff: `"wrapped"` packing can occasionally cut an example across
a pack boundary (vs. `"bfd"`'s more careful bin-packing) — a minor quality
cost given most conversations here are well under `max_seq_length`, versus
a real correctness risk. If you're on an older clone still hitting this,
`git pull`.

(One more thing you'd see if `assistant_only_loss` gets re-enabled and past
both numbered items above:
`WARNING:trl.trainer.sft_trainer:[RANK 0] The chat template does
not include the assistant turn's end-of-turn token in the loss mask; the
model may not learn to stop.` — not an error, and an already-known,
deliberate tradeoff: `patch_chat_template_for_assistant_masking()`'s
generation-marker span covers exactly the assistant's response text but
excludes the turn-closing `<turn|>` token — see its docstring/comment in
`tokenizer.py`. Extending the marker span to include `<turn|>` for
`role == 'model'` would be a larger change than the current
one-line-anchor patch, since `<turn|>`'s rendering is shared across all
roles in the template, not model-specific — see `chat_template.jinja`'s
`continues_into_next`/closing-tag logic.)

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

**Fine-tuned model generates garbled/prose text instead of PASS_0-4,
despite a very low `val_metrics.eval_loss`** (confirmed case: `eval_loss`
0.036, but `golden_metrics.exact_match_rate` 1.95% and `pass_metrics`
showing PASS_3/PASS_4 accuracy near zero, with the literal word `"system"`
spliced into generated text ~5 times per example on average, always at
structural boundaries like `"LOOKUPsystem\n\nPASS_3"` where a blank line
belonged). The low eval_loss (teacher-forced, computed with the *correct*
prior tokens fed in) proves the adapter itself learned the task correctly
— this is **not** a training/data-quality problem. Root cause, confirmed
via a live Colab diagnostic (compare a training-style render,
`apply_chat_template(messages, add_generation_prompt=False)`, against
`inference.build_prompt()`'s actual output): `google/gemma-4-12B-it`'s
chat template opens a native **thinking-mode channel**
(`<|channel>thought\n<channel|>`) whenever `add_generation_prompt=True` is
used — but `trainer.train()` renders every training conversation with
`add_generation_prompt=False`, so the model never once sees an assistant
turn that continues from that channel-opener; every training example's
assistant content follows `<|turn>model\n` *immediately*, with no channel
wrapper at all. Every actual inference call was therefore feeding the
model a prompt suffix completely outside its training distribution, which
produced free-form/prose continuations mixed with fragments of the
learned PASS_0-4 structure — explaining both the partial recovery on
simple early passes (PASS_0/PASS_2, ~79-81% accuracy) and the near-total
failure on later, more structure-dependent passes (PASS_3/PASS_4). Fixed
in `inference.build_prompt()`: instead of `add_generation_prompt=True`, it
renders the full conversation with `add_generation_prompt=False` (the
exact call `trainer.train()` uses) plus a placeholder final assistant turn
holding a unique sentinel, then truncates the rendered text right before
that sentinel — reproducing byte-for-byte what a real assistant turn's
prompt looks like in training, without hardcoding this template's
special-token spelling. **This fix requires no retraining** — re-run
Section 10 (Evaluate) or the inference-server notebook against your
existing adapter to see the corrected numbers.

**Confirmed with the `build_prompt()` fix above applied**: re-running the
256-example val benchmark afterward jumped `exact_match_rate` from 1.95%
to 99.6%, and every `pass_metrics` entry to 98-100% — the adapter was
correct all along; it only needed the right prompt.

**Generated text runs much longer than the gold reply and often
degenerates into a repeated tail after the correct PASS_0-4 answer**
(confirmed case, after the `build_prompt()` fix above: generated median
length ~2x gold's, 12% of a 256-example run spiraling into a repeated
"Please provide the deployment context..." tail). Doesn't affect
`pass_metrics`/`exact_match_rate` — `evaluator`'s PASS parsers extract
bounded sections by marker position, so trailing garbage after `PASS_4`'s
JSON is ignored — but it wastes most of the generation-eval speedup from
"Speeding up generation-based evaluation" above, since decoding runs close
to `max_new_tokens_eval` on nearly every example instead of stopping
early. Root cause, confirmed directly against
`google/gemma-4-12B-it`'s `tokenizer_config.json`: `eos_token` is
`"<eos>"`, a generic sequence-end token, but the chat template's actual
turn-closing marker is a **separate** field, `"eot_token": "<turn|>"`.
`inference.greedy_decode`/`greedy_decode_batch` only ever checked
`tokenizer.eos_token_id` — the model reliably predicts `<turn|>` (that's
what training targets), but the decode loop never recognized it as a stop
signal, so generation always ran to the full `max_new_tokens` budget even
after producing a fully correct answer. Fixed via
`inference._resolve_stop_token_ids()`: rather than hardcode the literal
`"<turn|>"` (fragile if the template changes upstream, and not
necessarily correct for a different model), it discovers the real
closing token the same way `build_prompt()` discovers the prompt prefix —
render a short conversation ending in a real assistant turn with
`add_generation_prompt=False` via a unique sentinel, then tokenize
whatever immediately follows that sentinel in the rendered text. Only
used as a stop id when it resolves to exactly one token (a multi-token
closing sequence isn't safe to stop on after a single argmax step —
degrades to `eos_token_id` alone with a logged warning in that case).
Wired through both `greedy_decode` and the batched `greedy_decode_batch`,
plus `model.generate()`'s fallback path (`eos_token_id=` now accepts the
full discovered set, not just the tokenizer's own single default).
**Also requires no retraining** — purely a decode-loop fix.

**`WARNING | ... | Could not discover an additional turn-closing stop
token ('Gemma4UnifiedProcessor' object has no attribute 'encode') — using
eos_token_id only, which may cause over-generation.`** — a real bug in
the fix above's first version, not something to ignore: `tokenizer` here
is often actually Unsloth's `Gemma4UnifiedProcessor` (Gemma 4 is nominally
multimodal — the same fact behind several other bugs in this file).
`apply_chat_template` is exposed on the processor directly, but
`.encode()` is not — only its *inner* `.tokenizer` has it.
`_resolve_stop_token_ids()` now calls `.encode()` on
`getattr(tokenizer, "tokenizer", tokenizer)`, matching the same
processor/tokenizer duality `trainer.py` already handles for exactly this
reason. Before this fix, discovery silently degraded to `eos_token_id`
alone on *every* call — harmless (no crash, no wrong output), but it
meant the over-generation problem above wasn't actually fixed despite the
warning being easy to miss in a long log.

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
If `assistant_only_loss` ever gets re-enabled (not the current default —
see "Model backend"):
`patch_chat_template_for_assistant_masking()`'s generation-marker span
covers exactly the assistant's response text (verified: decodes back to
precisely the PASS_0-4 content), but deliberately *excludes* the turn's
closing `<turn|>` token — meaning the model would get no direct gradient
signal on learning when to stop each response via this mechanism
specifically. This is a reasonable simplification (Gemma 4 already knows
generic turn-closing conventions from pretraining; this LoRA only needs to
relearn the PASS_0-4 content distribution) but is unverified end-to-end on
real hardware — if generation runs past a natural stopping point more than
before, this would be the first place to look.

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
`patch_chat_template_for_assistant_masking()` (currently unused by default
— see "Model backend" for why `assistant_only_loss` was reverted — but
kept and tested as ready-to-use infrastructure for whenever that gets
revisited: asserts the `{% generation %}`
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
