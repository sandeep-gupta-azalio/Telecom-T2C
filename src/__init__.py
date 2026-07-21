"""Telecom-T2C-Trainer source package.

Modules are organized as a strict dependency DAG (see README "Architecture"):
utils -> config -> {manifest, tokenizer -> statistics -> dataset, model,
checkpoint, wandb_logger, evaluator -> callbacks -> inference -> trainer ->
benchmark}.
"""
