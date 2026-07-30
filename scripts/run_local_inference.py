#!/usr/bin/env python
"""Run inference against a LoRA adapter on THIS machine directly, via Unsloth
-- no Colab/Kaggle/ngrok needed. See README "Running inference on your own
PC (no Colab/Kaggle/ngrok)".

The base model is read from the adapter's own adapter_config.json
(base_model_name_or_path, recorded automatically at training time) via
inference.load_model_for_inference() -- configs/experiment.yaml's own
`model.base_model` is NOT used here, only its data.max_seq_length.

Usage (single smoke-test query):
    python scripts/run_local_inference.py --adapter-dir /path/to/adapter

Usage (custom query):
    python scripts/run_local_inference.py --adapter-dir /path/to/adapter \\
        --query "Show ONU 48575443EC9D3DB0"

Usage (also serve it over HTTP on this machine -- point
scripts/run_remote_lookup_benchmark.py --provider openai
--base-url http://localhost:<port> at it, no ngrok tunnel required since
both ends are the same machine):
    python scripts/run_local_inference.py --adapter-dir /path/to/adapter --serve
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config as config_mod  # noqa: E402
from src import inference, utils  # noqa: E402
from src.lookup_level_queries import (  # noqa: E402
    LOOKUP_LEVEL_DEPLOYMENT_CONTEXT,
    LOOKUP_LEVEL_SYSTEM_PROMPT,
)

logger = utils.get_logger("run_local_inference")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter-dir", required=True, help="Path to the trained LoRA adapter directory.")
    parser.add_argument(
        "--config",
        default="configs/experiment.yaml",
        help="Only data.max_seq_length is read from this -- the base model comes from "
        "the adapter's own adapter_config.json, not this file.",
    )
    parser.add_argument(
        "--hf-token-env-var",
        default="HF_TOKEN",
        help="Env var (or Colab/Kaggle secret) to resolve an HF token from, if the base model needs auth.",
    )
    parser.add_argument(
        "--query",
        default="Show ONU 48575443EC9D3DB0",
        help="Smoke-test query to run through the canonical system prompt/deployment context "
        "shared with scripts/run_remote_lookup_benchmark.py. Ignored with --serve.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=400)
    parser.add_argument(
        "--serve",
        action="store_true",
        help="After loading, start src/server.py's FastAPI app on this machine instead of "
        "running a single smoke-test query.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="--serve only. Use 0.0.0.0 to accept connections from other machines on your network.",
    )
    parser.add_argument("--port", type=int, default=8000, help="--serve only.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    experiment_config = config_mod.load_config(Path(args.config))
    hf_token = utils.resolve_secret(args.hf_token_env_var)

    print(f"Loading adapter from {args.adapter_dir} ...")
    model, tokenizer = inference.load_model_for_inference(
        experiment_config.model,
        experiment_config.data.max_seq_length,
        args.adapter_dir,
        hf_token,
    )
    print("Loaded.")

    if args.serve:
        import uvicorn

        from src import server

        api_token = server.generate_api_token()
        app = server.build_app(model, tokenizer, api_token)
        print(f"\nBearer token: {api_token}")
        print(
            f"Point scripts/run_remote_lookup_benchmark.py --provider openai "
            f'--base-url http://{args.host}:{args.port} --api-token "{api_token}" at this.\n'
        )
        uvicorn.run(app, host=args.host, port=args.port)
        return 0

    messages = [
        {"role": "system", "content": LOOKUP_LEVEL_SYSTEM_PROMPT},
        {"role": "user", "content": LOOKUP_LEVEL_DEPLOYMENT_CONTEXT},
        {"role": "user", "content": f"## Query\n{args.query}"},
    ]
    print(f"\nQuery: {args.query}\n")
    result = inference.generate(model, tokenizer, messages, max_new_tokens=args.max_new_tokens)
    print("=== Generated ===")
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
