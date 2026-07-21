"""One-off generator for the tiny synthetic fixture JSONL files used by tests/.

Not a test itself and not imported at test time — run manually
(`python tests/fixtures/_build_fixtures.py`) if the fixtures ever need
regenerating. Kept in the repo so the fixture shape's provenance is obvious
and reproducible, matching the real dataset/phase1 shape (system + a
deployment-context user turn + repeated query/PASS_0-4-response pairs).
"""

import json
from pathlib import Path

SYSTEM_PROMPT = (
    "You are a GPON network inventory query compiler. Given deployment context and "
    "natural language queries, emit five passes per query: PASS_0 Normalization, "
    "PASS_1 Lexical Detection, PASS_2 Intent, PASS_3 Semantic Resolution, PASS_4 TIR envelope JSON."
)

DEPLOYMENT_CONTEXT = (
    "## Deployment context\n\nproduct_families:\n  OLT:\n    aliases:\n      - OLT\n      - MA5xxx\n"
)


def _pass4(operation: str, subject_entity: str, extra: str = "") -> str:
    envelope = {
        "status": "SUCCESS",
        "operation": {"type": operation},
        "subject": {"entity": subject_entity},
        "qualifiers": [],
    }
    return (
        "PASS_0\nNormalization\n(none)\n\n"
        "PASS_1\nLexical Detection\n- \"query\"\n\n"
        f"PASS_2\nIntent\n{operation}\n\n"
        "PASS_3\nsemantic:\n  operation: " + operation + "\n\n"
        "PASS_4\n" + json.dumps(envelope) + extra
    )


def _conversation(queries: list[tuple[str, str, str]]) -> dict:
    """queries: list of (query_text, operation, subject_entity)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": DEPLOYMENT_CONTEXT},
    ]
    for query_text, operation, subject_entity in queries:
        messages.append({"role": "user", "content": f"## Query\n{query_text}"})
        messages.append({"role": "assistant", "content": _pass4(operation, subject_entity)})
    return {"messages": messages}


TRAIN_CONVERSATIONS = [
    _conversation([("Pull up device at 10.147.48.25", "LOOKUP", "OLT")]),
    _conversation([("Which subscribers are on ALABAMA-23", "LIST", "ONU")]),
    _conversation(
        [
            ("Count OLTs in central region", "COUNT", "OLT"),
            ("List ONUs on that OLT", "LIST", "ONU"),
        ]
    ),
    _conversation([("Trace ONU with serial ABCDEF0123456789", "TRACE", "ONU")]),
    _conversation([("Show VPLS service named CORE-VPLS-1", "LIST", "VPLS")]),
]

VAL_CONVERSATIONS = [
    _conversation([("Pull up device at 10.10.3.2", "LOOKUP", "OLT")]),
    _conversation([("Count subscribers on TABA-04", "COUNT", "ONU")]),
]


def _write(path: Path, conversations: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for conv in conversations:
            f.write(json.dumps(conv, ensure_ascii=False))
            f.write("\n")


if __name__ == "__main__":
    fixtures_dir = Path(__file__).resolve().parent
    _write(fixtures_dir / "sample_train.jsonl", TRAIN_CONVERSATIONS)
    _write(fixtures_dir / "sample_val.jsonl", VAL_CONVERSATIONS)
    print(f"Wrote {len(TRAIN_CONVERSATIONS)} train + {len(VAL_CONVERSATIONS)} val conversations.")
