"""Dataset loading and validation.

The only place that reads raw JSONL and enforces the {"messages": [...]}
schema. Per project decision, conversations are used AS-IS — the whole
`messages` array (system + deployment-context user turn + repeated
query/PASS_0-4-response pairs) is fed straight into
`tokenizer.apply_chat_template()` later by trainer.py, with no reformatting
here.

Loading strategy: the real train split is ~544MB / 91,496 lines, so this
module never materializes the full file as a Python list. Each split is
streamed line-by-line, validated, and re-written to a small filtered temp
file (this also makes JSON-corruption handling trivial — a bad line is just
dropped during the stream, never reaching the Arrow loader). The filtered
temp file is then loaded via `datasets.load_dataset("json", ...)`, which
memory-maps the Arrow representation rather than holding it all in the
Python heap.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from src import utils
from src.config import DataConfig

logger = utils.get_logger("dataset")

_VALID_ROLES = frozenset({"system", "user", "assistant"})
_MAX_DROPPED_LINES_TRACKED = 50


@dataclass
class ValidationReport:
    total_lines: int
    valid: int
    invalid: int
    invalid_reasons: dict[str, int] = field(default_factory=dict)
    dropped_line_numbers: list[int] = field(default_factory=list)


class DatasetLoader:
    """Loads, validates, and reports statistics for the train/val/golden splits."""

    def __init__(self, data_config: DataConfig, tokenizer: Any, seed: int, logger_: Any = None):
        self.data_config = data_config
        self.tokenizer = tokenizer
        self.seed = seed
        self.logger = logger_ or logger
        self._scratch_dir = Path(tempfile.mkdtemp(prefix="telecom_t2c_dataset_"))

    # -- public API -----------------------------------------------------

    def load_train(self) -> Any:
        """Load the training split. Raises if the file is missing or has zero valid rows."""
        path = Path(self.data_config.train_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Training data not found: {path}. Check data.train_path in configs/experiment.yaml."
            )
        ds, report = self._load_split(path, "train", self.data_config.max_train_samples)
        self.print_statistics("train", report)
        if ds is None:
            raise ValueError(
                f"No valid training examples found in {path} after validation — see reasons above."
            )
        return ds

    def load_validation(self) -> Optional[Any]:
        """Load the validation split. Returns None gracefully if unset/missing/empty."""
        path_str = self.data_config.val_path
        if not path_str:
            self.logger.info("data.val_path unset — skipping validation split.")
            return None
        path = Path(path_str)
        if not path.is_file():
            self.logger.warning("data.val_path set but not found: %s — skipping validation split.", path)
            return None
        ds, report = self._load_split(path, "val", self.data_config.max_eval_samples)
        self.print_statistics("val", report)
        return ds

    def load_golden(self) -> Optional[Any]:
        """Load the golden eval split. Returns None gracefully if unset/missing/empty.

        There is no golden file shipped with this project yet — this is the
        normal, expected path today, not an error.
        """
        path_str = self.data_config.golden_path
        if not path_str:
            self.logger.info("data.golden_path unset — golden evaluation will be skipped.")
            return None
        path = Path(path_str)
        if not path.is_file():
            self.logger.info(
                "data.golden_path set but not found: %s — golden evaluation will be skipped.", path
            )
            return None
        ds, report = self._load_split(path, "golden", None)
        self.print_statistics("golden", report)
        return ds

    # -- validation (static, independently testable) --------------------

    @staticmethod
    def validate_json(line: str) -> tuple[bool, Optional[dict], Optional[str]]:
        """Parse one JSONL line. Returns (ok, record_or_None, reason_or_None)."""
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            return False, None, f"invalid_json:{exc.msg}"
        if not isinstance(data, dict):
            return False, None, "not_a_json_object"
        return True, data, None

    @staticmethod
    def validate_messages(record: dict) -> tuple[bool, Optional[str]]:
        """Check the record has a non-empty "messages" list."""
        messages = record.get("messages")
        if messages is None:
            return False, "missing_messages_key"
        if not isinstance(messages, list) or len(messages) == 0:
            return False, "messages_not_a_nonempty_list"
        return True, None

    @staticmethod
    def validate_roles(messages: list) -> tuple[bool, Optional[str]]:
        """Check each turn has a valid role + non-empty content, and turns are sanely ordered.

        The real dataset batches multiple query/response pairs under one
        deployment-context preamble, so two consecutive "user" turns are
        valid (deployment-context turn, then the first query turn) — only
        two consecutive "assistant" turns, or a system role appearing
        anywhere but first, are treated as malformed.
        """
        for i, message in enumerate(messages):
            if not isinstance(message, dict):
                return False, f"turn_{i}_not_a_dict"
            role = message.get("role")
            if role not in _VALID_ROLES:
                return False, f"turn_{i}_invalid_role:{role}"
            if role == "system" and i != 0:
                return False, "system_role_not_first"
            content = message.get("content")
            if not content or not str(content).strip():
                return False, f"turn_{i}_empty_content"

        non_system_roles = [m.get("role") for m in messages if m.get("role") != "system"]
        if not non_system_roles:
            return False, "no_user_or_assistant_turns"
        if non_system_roles[0] != "user":
            return False, "does_not_start_with_user_turn"
        for prev_role, curr_role in zip(non_system_roles, non_system_roles[1:]):
            if prev_role == "assistant" and curr_role == "assistant":
                return False, "consecutive_assistant_turns"
        return True, None

    # -- reporting --------------------------------------------------------

    def print_statistics(self, split_name: str, report: ValidationReport) -> None:
        """Print the validation summary (record counts / invalid reasons).

        Deeper token statistics (avg tokens, percentiles, histograms) are a
        separate concern handled by statistics.py, called independently by
        the notebook — kept out of this class to avoid a dataset<->statistics
        import cycle.
        """
        print(f"=== Dataset validation: {split_name} ===")
        print(f"  total lines: {report.total_lines:,}")
        print(f"  valid:       {report.valid:,}")
        print(f"  invalid:     {report.invalid:,}")
        if report.invalid_reasons:
            print("  invalid reasons:")
            for reason, count in sorted(report.invalid_reasons.items(), key=lambda kv: -kv[1]):
                print(f"    {reason}: {count}")
        if report.dropped_line_numbers:
            preview = ", ".join(str(n) for n in report.dropped_line_numbers[:10])
            more = "..." if report.invalid > len(report.dropped_line_numbers) else ""
            print(f"  first dropped line numbers: {preview}{more}")

    # -- internals ----------------------------------------------------------

    def _filtered_path_for(self, split_name: str) -> Path:
        return self._scratch_dir / f"{split_name}_filtered.jsonl"

    def _load_split(
        self, path: Path, split_name: str, max_samples: Optional[int]
    ) -> tuple[Optional[Any], ValidationReport]:
        total = 0
        valid = 0
        invalid = 0
        invalid_reasons: dict[str, int] = {}
        dropped_line_numbers: list[int] = []

        def _valid_rows():
            nonlocal total, valid, invalid
            with path.open("r", encoding="utf-8") as f:
                for line_number, raw_line in enumerate(f, start=1):
                    line = raw_line.strip()
                    if not line:
                        continue
                    total += 1

                    ok, record, reason = self.validate_json(line)
                    if not ok:
                        invalid += 1
                        invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1
                        if len(dropped_line_numbers) < _MAX_DROPPED_LINES_TRACKED:
                            dropped_line_numbers.append(line_number)
                        continue

                    ok, reason = self.validate_messages(record)
                    if not ok:
                        invalid += 1
                        invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1
                        if len(dropped_line_numbers) < _MAX_DROPPED_LINES_TRACKED:
                            dropped_line_numbers.append(line_number)
                        continue

                    ok, reason = self.validate_roles(record["messages"])
                    if not ok:
                        invalid += 1
                        invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1
                        if len(dropped_line_numbers) < _MAX_DROPPED_LINES_TRACKED:
                            dropped_line_numbers.append(line_number)
                        continue

                    valid += 1
                    yield {"messages": record["messages"]}

        filtered_path = self._filtered_path_for(split_name)
        utils.write_jsonl(filtered_path, _valid_rows())
        report = ValidationReport(
            total_lines=total, valid=valid, invalid=invalid,
            invalid_reasons=invalid_reasons, dropped_line_numbers=dropped_line_numbers,
        )

        if valid == 0:
            return None, report

        try:
            import datasets
        except ImportError as exc:
            raise ImportError(
                "The 'datasets' package is required to load JSONL splits. "
                "Install project requirements: pip install -r requirements.txt"
            ) from exc

        try:
            ds = datasets.load_dataset("json", data_files=str(filtered_path), split="train")
        except Exception as exc:
            raise RuntimeError(f"Failed to load filtered dataset for split '{split_name}': {exc}") from exc

        if max_samples is not None and len(ds) > max_samples:
            ds = ds.shuffle(seed=self.seed).select(range(max_samples))

        return ds, report
