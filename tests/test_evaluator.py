"""Smoke tests for src/evaluator.py's PASS_0-4 parsing and per-pass accuracy scoring.

Assistant-turn text fixtures below match the real rendering shape used by
the sibling t2c project's PassLabels.render_assistant (confirmed against
tests/fixtures/sample_train.jsonl) — this module has no import dependency
on t2c itself, so the shape is reproduced literally rather than generated.
"""

from src.evaluator import (
    PassAccuracy,
    PredictionRecord,
    evaluate_passes,
    parse_pass0_normalizations,
    parse_pass1_lexemes,
    parse_pass2_intent,
    parse_pass3_semantic,
    parse_pass4_envelope,
)


def _assistant_text(
    *,
    normalizations: str = "(none)",
    lexemes: str = '- "query"',
    intent: str = "LOOKUP",
    semantic: str = "semantic:\n  operation: LOOKUP",
    envelope: str = '{"status": "SUCCESS", "operation": {"type": "LOOKUP"}, "subject": {"entity": "OLT"}, "qualifiers": []}',
) -> str:
    return (
        f"PASS_0\nNormalization\n{normalizations}\n\n"
        f"PASS_1\nLexical Detection\n{lexemes}\n\n"
        f"PASS_2\nIntent\n{intent}\n\n"
        f"PASS_3\n{semantic}\n\n"
        f"PASS_4\n{envelope}"
    )


GOLD_TEXT = _assistant_text()


class TestParsePass0Normalizations:
    def test_none_case(self):
        assert parse_pass0_normalizations(_assistant_text(normalizations="(none)")) == []

    def test_single_pair(self):
        text = _assistant_text(normalizations="teh OLT\n↓\nthe OLT")
        assert parse_pass0_normalizations(text) == [("teh OLT", "the OLT")]

    def test_missing_marker_returns_none(self):
        assert parse_pass0_normalizations("no passes here") is None

    def test_malformed_triple_count_returns_none(self):
        text = _assistant_text(normalizations="only one line")
        # "only one line" isn't "(none)" and isn't a multiple-of-3 line count.
        assert parse_pass0_normalizations(text) is None

    def test_wrong_arrow_returns_none(self):
        text = _assistant_text(normalizations="teh OLT\n->\nthe OLT")
        assert parse_pass0_normalizations(text) is None


class TestParsePass1Lexemes:
    def test_single_lexeme(self):
        assert parse_pass1_lexemes(_assistant_text(lexemes='- "query"')) == ["query"]

    def test_multiple_lexemes_preserve_order(self):
        text = _assistant_text(lexemes='- "OLT"\n- "port"')
        assert parse_pass1_lexemes(text) == ["OLT", "port"]

    def test_no_lexemes_is_empty_list_not_none(self):
        # Header present, zero lexeme lines beneath it — a valid "nothing detected" case.
        text = "PASS_0\nNormalization\n(none)\n\nPASS_1\nLexical Detection\n\nPASS_2\nIntent\nLOOKUP"
        assert parse_pass1_lexemes(text) == []

    def test_unquoted_line_returns_none(self):
        text = _assistant_text(lexemes="- query")
        assert parse_pass1_lexemes(text) is None

    def test_missing_marker_returns_none(self):
        assert parse_pass1_lexemes("no passes here") is None


class TestParsePass2Intent:
    def test_single_intent(self):
        assert parse_pass2_intent(_assistant_text(intent="COUNT")) == "COUNT"

    def test_missing_marker_returns_none(self):
        assert parse_pass2_intent("no passes here") is None

    def test_multiple_lines_returns_none(self):
        text = "PASS_2\nIntent\nLOOKUP\nEXTRA\n\nPASS_3\nsemantic:\n  operation: LOOKUP"
        assert parse_pass2_intent(text) is None


class TestParsePass3Semantic:
    def test_parses_semantic_dict(self):
        result = parse_pass3_semantic(_assistant_text(semantic="semantic:\n  operation: LIST"))
        assert result == {"semantic": {"operation": "LIST"}}

    def test_missing_marker_returns_none(self):
        assert parse_pass3_semantic("no passes here") is None

    def test_non_mapping_yaml_returns_none(self):
        text = "PASS_3\n- just\n- a\n- list\n\nPASS_4\n{}"
        assert parse_pass3_semantic(text) is None

    def test_invalid_yaml_returns_none(self):
        text = "PASS_3\n  bad: [unterminated\n\nPASS_4\n{}"
        assert parse_pass3_semantic(text) is None


class TestParsePass4Envelope:
    def test_parses_envelope_dict(self):
        result = parse_pass4_envelope(GOLD_TEXT)
        assert result == {
            "status": "SUCCESS",
            "operation": {"type": "LOOKUP"},
            "subject": {"entity": "OLT"},
            "qualifiers": [],
        }

    def test_nested_braces_handled(self):
        text = _assistant_text(envelope='{"status": "SUCCESS", "nested": {"a": {"b": 1}}}')
        assert parse_pass4_envelope(text) == {"status": "SUCCESS", "nested": {"a": {"b": 1}}}

    def test_missing_marker_returns_none(self):
        assert parse_pass4_envelope("no passes here") is None

    def test_invalid_json_returns_none(self):
        text = _assistant_text(envelope="{not valid json")
        assert parse_pass4_envelope(text) is None


class TestEvaluatePasses:
    def test_all_passes_correct(self):
        record = PredictionRecord(prompt="p", generated=GOLD_TEXT, gold=GOLD_TEXT, exact_match=1.0)
        report = evaluate_passes([record])
        for pass_name, acc in report.items():
            assert isinstance(acc, PassAccuracy)
            assert acc.accuracy == 1.0, pass_name
            assert acc.num_scored == 1
            assert acc.num_gold_unparseable == 0
            assert acc.num_prediction_unparseable == 0

    def test_wrong_intent_scores_pass2_zero_others_unaffected(self):
        wrong_intent = _assistant_text(intent="COUNT")
        record = PredictionRecord(prompt="p", generated=wrong_intent, gold=GOLD_TEXT, exact_match=0.0)
        report = evaluate_passes([record])
        assert report["PASS_2"].accuracy == 0.0
        assert report["PASS_2"].num_prediction_unparseable == 0  # parsed fine, just wrong
        assert report["PASS_0"].accuracy == 1.0
        assert report["PASS_4"].accuracy == 1.0

    def test_malformed_prediction_counts_as_unparseable_not_excluded(self):
        malformed = "garbage output with no pass markers at all"
        record = PredictionRecord(prompt="p", generated=malformed, gold=GOLD_TEXT, exact_match=0.0)
        report = evaluate_passes([record])
        for pass_name, acc in report.items():
            assert acc.accuracy == 0.0, pass_name
            assert acc.num_scored == 1
            assert acc.num_prediction_unparseable == 1

    def test_gold_unparseable_excluded_from_denominator(self):
        broken_gold = "no pass markers in gold either"
        record = PredictionRecord(prompt="p", generated=GOLD_TEXT, gold=broken_gold, exact_match=0.0)
        report = evaluate_passes([record])
        for pass_name, acc in report.items():
            assert acc.num_scored == 0, pass_name
            assert acc.num_gold_unparseable == 1
            assert acc.accuracy == 0.0

    def test_accuracy_averages_across_multiple_records(self):
        correct = PredictionRecord(prompt="p1", generated=GOLD_TEXT, gold=GOLD_TEXT, exact_match=1.0)
        wrong = PredictionRecord(
            prompt="p2", generated=_assistant_text(intent="COUNT"), gold=GOLD_TEXT, exact_match=0.0
        )
        report = evaluate_passes([correct, wrong])
        assert report["PASS_2"].accuracy == 0.5
        assert report["PASS_2"].num_scored == 2
