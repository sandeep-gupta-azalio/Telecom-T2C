"""Smoke tests for src/evaluator.py's PASS_0-4 parsing and per-pass accuracy scoring.

Assistant-turn text fixtures below match the real rendering shape used by
the sibling t2c project's PassLabels.render_assistant (confirmed against
tests/fixtures/sample_train.jsonl) — this module has no import dependency
on t2c itself, so the shape is reproduced literally rather than generated.
"""

import pytest

from src.evaluator import (
    LookupLevelQuery,
    PassAccuracy,
    PredictionRecord,
    evaluate_passes,
    generate_predictions,
    parse_pass0_normalizations,
    parse_pass1_lexemes,
    parse_pass2_intent,
    parse_pass3_semantic,
    parse_pass4_envelope,
    run_lookup_level_benchmark,
    summarize_lookup_level_results,
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


def _dataset_row(query: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": query},
            {"role": "assistant", "content": GOLD_TEXT},
        ]
    }


class TestGeneratePredictions:
    def test_batch_size_one_calls_decode_fn_once_per_example(self):
        calls = []

        def decode_fn(model, tokenizer, prompt_messages, max_new_tokens):
            calls.append(prompt_messages)
            return GOLD_TEXT

        dataset = [_dataset_row("q1"), _dataset_row("q2")]
        records = generate_predictions(object(), object(), dataset, 100, decode_fn)

        assert len(records) == 2
        assert len(calls) == 2
        assert all(r.exact_match == 1.0 for r in records)

    def test_batch_size_greater_than_one_requires_batch_decode_fn(self):
        def decode_fn(model, tokenizer, prompt_messages, max_new_tokens):
            return GOLD_TEXT

        with pytest.raises(ValueError):
            generate_predictions(object(), object(), [_dataset_row("q1")], 100, decode_fn, batch_size=4)

    def test_batches_examples_into_chunks_of_batch_size(self):
        batch_calls = []

        def decode_fn(model, tokenizer, prompt_messages, max_new_tokens):
            raise AssertionError("decode_fn should not be used when batch_size > 1")

        def batch_decode_fn(model, tokenizer, prompt_messages_batch, max_new_tokens):
            batch_calls.append(len(prompt_messages_batch))
            return [GOLD_TEXT] * len(prompt_messages_batch)

        dataset = [_dataset_row(f"q{i}") for i in range(5)]
        records = generate_predictions(
            object(), object(), dataset, 100, decode_fn, batch_size=2, batch_decode_fn=batch_decode_fn
        )

        assert len(records) == 5
        assert batch_calls == [2, 2, 1]  # 5 examples in batches of 2 -> 2, 2, 1
        assert all(r.exact_match == 1.0 for r in records)

    def test_skips_conversations_not_ending_on_assistant_turn(self):
        def decode_fn(model, tokenizer, prompt_messages, max_new_tokens):
            return GOLD_TEXT

        dataset = [_dataset_row("q1"), {"messages": [{"role": "user", "content": "no reply"}]}]
        records = generate_predictions(object(), object(), dataset, 100, decode_fn)
        assert len(records) == 1


def _envelope_text(entity: str, value: str) -> str:
    envelope = (
        '{"status": "SUCCESS", "operation": {"type": "LOOKUP"}, '
        f'"subject": {{"entity": "{entity}"}}, '
        f'"qualifiers": [{{"attribute": "ID", "operator": "=", "value": "{value}"}}]}}'
    )
    return _assistant_text(envelope=envelope)


class TestRunLookupLevelBenchmark:
    def _make_decode_fn(self, responses: dict):
        def decode_fn(model, tokenizer, messages, max_new_tokens):
            query = messages[-1]["content"]
            return responses[query]

        return decode_fn

    def test_level_1_is_never_scored(self):
        queries = [LookupLevelQuery(group_id="g1", level=1, level_name="Explicit", query="q1")]
        decode_fn = self._make_decode_fn({"## Query\nq1": _envelope_text("ONU", "SN1")})

        results = run_lookup_level_benchmark(
            object(), object(), queries, "sys", "ctx", decode_fn, max_new_tokens=10
        )
        assert results[0].matches_level1 is None

    def test_matching_and_diverging_levels_within_a_group(self):
        queries = [
            LookupLevelQuery(group_id="g1", level=1, level_name="Explicit", query="show ONU SN1"),
            LookupLevelQuery(group_id="g1", level=3, level_name="Implicit", query="show SN1"),
            LookupLevelQuery(group_id="g1", level=4, level_name="Natural", query="details for SN1"),
        ]
        decode_fn = self._make_decode_fn({
            "## Query\nshow ONU SN1": _envelope_text("ONU", "SN1"),
            "## Query\nshow SN1": _envelope_text("ONU", "SN1"),  # matches level 1
            "## Query\ndetails for SN1": _envelope_text("OLT", "SN1"),  # diverges
        })

        results = run_lookup_level_benchmark(
            object(), object(), queries, "sys", "ctx", decode_fn, max_new_tokens=10
        )
        by_level = {r.level: r for r in results}
        assert by_level[1].matches_level1 is None
        assert by_level[3].matches_level1 is True
        assert by_level[4].matches_level1 is False

    def test_groups_are_scored_independently(self):
        queries = [
            LookupLevelQuery(group_id="onu", level=1, level_name="Explicit", query="onu q1"),
            LookupLevelQuery(group_id="onu", level=3, level_name="Implicit", query="onu q3"),
            LookupLevelQuery(group_id="ip", level=1, level_name="Explicit", query="ip q1"),
            LookupLevelQuery(group_id="ip", level=3, level_name="Implicit", query="ip q3"),
        ]
        decode_fn = self._make_decode_fn({
            "## Query\nonu q1": _envelope_text("ONU", "A"),
            "## Query\nonu q3": _envelope_text("ONU", "A"),  # matches its own group's level 1
            "## Query\nip q1": _envelope_text("NE", "B"),
            "## Query\nip q3": _envelope_text("ONU", "WRONG"),  # diverges from its own group's level 1
        })

        results = run_lookup_level_benchmark(
            object(), object(), queries, "sys", "ctx", decode_fn, max_new_tokens=10
        )
        by_group_level = {(r.group_id, r.level): r for r in results}
        assert by_group_level[("onu", 3)].matches_level1 is True
        assert by_group_level[("ip", 3)].matches_level1 is False

    def test_unparseable_level1_leaves_group_unscored(self):
        queries = [
            LookupLevelQuery(group_id="g1", level=1, level_name="Explicit", query="q1"),
            LookupLevelQuery(group_id="g1", level=3, level_name="Implicit", query="q3"),
        ]
        decode_fn = self._make_decode_fn({
            "## Query\nq1": "garbage, no PASS markers at all",
            "## Query\nq3": _envelope_text("ONU", "SN1"),
        })

        results = run_lookup_level_benchmark(
            object(), object(), queries, "sys", "ctx", decode_fn, max_new_tokens=10
        )
        by_level = {r.level: r for r in results}
        assert by_level[1].pass4_envelope is None
        assert by_level[3].matches_level1 is None

    def test_on_result_called_once_per_query_in_order_as_produced(self):
        queries = [
            LookupLevelQuery(group_id="g1", level=1, level_name="Explicit", query="q1"),
            LookupLevelQuery(group_id="g1", level=3, level_name="Implicit", query="q3"),
        ]
        decode_fn = self._make_decode_fn({
            "## Query\nq1": _envelope_text("ONU", "SN1"),
            "## Query\nq3": _envelope_text("ONU", "SN1"),
        })
        seen: list = []

        results = run_lookup_level_benchmark(
            object(), object(), queries, "sys", "ctx", decode_fn, max_new_tokens=10,
            on_result=seen.append,
        )
        assert seen == results
        assert [r.level for r in seen] == [1, 3]


class TestSummarizeLookupLevelResults:
    def test_level1_marked_as_reference_not_scored(self):
        from src.evaluator import LookupLevelResult

        results = [
            LookupLevelResult(
                group_id="g1", level=1, level_name="Explicit", query="q1",
                generated="x", pass4_envelope={}, matches_level1=None,
            )
        ]
        summary = summarize_lookup_level_results(results)
        assert "consistency_with_level1" not in summary[1]
        assert summary[1]["num_queries"] == 1

    def test_consistency_rate_averages_across_groups(self):
        from src.evaluator import LookupLevelResult

        results = [
            LookupLevelResult(
                group_id="g1", level=3, level_name="Implicit", query="q1",
                generated="x", pass4_envelope={}, matches_level1=True,
            ),
            LookupLevelResult(
                group_id="g2", level=3, level_name="Implicit", query="q2",
                generated="x", pass4_envelope={}, matches_level1=False,
            ),
        ]
        summary = summarize_lookup_level_results(results)
        assert summary[3]["consistency_with_level1"] == 0.5
        assert summary[3]["num_scoreable"] == 2

    def test_unscoreable_results_excluded_from_denominator(self):
        from src.evaluator import LookupLevelResult

        results = [
            LookupLevelResult(
                group_id="g1", level=3, level_name="Implicit", query="q1",
                generated="x", pass4_envelope=None, matches_level1=None,
            ),
        ]
        summary = summarize_lookup_level_results(results)
        assert summary[3]["num_scoreable"] == 0
        assert summary[3]["consistency_with_level1"] == 0.0
