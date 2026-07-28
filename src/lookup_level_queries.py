"""Canonical query set for the lookup-level robustness check (see evaluator.py's
run_lookup_level_benchmark). Single source of truth shared by
notebooks/Telecom_T2C_Benchmark.ipynb (Section 7, in-notebook/local-GPU run)
and scripts/run_remote_lookup_benchmark.py (HTTP run against a hosted server)
so the two paths can never drift into scoring different queries.
"""

from __future__ import annotations

from src.evaluator import LookupLevelQuery

LEVEL_NAMES: dict[int, str] = {
    1: "Explicit entity + explicit identifier",
    2: "Synonyms",
    3: "Implicit entity from identifier",
    4: "Natural operator language",
}

LOOKUP_LEVEL_SYSTEM_PROMPT = (
    "You are a GPON network inventory query compiler. Given deployment context and "
    "natural language queries, emit five passes per query:\n"
    "PASS_0 Normalization — spelling/token fixes only; (none) when query is already clean.\n"
    "PASS_1 Lexical Detection — quoted verbatim phrases from normalized text (lexer output).\n"
    "PASS_2 Intent — exactly one canonical operation (LOOKUP, LIST, TRACE, COUNT, "
    "UNSUPPORTED on failure traces, etc.).\n"
    "PASS_3 Semantic Resolution — YAML semantic record only (mention, entity, source, confidence).\n"
    "PASS_4 TIR envelope JSON with status (SUCCESS or failure status) and diagnostics when not SUCCESS.\n"
    "Never invent identifiers or filters. Use deployment aliases only when listed in context."
)

LOOKUP_LEVEL_DEPLOYMENT_CONTEXT = (
    "## Deployment context\n\n"
    "product_families:\n"
    "  OLT:\n"
    "    aliases:\n"
    "      - OLT\n"
    "      - MA5xxx\n"
    "      - DSLAM\n\n"
    "olt_name_aliases:\n"
    "  TABA-04: TABA-04_XABA+A03_GPON_CO\n"
)

# onu_sn: exactly the 4 phrasings given when this check was requested.
# olt_name: levels 1-2 given directly; 3-4 added to complete the pattern.
# ne_ip: levels 3-4 given directly; 1-2 added to complete the pattern.
DEFAULT_LOOKUP_LEVEL_QUERIES: list[LookupLevelQuery] = [
    LookupLevelQuery("onu_sn", 1, LEVEL_NAMES[1], "Show ONU 48575443EC9D3DB0"),
    LookupLevelQuery("onu_sn", 2, LEVEL_NAMES[2], "Find subscriber 48575443EC9D3DB0"),
    LookupLevelQuery("onu_sn", 3, LEVEL_NAMES[3], "Show 48575443EC9D3DB0"),
    LookupLevelQuery("onu_sn", 4, LEVEL_NAMES[4], "Need details for subscriber 48575443EC9D3DB0"),
    LookupLevelQuery("olt_name", 1, LEVEL_NAMES[1], "Find OLT TABA-04"),
    LookupLevelQuery("olt_name", 2, LEVEL_NAMES[2], "Locate ONT TABA-04"),
    LookupLevelQuery("olt_name", 3, LEVEL_NAMES[3], "Find TABA-04"),  # added
    LookupLevelQuery("olt_name", 4, LEVEL_NAMES[4], "Pull up the record for TABA-04"),  # added
    LookupLevelQuery("ne_ip", 1, LEVEL_NAMES[1], "Show NE 10.99.1.20"),  # added
    LookupLevelQuery("ne_ip", 2, LEVEL_NAMES[2], "Locate device 10.99.1.20"),  # added
    LookupLevelQuery("ne_ip", 3, LEVEL_NAMES[3], "Find 10.99.1.20"),
    LookupLevelQuery("ne_ip", 4, LEVEL_NAMES[4], "Pull up the device at 10.99.1.20"),
]
