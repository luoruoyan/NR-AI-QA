import copy

import pytest

from nr_ai_qa.ai.defect_analyzer import analyze_defects, collect_evidence
from nr_ai_qa.qa.coverage_registry import CoverageRegistry


@pytest.mark.parametrize(("suite", "case_id", "category", "field", "claim"), [
    ("ldpc", "FAULT-RM-FLIP", "CONTROLLED_INJECTION", "failure_category", "SDK_BUG"),
    ("ldpc", "BG-R067", "POTENTIAL_SDK_DEFECT", "confirmed", True),
    ("pusch", "P03-SMOKE-LOW", "STATISTICAL_UNCERTAINTY", "failure_category", "SDK_BUG"),
    ("pusch", "P05-NOISE-FAULT", "CONTROLLED_INJECTION", "hypothesis_ids", ["BLER_REGRESSION"]),
    ("pusch", "P05-DMRS-FAULT", "CONTROLLED_INJECTION", "clause", "38.212 §999"),
])
def test_model_cannot_escalate_or_invent_a_clause(suite, case_id, category, field, claim):
    registry = CoverageRegistry()
    targets = collect_evidence(registry, suite, [])
    before = copy.deepcopy(targets)
    reply = {"assessments": [{"case_id": case_id, "hypothesis_ids": [],
                              "evidence_ids": ["EV-OBSERVATION"], "investigation_ids": [],
                              "regression_case_ids": [], "confidence": "high"}]}
    reply["assessments"][0][field] = claim
    result = analyze_defects(targets, reply)
    assert result["rejected"]
    output = next(x for x in result["analyses"] if x["case_id"] == case_id)
    assert output["failure_category"] == category
    assert output["model_review"] == "NO_VALID_PROPOSAL"
    assert output["execution_status"] == before[case_id]["execution_status"]
    assert targets == before


def test_known_evidence_retains_distinct_native_outcomes():
    registry = CoverageRegistry()
    ldpc = collect_evidence(registry, "ldpc", [])
    pusch = collect_evidence(registry, "pusch", [])
    assert ldpc["BG-R067"]["actual"]["native_bg"] == 2
    assert ldpc["BG-R067"]["actual"]["sdk_helper_bg"] == 1
    assert ldpc["FAULT-RM-FLIP"]["actual"]["tb_crc"] == 0
    assert ldpc["FAULT-RM-FLIP"]["difference"]["count"] == 1
    assert ldpc["FAULT-CRC-FLIP"]["actual"]["tb_crc"] != 0
    assert pusch["P05-DMRS-FAULT"]["actual"]["detections"] == 8
    assert pusch["P05-NOISE-FAULT"]["actual"]["crc_failures"] == {"clean": 0, "injected": 0}
