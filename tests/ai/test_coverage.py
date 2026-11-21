from nr_ai_qa.qa.coverage_registry import CoverageRegistry
from nr_ai_qa.ai.coverage_analyzer import apply_advice


def test_verified_coverage_has_bounded_denominators_and_gap():
    registry = CoverageRegistry()
    report = registry.snapshot()
    assert report["suites"]["ldpc"]["requirements"]["R-BG"]["state"] == "PARTIALLY_COVERED"
    assert report["topics"]["PUSCH-MCS-PRB"]["state"] == "NOT_COVERED"
    assert report["topics"]["PUSCH-TDL"]["state"] == "UNSUPPORTED"
    assert report["suites"]["pusch"]["metrics"]["requirement"]["denominator"] == 6
    assert "P04-BAD-IQ" not in report["suites"]["pusch"]["native_gpu_cases"]


def test_model_cannot_invent_coverage_or_requirements():
    registry = CoverageRegistry()
    report = registry.snapshot()
    payload = {"advice": [
        {"topic_id": "PUSCH-MCS-PRB", "requirement_ids": ["FAKE-38"], "reason": "interaction"},
        {"topic_id": "PUSCH-MCS-PRB", "requirement_ids": ["NR-PUSCH-MCS-TBS"],
         "reason": "interaction", "state": "COVERED"},
    ]}
    verified = apply_advice(payload, registry, "pusch")
    assert not verified["accepted"]
    assert len(verified["rejected"]) == 2
    assert report["topics"]["PUSCH-MCS-PRB"]["state"] == "NOT_COVERED"


def test_unexecuted_candidate_does_not_fill_the_scenario_gap():
    registry = CoverageRegistry()
    candidate_only = {"status": "PASS", "SDK_commit": registry.snapshot()["SDK_commit"],
                      "GPU": "NVIDIA H20", "case_id": "AI-UNEXECUTED",
                      "test_parameters": {"kind": "pusch", "mcs": 4, "num_prbs": 32},
                      "requirement_ids": ["NR-PUSCH-MCS-TBS"], "actual_result": {}}
    assert registry.snapshot([candidate_only])["topics"]["PUSCH-MCS-PRB"]["state"] == "NOT_COVERED"
