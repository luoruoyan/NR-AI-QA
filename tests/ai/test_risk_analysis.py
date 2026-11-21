from nr_ai_qa.ai.risk_analyzer import rank_risks
from nr_ai_qa.qa.coverage_registry import CoverageRegistry


def test_risk_order_is_explainable_and_does_not_count_faults_as_bugs():
    risks = rank_risks(CoverageRegistry().snapshot())
    assert risks == sorted(risks, key=lambda row: (-row["score"], row["topic_id"]))
    for row in risks:
        assert row["score"] == round(sum(row["contributions"].values()), 6)
    dmrs = next(row for row in risks if row["topic_id"] == "PUSCH-DMRS")
    bg = next(row for row in risks if row["topic_id"] == "LDPC-BG")
    assert dmrs["factors"]["historical_failure"] == 0
    assert "P05-DMRS-FAULT" in dmrs["controlled_fault_cases"]
    assert "BG-R067" in bg["historical_anomalies"]
