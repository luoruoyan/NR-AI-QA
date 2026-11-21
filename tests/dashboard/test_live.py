import pytest

from nr_ai_qa.dashboard.live import live_arguments


def test_live_invokes_existing_cli_with_selected_scope():
    argv = live_arguments("pusch", "NR-PUSCH-MCS-TBS", "PUSCH-MCS-PRB")
    assert argv[1:4] == ["-m", "nr_ai_qa.ai_cli", "--suite"]
    assert argv[-4:] == ["--requirement", "NR-PUSCH-MCS-TBS", "--risk-topic", "PUSCH-MCS-PRB"]
    assert argv[argv.index("--max-cases") + 1] == "1"


def test_live_rejects_out_of_scope_arguments():
    with pytest.raises(ValueError):
        live_arguments("pusch", "--unsafe-option", "PUSCH-MCS-PRB")


def test_scoped_workflow_request_preserves_budget_and_rejects_unrelated_requirement():
    from nr_ai_qa.dashboard.data import ROOT
    if not (ROOT / "reports/pusch/results.json").exists():
        pytest.skip("Live request setup needs the originals")
    pytest.importorskip("pydantic")
    from nr_ai_qa.ai.test_generator import make_request
    from nr_ai_qa.qa.coverage_registry import CoverageRegistry

    registry = CoverageRegistry()
    req = make_request(registry, "pusch", 1, 42,
                       requirement_ids=["NR-PUSCH-MCS-TBS"], risk_topics=["PUSCH-MCS-PRB"])
    assert req.max_cases == 1 and req.requirement_ids == ["NR-PUSCH-MCS-TBS"]
    assert req.risk_focus == ["PUSCH-MCS-PRB"]
    with pytest.raises(ValueError, match="risk scope"):
        make_request(registry, "pusch", 1, 42,
                     requirement_ids=["R-BG"], risk_topics=["PUSCH-MCS-PRB"])
