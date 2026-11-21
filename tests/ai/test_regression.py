import pytest

from nr_ai_qa.ai.regression_planner import select_regression_tests
from nr_ai_qa.ai.schemas import Budget, ChangeContext
from nr_ai_qa.qa.coverage_registry import CoverageRegistry


def test_regression_prioritizes_known_case_and_golden():
    registry = CoverageRegistry()
    budget = Budget.model_validate(registry.config["budget"])
    change = ChangeContext(family="ldpc", category="ldpc_processing", affected_requirements=["R-BG"],
                           affected_modules=["base_graph"], known_failed_case_ids=["BG-R067"],
                           model_suggested_case_ids=[])
    plan = select_regression_tests(change, registry, budget)
    assert plan["status"] == "READY"
    assert plan["selected_case_ids"][:2] == ["BG-R067", "GOLD-QPSK"]
    assert "FAULT-CRC-FLIP" in plan["selected_case_ids"]
    assert "RV-BG2-REPEAT" in plan["selected_case_ids"]


def test_noise_regression_and_unknown_id():
    registry = CoverageRegistry()
    budget = Budget.model_validate(registry.config["budget"])
    change = ChangeContext(family="pusch", category="noise_variance", affected_requirements=[],
                           affected_modules=["noise_variance"], known_failed_case_ids=[],
                           model_suggested_case_ids=[])
    plan = select_regression_tests(change, registry, budget)
    assert {"P01-GOLDEN", "P02-CONSISTENCY", "P05-NOISE-FAULT", "P03-MID"} <= set(plan["selected_case_ids"])
    with pytest.raises(ValueError, match="Unknown"):
        select_regression_tests(change.model_copy(update={"model_suggested_case_ids": ["NO-SUCH-CASE"]}),
                                registry, budget)


def test_mandatory_budget_shortfall_blocks():
    registry = CoverageRegistry()
    budget = Budget.model_validate(registry.config["budget"] | {"max_regression_cases": 1})
    change = ChangeContext(family="ldpc", category="ldpc_processing", affected_requirements=["R-BG"],
                           affected_modules=[], known_failed_case_ids=["BG-R067"],
                           model_suggested_case_ids=[])
    assert select_regression_tests(change, registry, budget)["status"] == "BLOCKED"


def test_pusch_configuration_selects_matching_receiver_and_parameter_checks():
    registry = CoverageRegistry()
    budget = Budget.model_validate(registry.config["budget"])
    change = ChangeContext(family="pusch", category="configuration",
                           affected_requirements=["NR-PUSCH-MCS-TBS"], affected_modules=["mcs", "prb", "dmrs"],
                           known_failed_case_ids=[], model_suggested_case_ids=["P04-SISO"])
    plan = select_regression_tests(change, registry, budget)
    assert plan["selected_case_ids"][0] == "P01-GOLDEN"
    assert {"P02-CONSISTENCY", "P04-DMRS", "P04-MCS", "P04-PRB"} <= set(plan["selected_case_ids"])



def test_host_only_contract_is_not_reported_as_gpu_regression():
    registry = CoverageRegistry()
    budget = Budget.model_validate(registry.config["budget"] | {"max_regression_cases": 12})
    context = ChangeContext(family="pusch", category="configuration",
                            affected_requirements=["SDK-PUSCH-IO"], affected_modules=[],
                            known_failed_case_ids=[], model_suggested_case_ids=["P04-BAD-IQ"])
    plan = select_regression_tests(context, registry, budget)
    assert "P04-BAD-IQ" not in plan["selected_case_ids"]
    assert any(r["case_id"] == "P04-BAD-IQ" and r["reason"] == "HOST_ONLY_CONTRACT"
               for r in plan["skipped"])
