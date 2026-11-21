from copy import deepcopy

import pytest

from nr_ai_qa.ai.case_validator import CaseValidator, RejectedCase
from nr_ai_qa.ai.schemas import Budget, GenerationRequest
from nr_ai_qa.qa.coverage_registry import CoverageRegistry


@pytest.fixture
def inputs():
    registry = CoverageRegistry()
    parent = registry.profiles["ldpc"]["RV-BG2-3"]
    budget = Budget.model_validate(registry.config["budget"])
    request = GenerationRequest(test_family="ldpc", requirement_ids=["R-RV"], risk_focus=["LDPC-RV"],
                                max_cases=2, seed=771, execution_budget=budget, existing_coverage={})
    candidate = {"parent_case_id": "RV-BG2-3", "related_requirements": ["R-RV"],
                 "risk_tags": ["LDPC-RV"], "risk_reason": "seed_variation",
                 "case": parent | {"case_id": "AI-RV-NEW", "requirement_ids": ["R-RV"], "seed": 771}}
    return CaseValidator(registry, request), candidate


@pytest.mark.parametrize(("field", "value", "code"), [
    ("rv", 4, "INVALID_CASE"),
    ("expected_behavior", "measure", "INVALID_CASE"),
    ("oracle", {"force_pass": True}, "INVALID_CASE"),
    ("requirement_ids", ["FAKE-38"], "INVALID_CASE"),
    ("shell", "not_a_case_parameter", "INVALID_CASE"),
])
def test_invalid_and_oracle_override_are_rejected(inputs, field, value, code):
    validator, candidate = inputs
    candidate["case"][field] = value
    with pytest.raises(RejectedCase, match=code):
        validator.accept(candidate)


def test_semantic_duplicates_not_evaded_by_id_or_tags(inputs):
    validator, candidate = inputs
    first = validator.accept(candidate)
    assert first["case"]["rv"] == 3
    renamed = deepcopy(candidate)
    renamed["case"]["case_id"] = "AI-RENAMED"
    renamed["case"]["risk_tags"] = ["LDPC-BG"]
    with pytest.raises(RejectedCase, match="DUPLICATE"):
        validator.accept(renamed)


def test_pusch_budget_rejects_valid_but_large_case():
    registry = CoverageRegistry()
    request = GenerationRequest(test_family="pusch", requirement_ids=["NR-PUSCH-MCS-TBS"],
                                risk_focus=[], max_cases=2, seed=775,
                                execution_budget=Budget.model_validate(registry.config["budget"]),
                                existing_coverage={})
    case = registry.profiles["pusch"]["P04-PRB"] | {
        "case_id": "AI-BIG", "seed": 775, "num_transport_blocks": 9,
        "requirement_ids": ["NR-PUSCH-MCS-TBS"],
    }
    candidate = {"case": case, "parent_case_id": "P04-PRB",
                 "related_requirements": ["NR-PUSCH-MCS-TBS"], "risk_tags": [],
                 "risk_reason": "boundary"}
    with pytest.raises(RejectedCase, match="RESOURCE_BUDGET"):
        CaseValidator(registry, request).accept(candidate)
