"""Generate and independently qualify a bounded batch of model-proposed cases."""

from nr_ai_qa.ai.case_validator import CaseValidator, LDPC_ALLOWED, RejectedCase
from nr_ai_qa.ai.coverage_analyzer import REASONS
from nr_ai_qa.ai.llm_client import LlmError
from nr_ai_qa.ai.risk_analyzer import rank_risks
from nr_ai_qa.ai.schemas import Budget, GenerationRequest, LdpcGeneration, PuschGeneration
from nr_ai_qa.qa.runner import validate_case

SOURCES = {"mock": "MOCK", "replay": "RECORDED_REPLAY", "live": "REAL_LLM"}
BASE_EXAMPLES = {
    "ldpc": ["RV-BG2-3", "SEG-BG2-FOUR", "GOLD-QPSK", "BG-A3824", "RV-BG1-3"],
    "pusch": ["P04-PRB", "P04-MCS", "P04-DMRS", "P01-GOLDEN", "P04-SISO"],
}


def make_request(registry, family, max_cases, seed, coverage=None, *, requirement_ids=None, risk_topics=None):
    coverage = registry.snapshot() if coverage is None else coverage
    priorities = [r for r in rank_risks(coverage) if r["family"] == family]
    if risk_topics:
        if set(risk_topics) - {r["topic_id"] for r in priorities}:
            raise ValueError("Unknown or unsupported risk topic for the selected suite")
        priorities = [r for r in priorities if r["topic_id"] in risk_topics]
    eligible = {rid for row in priorities for rid in row["requirement_ids"]}
    if requirement_ids and set(requirement_ids) - eligible:
        raise ValueError("Selected requirements must belong to the selected suite and risk scope")
    requirement_ids = sorted(set(requirement_ids) if requirement_ids else eligible)
    overview = {
        "metrics": coverage["suites"][family]["metrics"],
        "topics": {rid: coverage["topics"][rid]["state"] for rid in
                   (row["topic_id"] for row in priorities)},
    }
    return GenerationRequest(
        test_family=family, requirement_ids=requirement_ids,
        risk_focus=[r["topic_id"] for r in priorities], max_cases=max_cases, seed=seed,
        execution_budget=Budget.model_validate(registry.config["budget"]), existing_coverage=overview,
    )


def _parents(registry, family):
    allowed = LDPC_ALLOWED if family == "ldpc" else {"golden"}
    priorities = BASE_EXAMPLES[family]
    return {name: validate_case(registry.profiles[family][name]).model_dump(mode="json")
            for name in priorities if registry.profiles[family][name]["test_type"] in allowed}


def generate_test_cases(request, registry, client):
    validator = CaseValidator(registry, request)
    family = request.test_family
    response_model = LdpcGeneration if family == "ldpc" else PuschGeneration
    allowed_reqs = set(request.requirement_ids)
    parents = _parents(registry, family)
    eligible = {key: [rid for rid in parent["requirement_ids"] if rid in allowed_reqs]
                for key, parent in parents.items()}
    parents = {key: parent | {"requirement_ids": eligible[key]} for key, parent in parents.items()
               if eligible[key]}
    if not parents:
        raise ValueError("No qualified generation parent covers the requested requirements")
    context = {
        "request": request.model_dump(mode="json"), "qualified_parents": parents,
        "eligible_parent_requirements": {key: eligible[key] for key in parents},
        "rules": (["Preserve the parent LDPC configuration except case ID, seed, requirements and tags."] if
                  family == "ldpc" else [
                      "Golden AWGN only, one TX layer, 1 or 2 RX branches, SNR 20–35 dB",
                      "Use MCS 2 or 4, PRBs 32 or 273, DMRS scrambling ID 41 or 42. "
                      "For MCS 4 and 32 PRBs use TBS 2536 bits.",
                  ]) + [
            "Choose one parent_case_id, then choose case.requirement_ids only from "
            "eligible_parent_requirements for that exact parent.",
            "Use an AI- case ID and a new seed. Risk tags must be IDs from request.risk_focus.",
        ],
    }
    accepted, rejected = [], []
    limit = min(request.max_cases, request.execution_budget.max_cases)
    for attempt in range(1, 4):
        round_errors, validated = [], []
        remaining = limit - len(accepted)
        context["remaining_cases"] = remaining
        context["previous_rejections"] = [{k: e[k] for k in ("code", "detail")} for e in rejected[-8:]]
        context["accepted_case_ids"] = [row["case"]["case_id"] for row in accepted]
        try:
            response = client.request("generation", family, context, response_model)
            items = response.get("candidates")
            if set(response) != {"candidates"} or not isinstance(items, list) or len(items) > 12:
                raise LlmError("Invalid generation envelope")
            for index, row in enumerate(items):
                try:
                    if client.mode == "live" and isinstance(row, dict) and "related_requirements" not in row:
                        proposed_case = row.get("case")
                        if isinstance(proposed_case, dict):
                            row = row | {"related_requirements": proposed_case.get("requirement_ids")}
                    clean = validator.accept(row)
                except RejectedCase as exc:
                    round_errors.append({"attempt": attempt, "index": index,
                                         "code": exc.code, "detail": exc.detail})
                    validated.append({"recorded_invalid_proposal": True})
                    continue
                meta = clean["metadata"]
                validated.append({"case": clean["case"], **meta})
                clean["metadata"] = meta | {
                    "generation_source": SOURCES[client.mode], "model_id": client.model_id,
                    "prompt_version": client.prompt_version,
                    "risk_explanation": REASONS[meta["risk_reason"]],
                }
                accepted.append(clean)
            if not items:
                round_errors.append({"attempt": attempt, "index": None,
                                     "code": "INVALID_CASE", "detail": "Model returned no candidates"})
            if hasattr(client, "validated"):
                client.validated(not round_errors, {"candidates": validated})
        except LlmError as exc:
            if hasattr(client, "validated") and client.calls:
                client.validated(False)
            round_errors.append({"attempt": attempt, "index": None, "code": "INVALID_RESPONSE", "detail": str(exc)})
        rejected.extend(round_errors)
        if len(accepted) >= limit or not round_errors:
            break
    return {"request": request.model_dump(mode="json"), "attempts": attempt, "accepted": accepted,
            "rejected": rejected, "estimated_gpu_seconds": validator.estimate}


def mock_response(purpose, family, context):
    """Locally authored examples used only when the caller explicitly selects mock."""
    if purpose == "coverage":
        return {"advice": [
            {"topic_id": row["topic_id"], "requirement_ids": row["requirement_ids"],
             "reason": "missing_evidence" if row["state"] == "NOT_COVERED" else "boundary"}
            for row in context["topics"][:2]
        ]}
    if purpose == "triage":
        return {"assessments": [
            {"case_id": row["case_id"], "hypothesis_ids": row["allowed_hypothesis_ids"][:1],
             "evidence_ids": row["allowed_evidence_ids"][:2],
             "investigation_ids": row["allowed_investigation_ids"][:1],
             "regression_case_ids": row["allowed_regression_case_ids"][:2], "confidence": "medium"}
            for row in context["cases"]
        ]}
    if purpose != "generation":
        raise LlmError("No mock response for the requested purpose")
    options = {
        "ldpc": [("RV-BG2-3", "R-RV", "LDPC-RV"), ("SEG-BG2-FOUR", "R-SEG", "LDPC-SEG")],
        "pusch": [("P04-PRB", "NR-PUSCH-MCS-TBS", "PUSCH-MCS-PRB"),
                  ("P04-DMRS", "NR-PUSCH-MAPPING", "PUSCH-DMRS")],
    }
    out = []
    for index, (parent, requirement, topic) in enumerate(options[family][:context["remaining_cases"]]):
        case = context["qualified_parents"][parent] | {
            "case_id": f"AI-MOCK-{family.upper()}-{index + 1}", "requirement_ids": [requirement],
            "seed": (context["request"]["seed"] + index) % 2**32, "risk_tags": [topic],
        }
        changed = family == "pusch" and index == 0
        if changed:
            case.update(mcs=4, tb_size=2536)
        out.append({"case": case, "parent_case_id": parent, "related_requirements": [requirement],
                    "risk_tags": [topic], "risk_reason": "interaction" if changed else "seed_variation"})
    return {"candidates": out}
