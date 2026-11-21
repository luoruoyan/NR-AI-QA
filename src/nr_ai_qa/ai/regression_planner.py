"""Select only registered regression cases within the remaining execution budget."""

from nr_ai_qa.ai.case_validator import estimated_seconds, sample_count
from nr_ai_qa.qa.regression_registry import GOLDEN, PRIORITY_MODULES, RULES

KNOWN = {"ldpc": ["BG-R067"], "pusch": []}


def select_regression_tests(change, registry, budget, *, spent_gpu_seconds=0.0):
    family = change.family
    cases, reqs = registry.profiles[family], registry.requirements[family]
    if (
        not set(change.affected_requirements) <= reqs.keys()
        or not set(change.known_failed_case_ids + change.model_suggested_case_ids) <= cases.keys()
        or (change.category == "ldpc_processing") != (family == "ldpc")
        or not set(change.affected_modules) <= {t["module"] for t in registry.config["topics"].values()
                                                if t["family"] == family}
    ):
        raise ValueError("Unknown or out-of-scope regression reference")

    mandatory = list(dict.fromkeys(change.known_failed_case_ids + KNOWN[family] + [GOLDEN[family]]))
    rows, skipped, seen = [], [], set()
    estimate, blocked = 8.0, False

    def consider(case_id, reason, priority):
        nonlocal estimate, blocked
        if case_id in seen:
            return
        seen.add(case_id)
        case = cases[case_id]
        cost = estimated_seconds(case)
        if case["test_type"] in {"config_contract", "iq_contract"}:
            skipped.append({"case_id": case_id, "selection_reason": reason, "reason": "HOST_ONLY_CONTRACT"})
            blocked = blocked or case_id in mandatory
            return
        if (len(rows) >= budget.max_regression_cases or sample_count(case) > budget.max_tb_per_case
                or spent_gpu_seconds + estimate + cost > budget.estimated_gpu_seconds):
            skipped.append({"case_id": case_id, "selection_reason": reason, "reason": "RESOURCE_BUDGET"})
            blocked = blocked or case_id in mandatory
            return
        estimate += cost
        past = registry.evidence[family].get(case_id, {})
        rows.append({
            "case_id": case_id, "priority": priority, "selection_reason": reason,
            "requirement_ids": case["requirement_ids"], "estimated_test_cost_seconds": cost,
            "prior_status": past.get("status"), "prior_evidence_path": past.get("path"),
        })

    for cid in mandatory:
        consider(cid, "Key Golden baseline" if cid == GOLDEN[family] else "Known observation replay", "P0")
    for cid in RULES[change.category]:
        consider(cid, f"Curated {change.category} regression path", "P1")
    for module in change.affected_modules:
        for cid in PRIORITY_MODULES.get(module, ()):
            consider(cid, f"Affected module {module}", "P1")
    for cid in change.model_suggested_case_ids:
        consider(cid, "Validated advisory regression reference", "P2")
    affected = set(change.affected_requirements)
    related = [(cid, case) for cid, case in cases.items() if affected & set(case["requirement_ids"])]
    related.sort(key=lambda v: (
        -sum(v[1].get(k) == val for k, val in change.known_failure_parameters.items()), v[0],
    ))
    for cid, _ in related:
        consider(cid, "Traceable affected requirement or matching input", "P2")

    return {
        "status": "BLOCKED" if blocked else "READY", "change_context": change.model_dump(mode="json"),
        "selected_case_ids": [row["case_id"] for row in rows], "selected": rows,
        "estimated_gpu_seconds": round(estimate, 3), "skipped": skipped,
    }
