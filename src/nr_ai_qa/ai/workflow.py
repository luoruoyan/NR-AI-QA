"""End-to-end advisory QA orchestration with per-workflow evidence."""

from collections import Counter
from pathlib import Path
import time

from nr_ai_qa.ai.coverage_analyzer import apply_advice
from nr_ai_qa.ai.defect_analyzer import analyze_defects, bug_report, collect_evidence, KNOWN
from nr_ai_qa.ai.llm_client import LlmError
from nr_ai_qa.ai.regression_planner import select_regression_tests
from nr_ai_qa.ai.risk_analyzer import rank_risks
from nr_ai_qa.ai.schemas import AdviceResponse, ChangeContext, TriageResponse
from nr_ai_qa.ai.test_generator import generate_test_cases, make_request
from nr_ai_qa.qa.reporting import json_text

STAGES = (
    "coverage", "risk", "generation", "validation", "gpu_execution", "oracle", "triage",
    "regression_selection", "regression_execution", "final_report",
)


def _counts(rows):
    return dict(Counter(row["status"] for row in rows))


def _change_context(suite, risks, analyses, registry, change_category=None):
    risks = [row for row in risks if row["family"] == suite]
    related = sorted({rid for row in risks[:4] for rid in row["requirement_ids"]})
    model_ids = list(dict.fromkeys(cid for row in analyses for cid in row["suggested_regression_tests"]))
    known = ["BG-R067"] if suite == "ldpc" else []
    source = registry.evidence[suite][known[0]]["row"]["test_parameters"] if known else {}
    params = {k: source[k] for k in ("tb_size", "code_rate", "rv") if k in source}
    return ChangeContext(
        family=suite, category=change_category or ("ldpc_processing" if suite == "ldpc" else "noise_variance"),
        affected_requirements=related, affected_modules=[r["module"] for r in risks[:4]],
        known_failed_case_ids=known, model_suggested_case_ids=model_ids,
        known_failure_parameters=params,
    )


def _json(directory: Path, name: str, workflow_id: str, payload: dict):
    path = directory / name
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json_text({"workflow_id": workflow_id, **payload}) + "\n", encoding="utf-8")
    tmp.replace(path)
    return name


def run_workflow(registry, client, executor, directory, *, suite, max_cases, seed=320601,
                 change_category=None, requirement_ids=None, risk_topics=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    workflow_id = directory.name
    stages = dict.fromkeys(STAGES, "NOT_RUN")
    started = time.monotonic()
    artifacts = {}
    summary = {
        "suite": suite, "mode": client.mode, "prompt_version": client.prompt_version,
        "execution_backend": executor.backend, "status": "RUNNING", "stages": stages,
        "artifacts": artifacts, "blocker": None,
    }

    def save(name, payload):
        artifacts[name] = _json(directory, name, workflow_id, payload)

    def finish(status, blocker=None):
        summary.update(status=status, blocker=blocker, model_id=client.model_id,
                       duration_seconds=round(time.monotonic() - started, 6),
                       live_verified=client.mode == "live" and status in {"COMPLETED", "COMPLETED_WITH_FINDINGS"},
                       replay_source=getattr(client, "replay_source", None))
        if client.calls:
            save("model_calls.json", {"calls": client.calls, "token_usage": getattr(client, "token_usage", 0),
                                      "estimated_cost_usd": getattr(client, "estimated_cost_usd", 0)})
        if getattr(client, "recording", None):
            save("recorded_responses.json", {"format_version": 1, "source": "live_capture",
                                             "prompt_version": client.prompt_version,
                                             "responses": client.recording})
        stages["final_report"] = "COMPLETED"
        _json(directory, "workflow_summary.json", workflow_id, summary)
        return {"workflow_id": workflow_id, **summary}

    def start(name):
        stages[name] = "RUNNING"
        _json(directory, "workflow_summary.json", workflow_id, summary)

    def stop(name, blocker):
        stages[name] = "BLOCKED"
        return finish("BLOCKED", blocker)

    active = "coverage"
    try:
        start(active)
        baseline = registry.snapshot()
        request = make_request(registry, suite, max_cases, seed, baseline,
                               requirement_ids=requirement_ids, risk_topics=risk_topics)
        budget = request.execution_budget
        deadline = started + budget.wall_seconds
        client.deadline = deadline
        topics = [{
            "topic_id": r["topic_id"], "requirement_ids": r["requirement_ids"], "score": r["score"],
            "state": baseline["topics"][r["topic_id"]]["state"],
            "parameters": r["recommended_parameters"],
        } for r in rank_risks(baseline) if r["family"] == suite]
        requirement_facts = [{
            "id": rid, "clause": req["clause"], "preconditions": req.get("preconditions"),
        } for rid, req in registry.requirements[suite].items() if req["coverage"] == "implemented"]
        save("coverage.json", {"baseline": baseline})
        response = client.request("coverage", suite, {"topics": topics, "requirements": requirement_facts},
                                  AdviceResponse)
        advice = apply_advice(response, registry, suite)
        if hasattr(client, "validated"):
            accepted = [{k: v for k, v in row.items() if k != "explanation"} for row in advice["accepted"]]
            client.validated(not advice["rejected"], {"advice": accepted} if not advice["rejected"] else None)
        save("coverage.json", {"baseline": baseline, "model_recommendations": advice})
        stages[active] = "COMPLETED"

        active = "risk"
        start(active)
        risks = rank_risks(baseline, advice["accepted"])
        save("risk_ranking.json", {"ranking": risks})
        stages[active] = "COMPLETED"

        active = "generation"
        start(active)
        generated = generate_test_cases(request, registry, client)
        save("generated_cases.json", generated)
        stages[active] = "COMPLETED"
        active = "validation"
        start(active)
        save("validation.json", {
            "accepted_case_ids": [r["case"]["case_id"] for r in generated["accepted"]],
            "rejected": generated["rejected"], "repair_attempts": generated["attempts"] - 1,
        })
        if not generated["accepted"]:
            return stop(active, "No generated case passed schema and execution qualification")
        stages[active] = "COMPLETED"

        active = "gpu_execution"
        start(active)
        results, commands = [], []
        for candidate in generated["accepted"]:
            execution = executor.generated(candidate, directory / "generated", deadline)
            results += execution["records"]
            commands += execution["commands"]
            if execution["status"] == "ENVIRONMENT_ERROR":
                save("execution_results.json", {"records": results, "commands": commands, "failure": execution})
                return stop(active, execution["reason"])
        save("execution_results.json", {"records": results, "commands": commands, "status_counts": _counts(results)})
        stages[active] = "COMPLETED"
        summary["generation"] = {
            "accepted": len(generated["accepted"]), "rejected": len(generated["rejected"]),
            "executed": len(results), "real_gpu": sum(r.get("real_gpu", False) for r in results),
            "status_counts": _counts(results),
        }

        active = "oracle"
        start(active)
        save("oracle_results.json", {"verdicts": [
            {"case_id": row["case_id"], "status": row["status"], "classification": row["execution_class"],
             "evidence_path": row.get("evidence_path")} for row in results
        ]})
        save("coverage.json", {"baseline": baseline, "after_gpu": registry.snapshot(results),
                               "model_recommendations": advice})
        stages[active] = "COMPLETED"

        active = "triage"
        start(active)
        if time.monotonic() >= deadline:
            return stop(active, "Workflow wall time budget exhausted")
        targets = collect_evidence(registry, suite, results)
        save("defect_analysis.json", {"input_evidence": targets})
        response = client.request("triage", suite, {"cases": list(targets.values())}, TriageResponse)
        triage = analyze_defects(targets, response)
        if hasattr(client, "validated"):
            client.validated(not triage["rejected"], triage["validated_response"] if not triage["rejected"] else None)
        historical = {family: collect_evidence(registry, family, []) for family in KNOWN}
        save("defect_analysis.json", {"input_evidence": targets, "historical_cases": historical, **triage})
        (directory / "bug_report.md").write_text(
            bug_report(triage["analyses"], targets, workflow_id), encoding="utf-8"
        )
        artifacts["bug_report.md"] = "bug_report.md"
        stages[active] = "COMPLETED"

        active = "regression_selection"
        start(active)
        context = _change_context(suite, risks, triage["analyses"], registry, change_category)
        plan = select_regression_tests(context, registry, budget, spent_gpu_seconds=generated["estimated_gpu_seconds"])
        save("regression_plan.json", plan)
        if plan["status"] != "READY":
            return stop(active, "A mandatory reproduction or Golden case is outside the GPU regression scope or budget")
        stages[active] = "COMPLETED"

        active = "regression_execution"
        start(active)
        regression = executor.regression(suite, plan["selected_case_ids"], directory / "regression", deadline)
        observed = regression["records"]
        prior = {row["case_id"]: row["prior_status"] for row in plan["selected"]}
        persistent = [r["case_id"] for r in observed
                      if r["status"] == "POTENTIAL_SDK_DEFECT" and r["status"] == prior[r["case_id"]]]
        changes = [{"case_id": r["case_id"], "baseline_status": prior[r["case_id"]], "actual_status": r["status"]}
                   for r in observed if prior[r["case_id"]] and prior[r["case_id"]] != r["status"]]
        summary["regression"] = {
            "selected": len(plan["selected_case_ids"]), "executed": len(observed),
            "real_gpu": sum(r.get("real_gpu", False) for r in observed), "status_counts": _counts(observed),
            "persistent_observations": persistent, "status_changes": changes,
        }
        save("regression_results.json", {**regression, "summary": summary["regression"]})
        if regression["status"] == "ENVIRONMENT_ERROR":
            return stop(active, regression["reason"])
        stages[active] = "COMPLETED"
        unsafe_results = [r for r in results + observed if r["status"] not in
                          {"PASS", "EXPECTED_DETECTION", "INCONCLUSIVE"} and r["case_id"] not in persistent]
        summary["new_or_unexpected_findings"] = [r["case_id"] for r in unsafe_results]
        findings = persistent or unsafe_results or changes or triage["rejected"] or advice["rejected"]
        return finish("COMPLETED_WITH_FINDINGS" if findings else "COMPLETED")
    except (LlmError, OSError, ValueError, KeyError) as exc:
        if client.calls and client.calls[-1].get("structure_validation") == "PENDING":
            client.validated(False)
        return stop(active, str(exc))
    except KeyboardInterrupt:
        return stop(active, "Workflow interrupted")


def unavailable_summary(directory, suite, reason, prompt_version):
    directory.mkdir(parents=True, exist_ok=False)
    status = {"mode": "live", "suite": suite, "status": "LIVE_NOT_VERIFIED", "live_verified": False,
              "blocker": reason, "prompt_version": prompt_version, "stages": dict.fromkeys(STAGES, "NOT_RUN")}
    _json(directory, "workflow_summary.json", directory.name, status)
    return {"workflow_id": directory.name, **status}
