"""Project recorded QA results into a small, portable evidence snapshot."""

import json
from pathlib import Path, PurePosixPath
import re

ROOT = Path(__file__).resolve().parents[3]
SNAPSHOT = Path("data/replay/portfolio.json")
SCHEMA_VERSION = 1
DEMO_WORKFLOW = "20261121T075721Z-live-pusch-774b8bcb"
WORKFLOWS = (
    "20261121T073203Z-mock-ldpc-18f303ea",
    "20261121T073219Z-replay-pusch-62fea579",
    "20261121T075331Z-live-pusch-163470d1",
    DEMO_WORKFLOW,
    "20261121T075827Z-replay-pusch-07a7a402",
)
BASELINE_LOGS = {
    "gpu": "logs/stage1-container-dNPBQjlN.log",
    "host": "logs/stage4-baseline-ai.log",
}
STANDARDS = {
    "TS 38.211": "17.10.0", "TS 38.212": "17.13.0",
    "TS 38.214": "17.13.0", "TR 38.901": "17.1.0",
}
CASE_FIELDS = (
    "kind", "case_id", "test_type", "requirement_ids", "tb_size", "mcs", "code_rate", "rv",
    "num_prbs", "num_rx_ant", "num_symbols", "data_re_per_prb", "dmrs_config", "channel_type",
    "snr_db", "esno_db", "samples", "num_transport_blocks", "seed", "risk_tags", "expected_behavior", "fault",
)
DOMAIN_FIELDS = ("scrambling_id", "mapping_type", "config_type", "symbol", "additional_positions",
                 "cdm_groups_without_data")


def source_path(value: str) -> str:
    """Validate a project-relative provenance reference. It is never a download URL."""
    if not isinstance(value, str):
        raise ValueError("Invalid evidence reference type")
    path = PurePosixPath(value)
    if (path.is_absolute() or str(path) != value or len(path.parts) < 2
            or path.parts[0] not in {"reports", "logs"} or ".." in path.parts
            or not re.fullmatch(r"[A-Za-z0-9._/-]+", value)):
        raise ValueError("Unsafe evidence reference")
    return value


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_snapshot(path: Path | None = None) -> dict:
    data = _read(path or ROOT / SNAPSHOT)
    if data.get("schema_version") != SCHEMA_VERSION or data.get("kind") != "SANITIZED_HISTORICAL_EVIDENCE":
        raise ValueError("Unsupported portfolio snapshot")
    return data


def _select(record: dict, keys) -> dict:
    return {key: record[key] for key in keys if key in record}


def _case(case: dict) -> dict:
    cfg = _select(case, CASE_FIELDS)
    if "dmrs_config" in cfg:
        cfg["dmrs_config"] = _select(case["dmrs_config"], DOMAIN_FIELDS)
    return cfg


def _evidence(value, workflow_id=None):
    return {"path": source_path(value), "workflow_id": workflow_id} if value else None


def _result(row, default_dir, workflow_id=None):
    result = _select(row, ("case_id", "requirement_ids", "GPU", "CUDA_version", "SDK_commit", "seed",
                           "status", "duration", "error_class", "execution_class"))
    result["parameters"] = _case(row["test_parameters"])
    path = row.get("evidence_path")
    if path and not path.startswith("reports/"):
        path = f"{default_dir}/{path}"
    result["evidence"] = _evidence(path, workflow_id)
    actual = row["actual_result"]
    recorded_gpu = row.get("real_gpu", bool(actual.get("gpu_execution_started") and row.get("GPU") == "NVIDIA H20"))
    result["execution"] = "REAL_GPU" if recorded_gpu else "HOST"
    result["observations"] = _select(actual, ("native_bg", "sdk_helper_bg", "tb_crc", "bit_errors",
                                             "samples", "ber", "bler", "crc_failures", "payload_errors",
                                             "tb_errors", "detections", "clean_passes", "bler_ci95_wilson",
                                             "ber_ci95_tb_bootstrap", "invalid_input_status"))
    if actual.get("frames") and all("crc" in f for f in actual["frames"]):
        frames = actual["frames"]
        result["observations"] |= {
            "tb_crc_values": [f["crc"] for f in frames],
            "tb_bit_errors": [f["bit_errors"] for f in frames],
        }
    return result


def _coverage(source):
    suites = {}
    for suite, value in source["suites"].items():
        reqs = {}
        for rid, row in value["requirements"].items():
            reqs[rid] = _select(row, ("clause", "kind", "state", "case_ids")) | {
                "evidence": [_select(e, ("case_id", "status", "basis", "qualifies")) |
                             {"source": _evidence(e["path"])} for e in row["evidence"]],
            }
        suites[suite] = {"requirements": reqs, "metrics": value["metrics"]}
    return suites


def _risk(rows, workflow_id):
    return [_select(r, ("topic_id", "family", "module", "requirement_ids", "score", "factors",
                        "contributions", "historical_anomalies", "controlled_fault_cases",
                        "recommended_parameters", "model_advice")) |
            {"source": _evidence(f"reports/ai/{workflow_id}/risk_ranking.json", workflow_id)}
            for r in rows]


def _defects(historical, analyses, workflow_id):
    rows = {}
    for suite, records in historical.items():
        for case_id, row in records.items():
            rows[case_id] = _select(row, ("case_id", "requirement_ids", "execution_status", "failure_category",
                                         "suspected_module", "oracle_type", "expected", "actual", "difference")) | {
                "family": suite, "parameters": _case(row["test_parameters"]),
                "evidence": _evidence(row["evidence_path"]),
                "logs": [_evidence(x) for x in row["relevant_logs"]],
                "observation": row["observations"]["EV-OBSERVATION"], "assessments": [],
            }
    for row in analyses:
        if row["case_id"] in rows:
            projected = _select(row, ("model_review", "confidence", "root_cause_hypotheses",
                                      "suggested_investigation", "suggested_regression_tests", "uncertainty"))
            projected["source"] = _evidence(f"reports/ai/{workflow_id}/defect_analysis.json", workflow_id)
            rows[row["case_id"]]["assessments"].append(projected)
    return rows


def _timeline(calls, generation, regression):
    ordered = []
    ordered.extend((f"Model · {c['purpose']}", c["duration_seconds"], "Model")
                   for c in calls if c["purpose"] != "triage")
    ordered.extend(("Generated case execution", c["wall_seconds"], "GPU") for c in generation
                   if "wall_seconds" in c)
    ordered.extend(("Model · triage", c["duration_seconds"], "Model")
                   for c in calls if c["purpose"] == "triage")
    ordered.extend(("Selected regression execution", c["wall_seconds"], "GPU") for c in regression
                   if "wall_seconds" in c)
    cursor, rows = 0.0, []
    for index, (label, duration, kind) in enumerate(ordered, 1):
        rows.append({"activity": f"{index}. {label}", "from": cursor, "to": cursor + duration,
                     "duration": duration, "kind": kind})
        cursor += duration
    return rows


def project_workflow(directory: Path) -> dict:
    """Read local Stage 3 artifacts without exporting request IDs, raw logs or arrays."""
    def artifact(name):
        file = directory / f"{name}.json"
        return _read(file) if file.is_file() else {}

    summary = artifact("workflow_summary")
    workflow_id = summary["workflow_id"]
    if directory.name != workflow_id or not re.fullmatch(r"[A-Za-z0-9-]+", workflow_id):
        raise ValueError("Unexpected workflow ID")
    prefix = f"reports/ai/{workflow_id}"
    generated, execution, cover = (artifact(n) for n in ("generated_cases", "execution_results", "coverage"))
    regression, plan = artifact("regression_results"), artifact("regression_plan")
    calls = artifact("model_calls").get("calls", [])
    projected = _select(summary, ("mode", "suite", "status", "model_id", "prompt_version", "live_verified",
                                  "duration_seconds", "stages", "generation", "regression"))
    projected["id"] = workflow_id
    projected["source"] = _evidence(f"{prefix}/workflow_summary.json", workflow_id)
    projected["model_calls"] = sum(bool(c.get("request_id")) for c in calls)
    projected["model_tokens"] = sum((c.get("tokens") or {}).get("total", 0) for c in calls)
    execution_by_id = {r["case_id"]: _result(r, prefix, workflow_id) for r in execution.get("records", [])}
    projected["generated"] = [
        {"case": _case(r["case"]), "metadata": _select(r["metadata"], (
            "generation_source", "model_id", "prompt_version", "risk_tags", "risk_reason", "parent_case_id",
            "related_requirements", "risk_explanation")), "estimate": r["estimated_gpu_seconds"],
         "execution": execution_by_id.get(r["case"]["case_id"])} for r in generated.get("accepted", [])
    ]
    projected["rejected"] = [_select(r, ("attempt", "index", "code", "detail"))
                             for r in generated.get("rejected", [])]
    outcomes = {r["case_id"]: _result(r, prefix, workflow_id) for r in regression.get("records", [])}
    projected["regressions"] = [
        _select(r, ("case_id", "priority", "selection_reason", "requirement_ids", "prior_status",
                    "estimated_test_cost_seconds")) |
        {"execution": outcomes.get(r["case_id"]), "prior_evidence": _evidence(r.get("prior_evidence_path"))}
        for r in plan.get("selected", [])
    ]
    before, after = cover.get("baseline", {}).get("topics", {}), cover.get("after_gpu", {}).get("topics", {})
    projected["scenario_changes"] = [
        {"topic_id": topic, "before": before[topic]["state"], "after": item["state"],
         "case_ids": item["case_ids"], "requirement_ids": item["requirement_ids"],
         "source": _evidence(f"{prefix}/coverage.json", workflow_id)}
        for topic, item in after.items() if topic in before and item["state"] != before[topic]["state"]
    ]
    projected["timeline"] = _timeline(calls, execution.get("commands", []), regression.get("commands", []))
    return projected


def _test_count(root, path):
    matches = re.findall(r"\b(\d+) passed(?:,| in )", (root / path).read_text(encoding="utf-8"))
    if len(matches) != 1:
        raise ValueError("Expected one passing pytest summary in the baseline log")
    return {"passed": int(matches[0]), "source": _evidence(path)}


def build_snapshot(root: Path = ROOT) -> dict:
    """Build the committed replay projection from the pinned historical inputs."""
    main = root / "reports/ai" / DEMO_WORKFLOW
    originals = {suite: _read(root / f"reports/{suite}/results.json") for suite in ("ldpc", "pusch")}
    example = originals["pusch"][0]
    environment = example["actual_result"]["environment"]
    cases, statistics = {}, {}
    for suite, rows in originals.items():
        cases[suite] = {r["case_id"]: _result(r, f"reports/{suite}") for r in rows}
        key = "esno_db" if suite == "ldpc" else "snr_db"
        statistics[suite] = [
            cases[suite][r["case_id"]] | {
                "db": r["test_parameters"][key],
                "snr_definition": ("Es=1; complex noise variance = 10^(-Es/N0 / 10)" if suite == "ldpc"
                                   else r["actual_result"]["snr_definition"]),
            } for r in rows if r["test_parameters"]["test_type"] == "awgn"
        ]
    workflows = [project_workflow(root / "reports/ai" / wf) for wf in WORKFLOWS]
    live_source = _read(main / "defect_analysis.json")
    defects = _defects(live_source["historical_cases"], live_source["analyses"], DEMO_WORKFLOW)
    mock = _read(root / "reports/ai" / WORKFLOWS[0] / "defect_analysis.json")
    for case_id, row in _defects(mock["historical_cases"], mock["analyses"], WORKFLOWS[0]).items():
        if row["assessments"]:
            defects[case_id]["assessments"].extend(row["assessments"])
    return {
        "schema_version": SCHEMA_VERSION, "kind": "SANITIZED_HISTORICAL_EVIDENCE",
        "environment": {"GPU": example["GPU"], "compute_capability": environment["compute_capability"],
                        "CUDA": example["CUDA_version"], "pyAerial": environment["pyaerial"], "Aerial": "26-2",
                        "SDK_commit": example["SDK_commit"], "image_digest": environment["image_digest"]},
        "standards": STANDARDS, "test_counts": {k: _test_count(root, v) for k, v in BASELINE_LOGS.items()},
        "latest_workflow_id": workflows[-1]["id"], "demo_workflow_id": DEMO_WORKFLOW,
        "coverage": _coverage(_read(main / "coverage.json")["baseline"]),
        "coverage_source": _evidence(f"reports/ai/{DEMO_WORKFLOW}/coverage.json", DEMO_WORKFLOW),
        "ranking": _risk(_read(main / "risk_ranking.json")["ranking"], DEMO_WORKFLOW),
        "cases": cases, "statistics": statistics, "defects": defects, "workflows": workflows,
    }


if __name__ == "__main__":
    output = ROOT / SNAPSHOT
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(build_snapshot(), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                      encoding="utf-8")
    print(output.relative_to(ROOT))
