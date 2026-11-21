"""Order risk topics using explicit engineering weights and measured evidence."""

WEIGHTS = {
    "boundary_risk": 0.30, "coverage_gap": 0.25,
    "historical_failure": 0.25, "parameter_interaction": 0.20,
}
GAPS = {"COVERED": 0.0, "PARTIALLY_COVERED": 0.5, "NOT_COVERED": 1.0}


def rank_risks(coverage: dict, recommendations: list[dict] | None = None) -> list[dict]:
    advice = {r["topic_id"]: r for r in recommendations or []}
    rows = []
    for topic_id, topic in coverage["topics"].items():
        if topic["state"] not in GAPS:
            continue
        abnormal = [r["case_id"] for r in topic["evidence"]
                    if r["basis"] == "REAL_GPU" and r["status"] in {"POTENTIAL_SDK_DEFECT", "FAIL"}]
        controlled = [r["case_id"] for r in topic["evidence"]
                      if r["basis"] == "REAL_GPU" and r["status"] == "EXPECTED_DETECTION"]
        factors = {
            "boundary_risk": topic["boundary"], "coverage_gap": GAPS[topic["state"]],
            "historical_failure": 1.0 if abnormal else 0.0,
            "parameter_interaction": topic["interaction"],
        }
        parts = {k: factors[k] * w for k, w in WEIGHTS.items()}
        rows.append({
            "topic_id": topic_id, "family": topic["family"], "module": topic["module"],
            "requirement_ids": topic["requirement_ids"], "score": round(sum(parts.values()), 6),
            "factors": factors, "contributions": parts, "historical_anomalies": abnormal,
            "controlled_fault_cases": controlled, "recommended_parameters": topic.get("parameters", {}),
            "model_advice": advice.get(topic_id),
        })
    return sorted(rows, key=lambda row: (-row["score"], row["topic_id"]))
