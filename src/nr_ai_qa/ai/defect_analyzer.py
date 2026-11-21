"""Constrain advisory triage to observed facts and qualified hypotheses."""

from copy import deepcopy

from pydantic import ValidationError

from nr_ai_qa.ai.schemas import StrictModel, TriageProposal

KNOWN = {
    "ldpc": ("BG-R067", "FAULT-RM-FLIP", "FAULT-CRC-FLIP"),
    "pusch": ("P05-DMRS-FAULT", "P05-NOISE-FAULT", "P03-SMOKE-LOW"),
}
BASELINE_LOGS = {"ldpc": "logs/stage1-container-KlBL3hnD.log",
                 "pusch": "logs/stage1-container-6i5takfI.log"}
CATALOG = {
    "BG-R067": (
        "ldpc_base_graph_helper", "BG_THRESHOLD", "TRACE_RATE_CONVERSION",
        "GPU BG2 with native threshold 0.67f; Python helper BG1 with threshold 2/3. "
        "The impact on final TB is unverified.",
        ["BG-R067", "BG-A3824", "GOLD-QPSK"],
    ),
    "FAULT-RM-FLIP": (
        "rate_matching", "RM_BIT_INJECTION", "COMPARE_RM_BEFORE_DECODER",
        "Controlled single-bit RM difference; LDPC recovered the payload and final TB CRC was zero.",
        ["FAULT-RM-FLIP", "RV-BG2-REPEAT", "GOLD-QPSK"],
    ),
    "FAULT-CRC-FLIP": (
        "crc_checker", "PRE_CRC_INJECTION", "COMPARE_PRE_CRC_PAYLOAD",
        "Controlled post-decode payload corruption was detected by the native CRC checker.",
        ["FAULT-CRC-FLIP", "GOLD-QPSK"],
    ),
    "P05-DMRS-FAULT": (
        "pusch_dmrs_configuration", "DMRS_MISMATCH", "CHECK_DMRS_ID_PAIR",
        "CONTROLLED FAULT-DMRS-CONFIG: RX-only DMRS mismatch changed CRC/payload in 8/8 pairs.",
        ["P05-DMRS-FAULT", "P04-DMRS", "P01-GOLDEN"],
    ),
    "P05-NOISE-FAULT": (
        "pusch_equalizer_noise_input", "LLR_SCALING", "PAIR_CLEAN_LLR",
        "CONTROLLED FAULT-NOISE-VARIANCE: active LLRs changed; BLER impact has not been established.",
        ["P05-NOISE-FAULT", "P02-CONSISTENCY", "P03-MID"],
    ),
    "P03-SMOKE-LOW": (
        "pusch_awgn_measurement", "LOW_SNR_CHANNEL", "RECHECK_WILSON_INTERVAL",
        "Low-SNR CRC/payload errors are observations from a small statistical sample.",
        ["P03-MID", "P01-GOLDEN"],
    ),
}
HYPOTHESES = {
    "BG_THRESHOLD": "Check whether wrapper and native rate thresholds can explain the boundary observation.",
    "RM_BIT_INJECTION": "Check whether the injected bit difference explains RM evidence and decoder recovery.",
    "PRE_CRC_INJECTION": "Check whether the controlled pre-CRC flip explains the CRC syndrome.",
    "DMRS_MISMATCH": "Check whether RX pilot-reference mismatch explains the paired payload outcome.",
    "LLR_SCALING": "Check whether the equalizer input bias explains the paired LLR magnitude change.",
    "LOW_SNR_CHANNEL": "Check whether sampling at low SNR explains the observed error counts.",
    "LOCALIZE_MISMATCH": "Localize the first differing observable on a repeat run.",
}
INVESTIGATIONS = {
    "TRACE_RATE_CONVERSION": "Compare the helper threshold and native float conversion at the boundary.",
    "COMPARE_RM_BEFORE_DECODER": "Replay the orthogonal RM bit check before decoding.",
    "COMPARE_PRE_CRC_PAYLOAD": "Compare decoded and deliberately modified bits before the native CRC check.",
    "CHECK_DMRS_ID_PAIR": "Inspect clean and altered receive pilot IDs for the saved IQ.",
    "PAIR_CLEAN_LLR": "Repeat clean and biased equalization on identical IQ, then compare active LLR values.",
    "RECHECK_WILSON_INTERVAL": "Check the sample count, SNR definition and Wilson interval.",
    "REPLAY_EXACT_INPUT": "Reproduce validated parameters with the saved seed and compare evidence.",
}


class DefectAnalysis(StrictModel):
    case_id: str
    execution_status: str
    failure_category: str
    suspected_module: str
    confirmed_sdk_defect: bool
    requirement_ids: list[str]
    root_cause_hypotheses: list[dict]
    supporting_evidence: list[dict]
    suggested_investigation: list[dict]
    suggested_regression_tests: list[str]
    confidence: str
    uncertainty: str
    model_review: str


def outcome_category(row: dict) -> str:
    status, error = row["status"], row.get("error_class")
    case_type = row.get("test_parameters", {}).get("test_type")
    if status == "EXPECTED_DETECTION" or case_type in {"fault", "dmrs_fault", "noise_fault"}:
        return "CONTROLLED_INJECTION"
    if status == "POTENTIAL_SDK_DEFECT":
        return status
    if status == "INCONCLUSIVE":
        return "STATISTICAL_UNCERTAINTY" if case_type == "awgn" else "INCONCLUSIVE"
    if status in {"UNSUPPORTED", "UNSUPPORTED_CONFIGURATION"}:
        return "UNSUPPORTED"
    if status == "CONFIGURATION_ERROR":
        return "INVALID_CASE"
    if error in {"ENVIRONMENT_ERROR", "SDK_ERROR"}:
        return "SDK_RUNTIME_ERROR" if error == "SDK_ERROR" else error
    if status in {"FAIL", "ERROR"}:
        return "ORACLE_MISMATCH"
    return "PASS"


def _facts(row):
    a, e = row.get("actual_result", {}), row.get("expected_result", {})
    cid = row["case_id"]
    if cid == "BG-R067":
        actual = {k: a[k] for k in ("native_bg", "sdk_helper_bg")}
        return {"native_bg": e["native_bg"], "sdk_helper_bg": e["sdk_helper_bg"]}, actual, {
            "expected_helper": e["sdk_helper_bg"], "observed_helper": a["sdk_helper_bg"],
        }
    if cid in {"FAULT-RM-FLIP", "FAULT-CRC-FLIP"}:
        return {"tb_crc": 0, "bit_errors": 0}, {k: a[k] for k in ("tb_crc", "bit_errors")}, a["fault"]["first_difference"]
    if cid in {"P05-DMRS-FAULT", "P05-NOISE-FAULT"}:
        frame = a["frames"][a["first_detected_frame"]]
        difference = frame.get("llr_difference") if cid.endswith("NOISE-FAULT") else frame.get("payload_difference")
        return {"output_measure": e["output_measure"]}, {
            k: a[k] for k in ("samples", "detections", "crc_failures", "payload_errors")
        }, difference
    if row["test_parameters"].get("test_type") == "awgn":
        return {k: e[k] for k in ("minimum_samples", "min_samples") if k in e}, {
            k: a[k] for k in ("samples", "crc_failures", "ber", "bler", "bler_ci95_wilson") if k in a
        }, None
    observed = {k: a[k] for k in ("checks", "bit_errors", "gpu_execution_started") if k in a}
    if a.get("frames"):
        frame = a["frames"][0]
        observed["first_frame"] = {k: frame[k] for k in ("slot", "crc", "bit_errors", "valid_output") if k in frame}
    expected = {k: e[k] for k in ("bit_errors", "crc", "tb_crc") if k in e}
    return expected, observed, a.get("differences")


def collect_evidence(registry, family, execution_rows):
    sources = {cid: (registry.evidence[family][cid]["row"], registry.evidence[family][cid]["path"])
               for cid in KNOWN[family]}
    for row in execution_rows:
        if outcome_category(row) != "PASS":
            sources[row["case_id"]] = (row, row["evidence_path"])
    results = {}
    for case_id, (row, path) in sources.items():
        module, hyp, investigation, observation, regression = CATALOG.get(case_id, (
            f"{family}_adapter_or_oracle", "LOCALIZE_MISMATCH", "REPLAY_EXACT_INPUT",
            "Inspect run evidence before assigning a product cause.",
            ["GOLD-QPSK"] if family == "ldpc" else ["P01-GOLDEN"],
        ))
        expected, actual, difference = _facts(row)
        params = {k: v for k, v in row["test_parameters"].items()
                  if k in {"tb_size", "code_rate", "mcs", "num_prbs", "rv", "seed", "dmrs_config",
                           "snr_db", "esno_db", "num_transport_blocks", "samples", "channel_type"}}
        results[case_id] = {
            "case_id": case_id, "requirement_ids": row["requirement_ids"],
            "execution_status": row["status"], "failure_category": outcome_category(row),
            "suspected_module": module, "test_parameters": params,
            "oracle_type": row.get("expected_result", {}).get("oracle_type", "Stage 1 Independent Oracle"),
            "expected": expected, "actual": actual, "difference": difference,
            "GPU": row["GPU"], "SDK_commit": row["SDK_commit"], "CUDA_version": row["CUDA_version"],
            "evidence_path": path, "relevant_logs": [
                row.get("runner_log", BASELINE_LOGS[family]),
                *(["logs/stage1-container-G7dcJiJW.log"] if case_id == "BG-R067" else []),
            ],
            "observations": {"EV-OBSERVATION": observation,
                             "EV-DIFFERENCE": difference, "EV-STATUS": row["status"]},
            "allowed_hypothesis_ids": [hyp],
            "allowed_evidence_ids": ["EV-OBSERVATION", "EV-DIFFERENCE", "EV-STATUS"],
            "allowed_investigation_ids": [investigation], "allowed_regression_case_ids": regression,
        }
    return deepcopy(results)


def analyze_defects(targets, payload):
    items = payload.get("assessments")
    if set(payload) != {"assessments"} or not isinstance(items, list) or len(items) > 24:
        raise ValueError("Invalid triage response envelope")
    accepted, rejected = {}, []
    for index, item in enumerate(items):
        try:
            proposal = TriageProposal.model_validate(item)
            target = targets.get(proposal.case_id)
            if target is None or proposal.case_id in accepted:
                raise ValueError("Unknown or duplicate case")
            for supplied, name in (
                (proposal.hypothesis_ids, "allowed_hypothesis_ids"),
                (proposal.evidence_ids, "allowed_evidence_ids"),
                (proposal.investigation_ids, "allowed_investigation_ids"),
                (proposal.regression_case_ids, "allowed_regression_case_ids"),
            ):
                if not set(supplied) <= set(target[name]) or len(supplied) != len(set(supplied)):
                    raise ValueError(f"Out-of-scope or duplicate references for {name}")
        except (ValidationError, ValueError) as exc:
            rejected.append({"index": index, "reason": str(exc).splitlines()[0]})
            continue
        accepted[proposal.case_id] = proposal

    analyses = []
    for cid, target in targets.items():
        proposal = accepted.get(cid)
        evidence_ids = proposal.evidence_ids if proposal else ["EV-OBSERVATION"]
        item = DefectAnalysis(
            case_id=cid, execution_status=target["execution_status"], failure_category=target["failure_category"],
            suspected_module=target["suspected_module"], confirmed_sdk_defect=False,
            requirement_ids=target["requirement_ids"],
            root_cause_hypotheses=[{"id": k, "hypothesis": HYPOTHESES[k]}
                                   for k in (proposal.hypothesis_ids if proposal else [])],
            supporting_evidence=[{"id": k, "value": target["observations"][k]}
                                 for k in evidence_ids if target["observations"][k] is not None],
            suggested_investigation=[{"id": k, "action": INVESTIGATIONS[k]}
                                     for k in (proposal.investigation_ids if proposal else [])],
            suggested_regression_tests=proposal.regression_case_ids if proposal else [],
            confidence=proposal.confidence if proposal else "unavailable",
            uncertainty="Hypotheses require reproduction and human review. Execution status remains authoritative.",
            model_review="ACCEPTED" if proposal else "NO_VALID_PROPOSAL",
        )
        analyses.append(item.model_dump())
    return {"analyses": analyses, "rejected": rejected,
            "validated_response": {"assessments": [x.model_dump() for x in accepted.values()]}}


def bug_report(analyses, targets, workflow_id):
    lines = ["# QA triage for human review", "", f"Workflow `{workflow_id}`", ""]
    for result in analyses:
        case_id = result["case_id"]
        src = targets[case_id]
        lines.extend([
            f"## {case_id}", "",
            f"Oracle status `{result['execution_status']}`, classification `{result['failure_category']}`.",
            "", src["observations"]["EV-OBSERVATION"], "",
            f"Module under review `{result['suspected_module']}`. "
            f"Model confidence `{result['confidence']}`.", "",
        ])
        for hypothesis in result["root_cause_hypotheses"]:
            lines.append(f"- Hypothesis — {hypothesis['hypothesis']}")
        for action in result["suggested_investigation"]:
            lines.append(f"- Investigation — {action['action']}")
        lines += ["", f"Evidence `{src['evidence_path']}`.", "", result["uncertainty"], ""]
    return "\n".join(lines)
