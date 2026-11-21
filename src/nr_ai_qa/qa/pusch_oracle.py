"""Trusted PUSCH case validation, execution and evidence."""

from pathlib import Path
import shlex
import time
import traceback

import numpy as np
from pydantic import ValidationError

from nr_ai_qa.qa.oracle import first_difference, intervals, mcs_parameters, scheduled_tbs
from nr_ai_qa.qa.reporting import json_text, save_evidence
from nr_ai_qa.qa.runner import ROOT, Result, hash_array, read_yaml
from nr_ai_qa.schemas.pusch_case import PuschCase


class UnsupportedPuschConfiguration(ValueError):
    """A legal parameter combination outside the verified receiver profile."""


def expected_tbs(mcs: int, prbs: int) -> int:
    order, rate = mcs_parameters(mcs)
    return scheduled_tbs(132 * prbs * order, rate)


def load_pusch_profiles() -> dict[str, dict]:
    profiles = read_yaml(ROOT / "configs/pusch_profiles.yaml")
    trace = read_yaml(ROOT / "specs/pusch_traceability.yaml")["test_cases"]
    return {
        name: ({"case_id": name, "requirement_ids": trace[name]} | profiles["defaults"] |
               {"tb_size": expected_tbs(fields.get("mcs", profiles["defaults"]["mcs"]),
                                        fields.get("num_prbs", profiles["defaults"]["num_prbs"]))} |
               fields)
        for name, fields in profiles["cases"].items()
    }


def validate_case(raw: dict | PuschCase) -> PuschCase:
    case = PuschCase.model_validate(raw.model_dump() if isinstance(raw, PuschCase) else raw)
    records = read_yaml(ROOT / "specs/pusch_requirements.yaml")["requirements"]
    active = {r["requirement_id"] for r in records if r["coverage"] == "implemented"}
    if set(case.requirement_ids) - active or len(set(case.requirement_ids)) != len(case.requirement_ids):
        raise ValueError("Unknown, pending or duplicate requirement ID")
    if case.channel_type != "awgn" or case.mcs > 9 or case.num_symbols != 12:
        raise UnsupportedPuschConfiguration("Profile requires AWGN, MCS table 1 QPSK and symbols 2–13")
    if case.tb_size != expected_tbs(case.mcs, case.num_prbs):
        raise ValueError(f"TBS must equal {expected_tbs(case.mcs, case.num_prbs)} bits for this allocation")
    if case.test_type != "awgn" and case.num_transport_blocks > 16:
        raise ValueError("Deterministic and paired cases are limited to 16 TBs")
    return case


def inspect_reception(payload, reception) -> dict:
    shapes = [list(tb.shape) for tb in reception.payloads]
    dtypes = [str(tb.dtype) for tb in reception.payloads]
    valid = (shapes == [list(payload.shape)] and dtypes == ["uint8"] and
             reception.crc.shape == (1,) and reception.crc.dtype == np.uint32)
    bit_errors, first = None, None
    if valid:
        rx = reception.payloads[0]
        bit_errors = int(np.unpackbits(payload ^ rx).sum())
        first = first_difference(np.unpackbits(payload), np.unpackbits(rx))
    return {"valid_output": valid, "payload_shapes": shapes, "payload_dtypes": dtypes,
            "crc_shape": list(reception.crc.shape), "crc_dtype": str(reception.crc.dtype),
            "crc": reception.crc.tolist(), "bit_errors": bit_errors, "first_bit_difference": first}


def golden(adapter, case) -> tuple[str, str | None, dict, dict, dict]:
    tx_cfg, rx_cfg = adapter.configs(case)
    frames, arrays = [], {}
    for index in range(case.num_transport_blocks):
        frame = adapter.prepare(case, index, tx_cfg)
        reception = adapter.receive(frame, rx_cfg)
        iq = frame.iq.get()
        frames.append(inspect_reception(frame.payload, reception) | {
            "slot": frame.slot, "data_re_power": frame.data_re_power,
            "noise_variance": frame.noise_variance, "iq_sha256": hash_array(iq),
            "data_re_count": frame.data_re_count,
        })
        observed = frames[-1]
        unexpected = (not observed["valid_output"] or observed["crc"] != [0] or
                      observed["bit_errors"] != 0 or frame.data_re_count != 132 * case.num_prbs)
        if index == 0 or (unexpected and "failing_frame" not in arrays):
            arrays = {"iq": iq, "tx_payload": frame.payload, "crc": reception.crc}
            if unexpected:
                arrays["failing_frame"] = np.array(index, dtype=np.int32)
            if reception.payloads:
                arrays["rx_payload"] = reception.payloads[0]
    data_re = 132 * case.num_prbs
    passed = all(f["valid_output"] and f["crc"] == [0] and f["bit_errors"] == 0
                 and f["data_re_count"] == data_re for f in frames)
    return ("PASS" if passed else "FAIL", None if passed else "Unexpected TB, CRC, shape or data REs",
            {"oracle_type": "BIT_EXACT", "crc": [0], "bit_errors": 0, "data_re_count": data_re},
            {"frames": frames}, arrays)


def consistent(adapter, case) -> tuple[str, str | None, dict, dict, dict]:
    tx_cfg, rx_cfg = adapter.configs(case)
    frames, arrays = [], {}
    # LLRs are FP16. For EQ symbols the observed source data also changes
    # in half-precision-size steps, independent of the Python output dtype.
    tolerances = {"llrs": {"rtol": 3e-3, "atol": 1e-3},
                  "eq_symbols": {"rtol": 2e-3, "atol": 1e-3}}
    for index in range(case.num_transport_blocks):
        frame = adapter.prepare(case, index, tx_cfg)
        iq = frame.iq.get()
        fused = adapter.receive(frame, rx_cfg)
        separable = adapter.receive(frame, rx_cfg, path="separable", capture=True)
        repeated = adapter.receive(frame, rx_cfg, path="separable", capture=True)
        f_obs = inspect_reception(frame.payload, fused)
        s_obs = inspect_reception(frame.payload, separable)
        r_obs = inspect_reception(frame.payload, repeated)
        shapes_match = (f_obs["payload_shapes"] == s_obs["payload_shapes"] and
                        f_obs["crc_shape"] == s_obs["crc_shape"])
        bit_exact = (f_obs["valid_output"] and s_obs["valid_output"] and
                     np.array_equal(fused.payloads[0], separable.payloads[0]) and
                     np.array_equal(fused.crc, separable.crc))
        numeric = {}
        for name in ("llrs", "eq_symbols"):
            values = separable.intermediate[name]
            other = repeated.intermediate[name]
            dtype = np.float32 if name == "llrs" else np.complex64
            a, b = values.astype(dtype), other.astype(dtype)
            comparable = a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
            numeric[name] = {
                "shape": list(values.shape), "dtype": str(values.dtype),
                "within_tolerance": bool(comparable and np.allclose(a, b, **tolerances[name])),
                "max_abs_difference": float(np.max(np.abs(a - b))) if comparable else None,
            }
        frame_result = {
            "slot": frame.slot, "iq_sha256": hash_array(iq), "fused": f_obs,
            "separable": s_obs, "separable_repeat": r_obs, "output_shapes_match": shapes_match,
            "bit_exact": bit_exact, "separable_repeat_numeric": numeric,
        }
        frames.append(frame_result)
        unexpected = (not bit_exact or not shapes_match or
                      any(not v["within_tolerance"] for v in numeric.values()) or
                      any(o["bit_errors"] != 0 or o["crc"] != [0] for o in (f_obs, s_obs, r_obs)))
        if index == 0 or (unexpected and "failing_frame" not in arrays):
            arrays = {"iq": iq, "tx_payload": frame.payload, "fused_crc": fused.crc,
                      "separable_crc": separable.crc, **separable.intermediate,
                      **{f"repeat_{k}": v for k, v in repeated.intermediate.items()}}
            if unexpected:
                arrays["failing_frame"] = np.array(index, dtype=np.int32)
            if fused.payloads and separable.payloads:
                arrays.update(fused_payload=fused.payloads[0], separable_payload=separable.payloads[0])
    passed = all(
        f["bit_exact"] and f["output_shapes_match"] and
        all(f[p]["valid_output"] and f[p]["bit_errors"] == 0 and f[p]["crc"] == [0]
            for p in ("fused", "separable", "separable_repeat")) and
        set(f["separable_repeat_numeric"]) == {"llrs", "eq_symbols"} and
        all(v["within_tolerance"] for v in f["separable_repeat_numeric"].values())
        for f in frames
    )
    return ("PASS" if passed else "FAIL", None if passed else "Receiver comparison failed",
            {"oracle_type": ["BIT_EXACT", "NUMERIC_TOLERANCE"], "tolerances": tolerances},
            {"frames": frames, "cross_receiver_intermediates": "UNSUPPORTED",
             "intermediate_scope": "Separable repeat on identical IQ; Fused exposes only TB/CRC"},
            arrays)


def awgn(adapter, case) -> tuple[str, str | None, dict, dict, dict]:
    tx_cfg, rx_cfg = adapter.configs(case)
    frames, arrays = [], {}
    start = time.perf_counter()
    for index in range(case.num_transport_blocks):
        frame = adapter.prepare(case, index, tx_cfg)
        reception = adapter.receive(frame, rx_cfg)
        iq = frame.iq.get()
        obs = inspect_reception(frame.payload, reception)
        frames.append(obs | {"slot": frame.slot, "iq_sha256": hash_array(iq),
                             "data_re_power": frame.data_re_power,
                             "noise_variance_per_branch": frame.noise_variance})
        if index == 0 or not obs["valid_output"]:
            arrays = {"iq": iq, "tx_payload": frame.payload, "crc": reception.crc}
            if reception.payloads:
                arrays["rx_payload"] = reception.payloads[0]
        if not obs["valid_output"]:
            return ("FAIL", "Receiver returned malformed CRC or TB", {"oracle_type": "OUTPUT_CONTRACT"},
                    {"frames": frames}, arrays)
    metrics = intervals([f["bit_errors"] for f in frames], case.tb_size, case.seed)
    crc_counts = [int(f["crc"] != [0]) for f in frames]
    crc_metrics = intervals(crc_counts, 1, case.seed)
    low, high = metrics["bler_ci95_wilson"]
    sufficient = metrics["samples"] >= 48 and high - low <= 0.28
    return (
        "PASS" if sufficient else "INCONCLUSIVE",
        None if sufficient else "Sample count below 48 or payload BLER Wilson interval wider than 0.28",
        {"oracle_type": "STATISTICAL", "minimum_samples": 48, "max_wilson_width": 0.28},
        metrics | {"frames": frames, "snr_db": case.snr_db, "crc_failures": sum(crc_counts),
                   "payload_errors": metrics["tb_errors"], "crc_bler": crc_metrics["bler"],
                   "crc_bler_ci95_wilson": crc_metrics["bler_ci95_wilson"],
                   "link_duration_seconds": round(time.perf_counter() - start, 6),
                   "snr_definition": "Average TX data-RE power divided by complex noise variance per RX branch; unity gain",
                   "rng": "NumPy default_rng(SeedSequence([seed, frame_index])); first TB bytes, then real/imag normals"},
        arrays,
    )


def contract_check(case) -> tuple[str, str | None, dict, dict, dict]:
    """Exercise the same front-door checks used by ordinary cases."""
    if case.test_type == "config_contract":
        illegal_tb_bits = case.tb_size + 8
        rejected, reason = False, None
        try:
            validate_case(case.model_copy(update={"tb_size": illegal_tb_bits}))
        except ValueError as exc:
            reason = str(exc)
            rejected = reason.startswith("TBS must equal")
        actual = {"valid_tbs_bits": case.tb_size, "bad_tbs_bits": illegal_tb_bits}
    else:
        from nr_ai_qa.adapters.pusch_adapter import (
            ReceiverContractError, SLOT_SYMBOLS, SUBCARRIERS, validate_rx_iq,
        )

        valid_shape = (SUBCARRIERS, SLOT_SYMBOLS, case.num_rx_ant)
        validate_rx_iq(np.empty(valid_shape, dtype=np.complex64), case.num_rx_ant)
        malformed = np.empty((SUBCARRIERS, SLOT_SYMBOLS - 1, case.num_rx_ant), dtype=np.complex64)
        rejected, reason = False, None
        try:
            validate_rx_iq(malformed, case.num_rx_ant)
        except ReceiverContractError as exc:
            reason, rejected = str(exc), True
        actual = {"valid_shape": list(valid_shape), "bad_shape": list(malformed.shape)}
    return (
        "PASS" if rejected else "FAIL", None if rejected else "Validation accepted malformed input",
        {"oracle_type": "PRE_GPU_REJECTION", "invalid_input_status": "CONFIGURATION_ERROR"},
        actual | {"invalid_input_status": "CONFIGURATION_ERROR" if rejected else "ACCEPTED",
                  "validation_error": reason}, {},
    )


def llr_difference(clean: np.ndarray, changed: np.ndarray) -> dict:
    if clean.shape != changed.shape:
        return {"shapes": [list(clean.shape), list(changed.shape)], "comparable": False}
    a, b = clean.astype(np.float32), changed.astype(np.float32)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return {"comparable": False, "reason": "Nonfinite LLR values"}
    different = np.flatnonzero(~np.isclose(a, b, rtol=1e-3, atol=1e-3))
    first = None
    if different.size:
        offset = int(different[0])
        first = {"flat_index": offset, "clean": float(a.flat[offset]), "injected": float(b.flat[offset])}
    return {"comparable": True, "fraction_changed": float(different.size / a.size),
            "max_abs_difference": float(np.max(np.abs(a - b))), "first_difference": first}


def paired_fault(adapter, case) -> tuple[str, str | None, dict, dict, dict]:
    tx_cfg, rx_cfg = adapter.configs(case)
    noise_bias = case.test_type == "noise_fault"
    nominal_id = case.dmrs_config.scrambling_id
    mismatch = 777 if nominal_id != 777 else 778
    fault_cfg = rx_cfg if noise_bias else adapter.configs(case, rx_dmrs_id=mismatch)[1]
    path = "separable" if noise_bias else "fused"
    offset_db = 36.0 if noise_bias else None
    input_clean = {"dmrs_scrambling_id": nominal_id, "equalizer_noise_offset_db": 0.0}
    input_fault = {"dmrs_scrambling_id": nominal_id if noise_bias else mismatch,
                   "equalizer_noise_offset_db": offset_db if noise_bias else 0.0}
    frames, arrays = [], {}
    saved_detection = False
    for index in range(case.num_transport_blocks):
        frame = adapter.prepare(case, index, tx_cfg)
        iq = frame.iq.get()
        clean = adapter.receive(frame, rx_cfg, path=path, capture=noise_bias)
        injected = adapter.receive(frame, fault_cfg, path=path, capture=noise_bias,
                                   noise_offset_db=offset_db)
        nominal = inspect_reception(frame.payload, clean)
        actual = inspect_reception(frame.payload, injected)
        clean_ok = nominal["valid_output"] and nominal["crc"] == [0] and nominal["bit_errors"] == 0
        valid = clean_ok and actual["valid_output"]
        payload_delta = (first_difference(np.unpackbits(clean.payloads[0]),
                                          np.unpackbits(injected.payloads[0])) if valid else None)
        crc_changed = bool(valid and not np.array_equal(clean.crc, injected.crc))
        frame_result = {"index": index, "slot": frame.slot, "iq_sha256": hash_array(iq),
                        "clean": nominal, "injected": actual, "payload_difference": payload_delta,
                        "crc_changed": crc_changed}
        if noise_bias:
            repeated = adapter.receive(frame, rx_cfg, path=path, capture=True)
            llr = llr_difference(clean.intermediate["llrs"], injected.intermediate["llrs"])
            baseline = llr_difference(clean.intermediate["llrs"], repeated.intermediate["llrs"])
            clean_db = clean.intermediate["applied_noise_db"]
            estimation = injected.intermediate["estimated_noise_db"]
            applied = injected.intermediate["applied_noise_db"]
            measured_delta = applied - clean_db
            matched_delta = bool(np.allclose(measured_delta, offset_db, rtol=0, atol=1e-4))
            # Compare against the numerical variation of this exact clean IQ.
            # The first calibration run showed sparse FP16 differences on repeats.
            measured_change = (
                llr["comparable"] and baseline["comparable"] and
                baseline["fraction_changed"] <= 0.01 and llr["fraction_changed"] >= 0.10 and
                llr["fraction_changed"] >= 10 * baseline["fraction_changed"]
            )
            detected = bool(valid and matched_delta and measured_change)
            frame_result.update(llr_difference=llr, clean_repeat_llr_difference=baseline,
                                estimated_noise_db=estimation.tolist(), applied_noise_db=applied.tolist(),
                                clean_noise_db=clean_db.tolist(), measured_noise_delta_db=measured_delta.tolist())
        else:
            detected = bool(valid and (payload_delta is not None or crc_changed))
        frames.append(frame_result | {"clean_pass": clean_ok, "detected": detected})
        if index == 0 or (detected and not saved_detection):
            arrays = {
                "iq": iq, "tx_payload": frame.payload, "clean_crc": clean.crc,
                "injected_crc": injected.crc,
                **{f"clean_{key}": value for key, value in clean.intermediate.items()},
                **{f"injected_{key}": value for key, value in injected.intermediate.items()},
            }
            if clean.payloads and injected.payloads:
                arrays.update(clean_payload=clean.payloads[0], injected_payload=injected.payloads[0])
            saved_index, saved_detection = index, detected
    hits = sum(f["detected"] for f in frames)
    clean_pass = sum(f["clean_pass"] for f in frames)
    invalid = any(not f["clean"]["valid_output"] or not f["injected"]["valid_output"] for f in frames)
    enough = hits >= 3 and hits * 4 >= len(frames) * 3 and clean_pass == len(frames)
    payload_errors = {
        variant: sum((f[variant]["bit_errors"] or 0) > 0 for f in frames)
        for variant in ("clean", "injected")
    }
    crc_failures = {
        variant: sum(f[variant]["crc"] != [0] for f in frames) for variant in ("clean", "injected")
    }
    return (
        "FAIL" if invalid else "EXPECTED_DETECTION" if enough else "INCONCLUSIVE",
        "Malformed receiver output" if invalid else
        None if enough else "Fewer than three repeated detections, below 75% of pairs, or clean baseline failed",
        {"oracle_type": "PAIRED_OUTPUT_DIFFERENCE",
         "output_measure": "active LLR" if noise_bias else "TB bits or CRC",
         "required_detections": "At least three and at least 75% of pairs with all clean pairs recovering",
         "llr_change_rule": ("rtol=1e-3, atol=1e-3; clean repeat ≤1%, injected ≥10%, and injected "
                             "fraction ≥10× clean repeat fraction") if noise_bias else None},
        {"frames": frames, "receiver": path,
         "injection_location": "ChannelEqualizer.equalize noise_var_pre_eq" if noise_bias else "PuschConfig.dmrs_scrm_id",
         "clean_input": input_clean, "injected_input": input_fault,
         "samples": len(frames), "detections": hits, "clean_passes": clean_pass,
         "payload_errors": payload_errors, "crc_failures": crc_failures,
         "first_detected_frame": next((f["index"] for f in frames if f["detected"]), None),
         "saved_frame": saved_index,
         "bler_inference": "INCONCLUSIVE: finite paired diagnostic; BLER effect is outside the oracle"},
        arrays,
    )


def execute_pusch(runner, raw: dict | PuschCase) -> Result:
    start = time.perf_counter()
    parameters = raw.model_dump(mode="json") if isinstance(raw, PuschCase) else raw
    cid = parameters.get("case_id")
    case, phase = None, "validation"
    context, expected, actual, arrays = {}, {}, {}, {}
    error_class = None
    try:
        case = validate_case(raw)
        parameters = case.model_dump(mode="json")
        if case.test_type in {"config_contract", "iq_contract"}:
            phase = "oracle"
            status, reason, expected, actual, arrays = contract_check(case)
            context = runner.metadata or {}
        else:
            phase = "environment"
            runner.identify_environment()
            context = runner.metadata
            phase = "sdk"
            if runner.pusch_adapter is None:
                from nr_ai_qa.adapters.pusch_adapter import PyAerialPuschAdapter

                runner.pusch_adapter = PyAerialPuschAdapter()
            test = {"golden": golden, "consistency": consistent, "awgn": awgn,
                    "dmrs_fault": paired_fault, "noise_fault": paired_fault}[case.test_type]
            status, reason, expected, actual, arrays = test(runner.pusch_adapter, case)
    except (UnsupportedPuschConfiguration, NotImplementedError) as exc:
        status, reason, error_class = "UNSUPPORTED", str(exc), "UNSUPPORTED_CONFIGURATION"
    except (ValidationError, ValueError) as exc:
        if phase == "validation":
            status, error_class = "CONFIGURATION_ERROR", "TEST_CASE_ERROR"
        else:
            error_class = {"environment": "ENVIRONMENT_ERROR", "sdk": "SDK_ERROR",
                           "oracle": "ORACLE_ERROR"}[phase]
            status = "FAIL"
            actual["traceback"] = traceback.format_exc()
        reason = str(exc)
    except Exception as exc:
        status, reason = "FAIL", str(exc)
        error_class = {"environment": "ENVIRONMENT_ERROR", "sdk": "SDK_ERROR",
                       "oracle": "ORACLE_ERROR"}.get(phase, "TEST_CASE_ERROR")
        actual["traceback"] = traceback.format_exc()

    # Never use a rejected, untrusted case identifier as a filesystem path.
    if case is None:
        from pydantic import TypeAdapter
        from nr_ai_qa.schemas.pusch_case import Identifier

        try:
            cid = TypeAdapter(Identifier).validate_python(cid)
        except ValidationError:
            cid = "INVALID-PUSCH-CASE"
        parameters = {"kind": "pusch", "case_id": cid}
    actual["environment"] = context
    actual["numpy_version"] = np.__version__
    if case:
        q, rate = mcs_parameters(case.mcs)
        actual["allocation"] = {
            "tbs_bits": case.tb_size, "allocated_data_re": 132 * case.num_prbs,
            "Qm": q, "target_code_rate": rate, "tx_antennas": 1, "rx_antennas": case.num_rx_ant,
            "rnti": 1234, "cell_id": 41, "rv": 0, "numerology": 1,
        }
    actual["gpu_execution_started"] = phase == "sdk"
    actual["pyaerial_api"] = (
        ["PdschTxPipelineFactory", "PuschRxPipelineFactory"] if phase == "sdk" else []
    )
    if phase == "sdk" and case.test_type in {"consistency", "noise_fault"}:
        if case.test_type == "noise_fault":
            actual["pyaerial_api"].remove("PuschRxPipelineFactory")
        actual["pyaerial_api"].append("SeparablePuschRxPipelineFactory")
    result = Result(cid, case.requirement_ids if case else [], context.get("commit", "not-executed"),
                    context.get("GPU", "not-executed"), context.get("CUDA_version", "not-executed"),
                    parameters, case.seed if case else None, expected, actual, status, reason,
                    round(time.perf_counter() - start, 6), error_class=error_class)
    if runner.evidence_root is not None:
        if case:
            source = Path("evidence") / f"{cid}.case.json"
            source_path = runner.evidence_root / source
            source_path.parent.mkdir(parents=True, exist_ok=True)
            source_path.write_text(json_text(parameters) + "\n", encoding="utf-8")
            actual["input_case"] = source.as_posix()
            actual["reproduction_command"] = shlex.join([
                "scripts/run_qa_container.sh", "-m", "nr_ai_qa.cli", "--json", str(source_path),
                "--out", str(runner.evidence_root / "reproduced"),
            ])
        if arrays:
            path = Path("evidence") / f"{cid}.npz"
            (runner.evidence_root / path).parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(runner.evidence_root / path, **arrays)
            actual["array_evidence"] = path.as_posix()
        save_evidence(runner.evidence_root, result, {})
    runner.results[cid] = result
    return result
