"""Validate inputs, execute cuPHY and evaluate trusted oracles."""

from dataclasses import asdict, dataclass
from hashlib import sha256
import os
from pathlib import Path
import subprocess
import time
import traceback

import numpy as np
from pydantic import ValidationError
import yaml

from nr_ai_qa.qa.oracle import (
    first_difference, geometry, intervals, matched_indices, matched_reference, RV_FACTORS,
    modulation_reference, segmented_reference,
)
from nr_ai_qa.schemas.ldpc_case import LdpcCase, UnsupportedConfiguration, validate_profile_budget
from nr_ai_qa.schemas.pusch_case import PuschCase

ROOT = Path(__file__).resolve().parents[3]
PINNED_COMMIT = "4f65f97c1d5f701ce911f7dda8f1b1f3f0c7693c"


def read_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def load_profiles() -> dict[str, dict]:
    cases = read_yaml(ROOT / "configs/ldpc_profiles.yaml")["cases"]
    profiles = {case["case_id"]: case for case in cases}
    if len(profiles) != len(cases):
        raise ValueError("Duplicate profile case ID")
    return profiles


def validate_case(raw: dict | LdpcCase | PuschCase) -> LdpcCase | PuschCase:
    if isinstance(raw, PuschCase) or isinstance(raw, dict) and raw.get("kind") == "pusch":
        from nr_ai_qa.qa.pusch_oracle import validate_case as validate_pusch_case

        return validate_pusch_case(raw)
    case = LdpcCase.model_validate(raw.model_dump() if isinstance(raw, LdpcCase) else raw)
    records = read_yaml(ROOT / "specs/ldpc_requirements.yaml")["requirements"]
    active = {r["requirement_id"] for r in records if r["coverage"] == "implemented"}
    if set(case.requirement_ids) - active:
        raise ValueError("Unknown or pending requirement ID")
    return validate_profile_budget(case)


@dataclass
class Result:
    case_id: str
    requirement_ids: list[str]
    SDK_commit: str
    GPU: str
    CUDA_version: str
    test_parameters: dict
    seed: int | None
    expected_result: dict
    actual_result: dict
    status: str
    failure_reason: str | None
    duration: float
    evidence_path: str | None = None
    error_class: str | None = None


def hash_array(data: np.ndarray) -> str:
    return sha256(data.tobytes()).hexdigest()


def inspect_frame(case: LdpcCase, frame) -> tuple[dict, dict, dict]:
    geo = geometry(case.tb_size, case.code_rate)
    observed_candidates = []
    for bg in (1, 2):
        try:
            candidate = geometry(case.tb_size, case.code_rate, bg=bg)
        except (ValueError, StopIteration):
            continue
        if frame.encoded.shape == (candidate.n, candidate.count):
            observed_candidates.append(bg)
    expected = {"geometry": asdict(geo), "filler_per_cb": geo.filler, "native_bg": geo.bg}
    seg_reference = segmented_reference(frame.payload, geo)
    seg_difference = first_difference(seg_reference, frame.blocks[:geo.k_prime])
    systematic_reference = seg_reference[2 * geo.z:]
    systematic = frame.encoded[:len(systematic_reference)]
    diffs = {"segmentation": seg_difference,
             "systematic": first_difference(systematic_reference, systematic)}
    checks = {"crc_and_segmentation": seg_difference is None and frame.blocks.shape == (geo.k, geo.count),
              "sdk_filler_zero": bool(np.all(frame.blocks[geo.k_prime:] == 0)),
              "native_geometry": frame.encoded.shape == (geo.n, geo.count),
              "systematic_bits": diffs["systematic"] is None,
              "encoded_binary": bool(np.isin(frame.encoded, (0, 1)).all())}
    actual = {
        "cb_shape": list(frame.blocks.shape), "encoded_shape": list(frame.encoded.shape),
        "native_bg": observed_candidates[0] if len(observed_candidates) == 1 else None,
        "native_bg_candidates": observed_candidates, "sdk_helper_bg": frame.sdk_helper_bg,
        "checks": checks, "differences": diffs, "payload_sha256": hash_array(frame.payload),
        "encoded_sha256": hash_array(frame.encoded),
    }
    arrays = {"payload_bytes": frame.payload, "segmented": frame.blocks,
              "reference_segments": seg_reference, "encoded": frame.encoded}
    if frame.matched is None:
        return expected, actual, arrays

    indices = matched_indices(geo, case.g, case.mod_order, case.rv)
    reference = matched_reference(frame.encoded, indices) if checks["native_geometry"] else None
    dims = [len(positions) for positions in indices]
    expected.update(G=case.g, E_r=dims, Qm=case.mod_order, tb_crc=0, bit_errors=0,
                    k0=RV_FACTORS[geo.bg][case.rv] * geo.z,
                    rx_bytes=case.tb_size // 8, de_rate_matched_shape=[geo.n + 2 * geo.z, geo.count])
    if case.test_type == "awgn":
        expected.pop("tb_crc")
        expected.pop("bit_errors")
    if reference is not None:
        diffs["rate_matching"] = first_difference(reference, frame.matched)
        arrays["reference_matched"] = reference
    rx_bits = np.unpackbits(frame.received, bitorder="big")
    tx_bits = np.unpackbits(frame.payload, bitorder="big")
    diffs["payload"] = first_difference(tx_bits, rx_bits)
    bit_errors = int(np.count_nonzero(tx_bits != rx_bits)) if tx_bits.shape == rx_bits.shape else None
    checks.update(
        rate_matching_length=frame.matched.shape == (case.g,) and list(frame.rm_lengths) == dims,
        rate_matching_bits=reference is not None and diffs["rate_matching"] is None,
        modulation=bool(np.allclose(frame.symbols, modulation_reference(frame.matched, case.mod_order),
                                    atol=2e-7, rtol=0)),
        finite_llrs=frame.finite_llrs,
        de_rate_matching_shape=frame.derated_shape == (geo.n + 2 * geo.z, geo.count),
        decoded_shape=frame.decoded.shape == (geo.k, geo.count),
        received_length=frame.received.shape == frame.payload.shape,
    )
    actual.update(G=len(frame.matched), E_r=list(frame.rm_lengths), rx_bytes=len(frame.received),
                  tb_crc=frame.tb_crc, cb_crc=list(frame.cb_crc),
                  bit_errors=bit_errors, decoded_shape=list(frame.decoded.shape),
                  de_rate_matched_shape=list(frame.derated_shape), llr_shape=list(frame.llr_shape),
                  matched_sha256=hash_array(frame.matched), received_sha256=hash_array(frame.received))
    if case.test_type != "awgn":
        checks.update(payload_recovered=bit_errors == 0, tb_crc_pass=frame.tb_crc == 0)
        if geo.count > 1:
            checks["cb_crc_pass"] = len(frame.cb_crc) == geo.count and not any(frame.cb_crc)
    arrays.update(matched=frame.matched, matched_before_fault=frame.matched_before_fault,
                  received_bytes=frame.received, decoded=frame.decoded)
    return expected, actual, arrays


class QaRunner:
    def __init__(self, evidence_root: Path | None = None):
        self.evidence_root = evidence_root
        self.results = {}
        self.adapter = None
        self.pusch_adapter = None
        self.metadata = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        try:
            if self.pusch_adapter is not None:
                self.pusch_adapter.close()
        finally:
            self.pusch_adapter = None
            if self.adapter is not None:
                try:
                    self.adapter.close()
                finally:
                    self.adapter = None

    def identify_environment(self):
        if self.metadata is not None:
            return
        import ctypes
        from importlib.metadata import version
        import cupy as cp
        from aerial.pycuphy import _pycuphy

        sdk = Path("/opt/nvidia/cuBB")
        commit = subprocess.check_output(["git", "-C", sdk, "rev-parse", "HEAD"], text=True).strip()
        ext = Path(_pycuphy.__file__).resolve()
        if commit != PINNED_COMMIT or not ext.is_relative_to(sdk / "pyaerial/src/aerial/pycuphy"):
            raise RuntimeError("Unexpected SDK provenance")
        gpu = cp.cuda.runtime.getDeviceProperties(0)
        # The Python wheel and the native extension can link different
        # runtimes. Record both and use the actual native DSO for CUDA_version.
        libcudart = next(path for path in Path("/proc/self/maps").read_text().splitlines()
                         if "/usr/local/cuda-13.3/" in path and "/libcudart.so." in path).split()[-1]
        native_cuda = ctypes.c_int()
        rc = ctypes.CDLL(libcudart).cudaRuntimeGetVersion(ctypes.byref(native_cuda))
        if rc:
            raise RuntimeError(f"cudaRuntimeGetVersion failed with status {rc}")
        raw_cuda = native_cuda.value
        self.metadata = {"commit": commit, "GPU": gpu["name"].decode(),
                         "compute_capability": f'{gpu["major"]}.{gpu["minor"]}',
                         "gpu_total_bytes": int(gpu["totalGlobalMem"]),
                         "CUDA_version": f"{raw_cuda // 1000}.{raw_cuda % 1000 // 10}",
                         "driver_api": cp.cuda.runtime.driverGetVersion(),
                         "cupy_runtime": cp.cuda.runtime.runtimeGetVersion(),
                         "native_runtime_path": libcudart, "pyaerial": version("pyaerial"),
                         "extension_path": str(ext), "image_digest": os.getenv("NRQA_IMAGE_DIGEST")}
    def setup(self):
        if self.adapter is None:
            from nr_ai_qa.adapters.ldpc_adapter import PyAerialLdpcAdapter

            self.identify_environment()
            self.adapter = PyAerialLdpcAdapter()

    def execute_case(self, raw: dict | LdpcCase | PuschCase) -> Result:
        if isinstance(raw, PuschCase) or isinstance(raw, dict) and raw.get("kind") == "pusch":
            from nr_ai_qa.qa.pusch_oracle import execute_pusch

            return execute_pusch(self, raw)
        from nr_ai_qa.qa.reporting import save_evidence

        start = time.perf_counter()
        parameters = raw.model_dump(mode="json") if isinstance(raw, LdpcCase) else raw
        case = None
        expected, actual, arrays = {}, {}, {}
        phase = "validation"
        gpu_execution_started = False
        error_class = None
        try:
            case = validate_case(raw)
            parameters = case.model_dump(mode="json")
            phase = "environment"
            self.setup()
            rng = np.random.default_rng(case.seed)
            phase = "sdk"
            gpu_execution_started = True
            frame = self.adapter.execute(case, rng)
            phase = "oracle"
            expected, actual, arrays = inspect_frame(case, frame)
            status, reason = "PASS", None
            if case.test_type == "bg_probe":
                expected["sdk_helper_bg"] = expected["native_bg"]
                actual["checks"]["helper_bg_agrees"] = actual["sdk_helper_bg"] == expected["native_bg"]
            if case.test_type == "awgn":
                errors = [actual["bit_errors"]]
                crc_failures = int(frame.tb_crc != 0)
                crc_disagreements = int((frame.tb_crc != 0) != (actual["bit_errors"] != 0))
                failed = None if all(actual["checks"].values()) else {"sample": 0, "actual": actual.copy()}
                for index in range(1, case.samples):
                    if failed is not None:
                        break
                    phase = "sdk"
                    next_frame = self.adapter.execute(case, rng)
                    phase = "oracle"
                    _, observation, sample_arrays = inspect_frame(case, next_frame)
                    errors.append(observation["bit_errors"])
                    crc_failures += int(next_frame.tb_crc != 0)
                    crc_disagreements += int((next_frame.tb_crc != 0) != (observation["bit_errors"] != 0))
                    if not all(observation["checks"].values()):
                        failed = {"sample": index, "actual": observation}
                        arrays = sample_arrays
                if failed is not None:
                    actual["first_invalid_sample"] = failed
                    status, reason = "POTENTIAL_SDK_DEFECT", "AWGN sample violated a deterministic invariant"
                else:
                    stats = intervals(errors, case.tb_size, case.seed ^ 0x38_212)
                    actual.update(stats, esno_db=case.esno_db, crc_failures=crc_failures,
                                  crc_vs_payload_disagreements=crc_disagreements,
                                  sample_bit_errors=errors)
                    expected.update(min_samples=64, confidence=0.95,
                                    noise_model="Es=1, E[|n|²]=10^(-EsN0_dB/10)")
                    if case.samples < 64 or np.ptp(stats["bler_ci95_wilson"]) > 0.22:
                        status, reason = "INCONCLUSIVE", "Insufficient TB samples or BLER precision"
            elif case.test_type == "fault":
                phase = "sdk"
                clean = self.adapter.execute(case, np.random.default_rng(case.seed), inject=False)
                phase = "oracle"
                _, baseline, _ = inspect_frame(case, clean)
                actual["clean_control_checks"] = baseline["checks"]
                observed_failures = {key for key, passed in actual["checks"].items() if not passed}
                if case.fault == "rm_bit_flip":
                    difference = actual["differences"].get("rate_matching")
                    detected = difference is not None and difference.get("count") == 1
                    permitted = {"rate_matching_bits"}
                    # The decoder may correct the flipped bit or expose downstream loss.
                    permitted |= {"payload_recovered", "tb_crc_pass", "cb_crc_pass"}
                else:
                    difference = actual["differences"]["payload"]
                    detected = (difference is not None and difference.get("count") == 1
                                and actual["tb_crc"] != 0)
                    permitted = {"payload_recovered", "tb_crc_pass"}
                actual["fault"] = {**frame.fault, "first_difference": difference,
                                   "replay_case_id": case.case_id, "seed": case.seed}
                expected["fault_policy"] = "clean control passes and data-derived detector trips"
                if detected and all(baseline["checks"].values()) and not observed_failures - permitted:
                    status = "EXPECTED_DETECTION"
                else:
                    status, reason = "POTENTIAL_SDK_DEFECT", "Clean control or fault detection did not qualify"
            elif not all(actual["checks"].values()):
                status = "POTENTIAL_SDK_DEFECT"
                reason = ", ".join(name for name, value in actual["checks"].items() if not value)
            if status == "POTENTIAL_SDK_DEFECT":
                error_class = "SDK_ERROR"
        except (ValidationError, ValueError) as exc:
            if phase == "validation":
                error_class = ("UNSUPPORTED_CONFIGURATION" if isinstance(exc, UnsupportedConfiguration)
                               else "TEST_CASE_ERROR")
                status = "UNSUPPORTED_CONFIGURATION" if error_class.startswith("UNSUPPORTED") else "CONFIGURATION_ERROR"
            else:
                error_class = {"environment": "ENVIRONMENT_ERROR", "sdk": "SDK_ERROR",
                               "oracle": "ORACLE_ERROR"}[phase]
                status = "ERROR"
            reason = str(exc)
            actual["traceback"] = traceback.format_exc()
        except Exception as exc:
            error_class = {"environment": "ENVIRONMENT_ERROR", "sdk": "SDK_ERROR",
                           "oracle": "ORACLE_ERROR", "validation": "TEST_CASE_ERROR"}[phase]
            status, reason = "ERROR", str(exc)
            actual["traceback"] = traceback.format_exc()

        context = self.metadata or {}
        safe_parameters = parameters if isinstance(parameters, dict) else {"input": parameters}
        cid = case.case_id if case else "INVALID-INPUT"
        actual["environment"] = context
        actual["gpu_execution_started"] = gpu_execution_started
        result = Result(cid, case.requirement_ids if case else [],
                        context.get("commit", "not-executed"), context.get("GPU", "not-executed"),
                        context.get("CUDA_version", "not-executed"), safe_parameters,
                        case.seed if case else None, expected, actual, status, reason,
                        round(time.perf_counter() - start, 6), error_class=error_class)
        if self.evidence_root is not None:
            save_evidence(self.evidence_root, result, arrays)
        self.results[cid] = result
        return result

    def collect_result(self, raw: dict | LdpcCase | PuschCase) -> Result:
        case = validate_case(raw)
        result = self.results[case.case_id]
        if result.test_parameters != case.model_dump(mode="json"):
            raise KeyError("No result for the supplied parameters")
        return result
