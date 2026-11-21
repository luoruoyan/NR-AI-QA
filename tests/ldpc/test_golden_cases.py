import numpy as np

from nr_ai_qa.qa.oracle import crc_bits
from nr_ai_qa.qa.runner import QaRunner, load_profiles


def test_independent_crc16_check_value():
    bits = np.unpackbits(np.frombuffer(b"123456789", dtype=np.uint8))
    assert np.packbits(crc_bits(bits, 16, 0x1021)).tobytes().hex() == "31c3"


def test_golden_qpsk():
    with QaRunner() as runner:
        result = runner.execute_case(load_profiles()["GOLD-QPSK"])
    assert result.status == "PASS", result.failure_reason
    assert result.actual_result["bit_errors"] == 0
    assert result.actual_result["tb_crc"] == 0
    assert result.actual_result["checks"]["crc_and_segmentation"]
    assert result.actual_result["checks"]["rate_matching_bits"]


def test_golden_bg1(suite):
    runner, profiles = suite
    result = runner.execute_case(profiles["GOLD-BG1"])
    assert result.status == "PASS", result.failure_reason
    assert result.actual_result["native_bg"] == 1
    assert result.expected_result["Qm"] == 4
    assert result.actual_result["checks"]["modulation"]
