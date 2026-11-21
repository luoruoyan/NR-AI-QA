import numpy as np
import pytest

from nr_ai_qa.qa.oracle import Geometry, matched_indices, matched_reference


def test_independent_circular_selection_sanity():
    # Small artificial index sequence verifies wrap, NULL exclusion, Qm=2
    # interleave and concatenation independently of the GPU encoder.
    geo = Geometry(bg=2, tb_crc=16, cb_crc=24, count=2, z=1, k_prime=4, k=5, n=6)
    indices = matched_indices(geo, g=14, q=2, rv=0)
    assert [x.tolist() for x in indices] == [[0, 4, 1, 5, 3, 0], [0, 5, 1, 0, 3, 1, 4, 3]]
    shaped = np.stack([np.arange(6), np.arange(6) + 10], axis=1)
    assert matched_reference(shaped, indices).tolist() == [0, 4, 1, 5, 3, 0, 10, 15, 11, 10, 13, 11, 14, 13]


@pytest.mark.parametrize("bg,start_factors", [(1, (0, 17, 33, 56)), (2, (0, 13, 25, 43))])
@pytest.mark.parametrize("rv", range(4))
def test_gpu_rate_matching_every_rv(suite, bg, start_factors, rv):
    runner, profiles = suite
    result = runner.execute_case(profiles[f"RV-BG{bg}-{rv}"])
    assert result.status == "PASS", (result.failure_reason, result.actual_result)
    expected, actual = result.expected_result, result.actual_result
    assert actual["native_bg"] == bg
    assert expected["k0"] == expected["geometry"]["z"] * start_factors[rv]
    assert sum(actual["E_r"]) == expected["G"]
    assert actual["checks"]["rate_matching_bits"]
    assert actual["tb_crc"] == 0 and actual["bit_errors"] == 0


def test_real_circular_repetition(suite):
    runner, profiles = suite
    result = runner.execute_case(profiles["RV-BG2-REPEAT"])
    assert result.status == "PASS", (result.failure_reason, result.actual_result)
    active_cb = result.expected_result["geometry"]["n"] - result.expected_result["filler_per_cb"]
    assert min(result.actual_result["E_r"]) > active_cb
    assert result.actual_result["checks"]["rate_matching_bits"]
