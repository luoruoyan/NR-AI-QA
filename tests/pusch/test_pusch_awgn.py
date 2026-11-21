import pytest

from nr_ai_qa.qa.pusch_oracle import load_pusch_profiles
from nr_ai_qa.qa.runner import QaRunner


@pytest.mark.parametrize("case_id", ["P03-SMOKE-MID", "P03-MID"])
def test_p03_gpu_awgn_statistics(case_id):
    case = load_pusch_profiles()[case_id]
    with QaRunner() as runner:
        result = runner.execute_case(case)
    expected_status = "INCONCLUSIVE" if case_id == "P03-SMOKE-MID" else "PASS"
    assert result.status == expected_status, result.failure_reason
    actual = result.actual_result
    assert actual["samples"] == case["num_transport_blocks"]
    assert actual["payload_errors"] == sum(f["bit_errors"] > 0 for f in actual["frames"])
    assert actual["crc_failures"] == sum(f["crc"] != [0] for f in actual["frames"])
    assert actual["bler_ci95_wilson"][0] <= actual["bler"] <= actual["bler_ci95_wilson"][1]
    for frame in actual["frames"]:
        assert frame["data_re_power"] / frame["noise_variance_per_branch"] == pytest.approx(
            10**(case["snr_db"] / 10)
        )
