import pytest


def test_awgn_smoke_is_inconclusive(suite):
    runner, profiles = suite
    result = runner.execute_case(profiles["AWGN-SMOKE"])
    assert result.status == "INCONCLUSIVE", (result.failure_reason, result.actual_result)
    assert result.actual_result["samples"] == 8


@pytest.mark.parametrize("point", ["LOW", "MID", "HIGH"])
def test_awgn_statistics(suite, point):
    runner, profiles = suite
    result = runner.execute_case(profiles[f"AWGN-{point}"])
    assert result.status == "PASS", (result.failure_reason, result.actual_result)
    stats = result.actual_result
    assert stats["samples"] == 96
    assert stats["tb_errors"] == sum(x > 0 for x in stats["sample_bit_errors"])
    assert stats["bit_errors"] == sum(stats["sample_bit_errors"])
    assert stats["bler"] == stats["tb_errors"] / 96
    assert stats["ber"] == stats["bit_errors"] / stats["bits_tested"]
    assert stats["bler_ci95_wilson"][0] <= stats["bler"] + 1e-15
    assert stats["bler"] <= stats["bler_ci95_wilson"][1] + 1e-15
    assert len(stats["ber_ci95_tb_bootstrap"]) == 2


def test_awgn_reproducible_point(suite):
    runner, profiles = suite
    case = profiles["AWGN-MID"]
    first = runner.execute_case(case)
    second = runner.execute_case(case)
    assert first.status == second.status == "PASS"
    assert first.actual_result["sample_bit_errors"] == second.actual_result["sample_bit_errors"]
    assert first.actual_result["bler_ci95_wilson"] == second.actual_result["bler_ci95_wilson"]
    assert first.actual_result["ber_ci95_tb_bootstrap"] == second.actual_result["ber_ci95_tb_bootstrap"]
