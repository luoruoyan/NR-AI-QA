import pytest


@pytest.mark.parametrize("case_id,location", [
    ("FAULT-RM-FLIP", "after_rate_matching"),
    ("FAULT-CRC-FLIP", "after_ldpc_decode_before_crc"),
])
def test_fault_detected_consistently_by_data(suite, case_id, location):
    runner, profiles = suite
    first = runner.execute_case(profiles[case_id])
    second = runner.execute_case(profiles[case_id])
    assert first.status == second.status == "EXPECTED_DETECTION", (first.failure_reason, first.actual_result)
    observed = first.actual_result
    assert all(observed["clean_control_checks"].values())
    assert observed["fault"] == second.actual_result["fault"]
    fault = observed["fault"]
    assert fault["location"] == location
    assert fault["before"] != fault["after"]
    assert fault["first_difference"]["flat_index"] == fault["offset"] == 7
    assert fault["first_difference"]["count"] == 1
    if case_id == "FAULT-RM-FLIP":
        assert not observed["checks"]["rate_matching_bits"]
    else:
        assert observed["tb_crc"] != 0
        assert observed["bit_errors"] == 1


def test_invalid_rv_is_classified_without_calling_gpu(suite):
    runner, profiles = suite
    result = runner.execute_case(profiles["FAULT-RM-FLIP"] | {"rv": 4})
    assert result.status == "CONFIGURATION_ERROR"
    assert result.error_class == "TEST_CASE_ERROR"
    assert "rv" in result.failure_reason
