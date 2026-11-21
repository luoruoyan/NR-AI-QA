import pytest

from nr_ai_qa.qa.oracle import base_graph
from nr_ai_qa.qa.runner import validate_case


@pytest.mark.parametrize("a,rate,want", [
    (292, 0.8, 2), (293, 0.8, 1), (3824, 0.67, 2), (3825, 0.67, 1),
    (3840, 0.25, 2), (3840, 0.25001, 1), (3824, 0.67001, 1),
])
def test_clause_exact_boundaries(a, rate, want):
    assert base_graph(a, rate) == want


@pytest.mark.parametrize("case_id,bg", [
    ("BG-A288", 2), ("BG-A304", 1), ("BG-A3824", 2), ("BG-A3840", 1),
    ("BG-RLOW", 2), ("BG-RHIGH", 1), ("BG-R25", 2), ("BG-R067-ABOVE", 1),
])
def test_native_bg_boundaries(suite, case_id, bg):
    runner, profiles = suite
    result = runner.execute_case(profiles[case_id])
    assert result.status == "PASS", result.failure_reason
    assert result.actual_result["native_bg"] == result.expected_result["native_bg"] == bg


def test_native_and_python_bg_disagree_at_release17_threshold(suite):
    runner, profiles = suite
    result = runner.execute_case(profiles["BG-R067"])
    assert result.actual_result["native_bg"] == result.expected_result["native_bg"] == 2
    assert result.actual_result["sdk_helper_bg"] == 1
    assert result.status == "POTENTIAL_SDK_DEFECT"
    assert {name for name, ok in result.actual_result["checks"].items() if not ok} == {"helper_bg_agrees"}


@pytest.mark.parametrize("case_id,count,tb_crc,cb_crc", [
    ("SEG-BG2-ONE", 1, 16, 0), ("SEG-BG2-TWO", 2, 24, 24),
    ("SEG-BG1-ONE", 1, 24, 0), ("SEG-BG1-TWO", 2, 24, 24),
    ("SEG-BG2-FOUR", 4, 24, 24),
])
def test_segmentation_and_filler(suite, case_id, count, tb_crc, cb_crc):
    runner, profiles = suite
    result = runner.execute_case(profiles[case_id])
    assert result.status == "PASS", (result.failure_reason, result.actual_result)
    geo = result.expected_result["geometry"]
    assert (geo["count"], geo["tb_crc"], geo["cb_crc"]) == (count, tb_crc, cb_crc)
    assert result.actual_result["cb_shape"] == [geo["k"], count]
    assert result.actual_result["checks"]["sdk_filler_zero"]
    if count == 4:
        lengths = result.actual_result["E_r"]
        assert lengths[0] == lengths[1] < lengths[2] == lengths[3]


def test_rejects_non_schedulable_tbs_and_untrusted_expectation(suite):
    _, profiles = suite
    case = profiles["GOLD-QPSK"]
    with pytest.raises(ValueError, match="disagrees"):
        validate_case(case | {"tb_size": 304})
    with pytest.raises(ValueError, match="fixed test policy"):
        validate_case(case | {"expected_behavior": "detect"})
