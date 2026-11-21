from nr_ai_qa.qa.pusch_oracle import load_pusch_profiles
from nr_ai_qa.qa.runner import QaRunner


def test_p01_gpu_golden(tmp_path):
    case = load_pusch_profiles()["P01-GOLDEN"]
    with QaRunner(tmp_path) as runner:
        result = runner.execute_case(case)
        assert runner.collect_result(case) is result
    assert result.status == "PASS", result.failure_reason
    assert result.actual_result["frames"][0]["bit_errors"] == 0
    assert result.actual_result["frames"][0]["crc"] == [0]
    assert (tmp_path / result.actual_result["array_evidence"]).is_file()
