from nr_ai_qa.qa.pusch_oracle import load_pusch_profiles
from nr_ai_qa.qa.runner import QaRunner


def test_p02_gpu_receivers_on_same_iq(tmp_path):
    with QaRunner(tmp_path) as runner:
        result = runner.execute_case(load_pusch_profiles()["P02-CONSISTENCY"])
    assert result.status == "PASS", result.failure_reason
    for frame in result.actual_result["frames"]:
        assert frame["bit_exact"] and frame["output_shapes_match"]
        assert all(t["within_tolerance"] for t in frame["separable_repeat_numeric"].values())
    assert result.actual_result["cross_receiver_intermediates"] == "UNSUPPORTED"
