import numpy as np
import pytest

from nr_ai_qa.qa.pusch_oracle import load_pusch_profiles
from nr_ai_qa.qa.runner import QaRunner


@pytest.mark.parametrize(("name", "oracle"), [
    ("P05-DMRS-FAULT", "TB bits or CRC"),
    ("P05-NOISE-FAULT", "active LLR"),
])
def test_p05_detects_repeated_output_change(name, oracle, tmp_path):
    with QaRunner(tmp_path) as runner:
        result = runner.execute_case(load_pusch_profiles()[name])
    assert result.status == "EXPECTED_DETECTION", result.failure_reason
    assert result.expected_result["output_measure"] == oracle
    actual = result.actual_result
    assert actual["clean_passes"] == actual["samples"]
    assert actual["detections"] >= 3
    assert actual["detections"] * 4 >= actual["samples"] * 3
    pair = actual["frames"][actual["first_detected_frame"]]
    evidence = tmp_path / actual["array_evidence"]
    with np.load(evidence) as data:
        assert np.array_equal(data["tx_payload"], data["clean_payload"])
        if oracle == "active LLR":
            assert pair["llr_difference"]["first_difference"] is not None
            repeat_fraction = pair["clean_repeat_llr_difference"]["fraction_changed"]
            injected_fraction = pair["llr_difference"]["fraction_changed"]
            assert repeat_fraction <= 0.01
            assert injected_fraction >= max(0.10, 10 * repeat_fraction)
            assert not np.array_equal(data["clean_llrs"], data["injected_llrs"])
            assert np.allclose(data["injected_applied_noise_db"] - data["clean_applied_noise_db"], 36)
        else:
            assert pair["payload_difference"] is not None or pair["crc_changed"]
            assert (not np.array_equal(data["clean_payload"], data["injected_payload"]) or
                    not np.array_equal(data["clean_crc"], data["injected_crc"]))
