import json

import pytest

from nr_ai_qa.cli import load_json_case
from nr_ai_qa.qa.pusch_oracle import load_pusch_profiles
from nr_ai_qa.qa.runner import QaRunner, ROOT, read_yaml, validate_case


@pytest.fixture(scope="module")
def runner():
    with QaRunner() as instance:
        yield instance


@pytest.mark.parametrize("name", ["P04-MCS", "P04-PRB", "P04-DMRS", "P04-SISO"])
def test_p04_legal_parameter_variations(runner, name):
    case = load_pusch_profiles()[name]
    result = runner.execute_case(case)
    assert result.status == "PASS", result.failure_reason
    for frame in result.actual_result["frames"]:
        assert frame["data_re_count"] == case["num_prbs"] * 132
        assert frame["payload_shapes"] == [[case["tb_size"] // 8]]


@pytest.mark.parametrize("name", ["P04-BAD-TBS", "P04-BAD-IQ"])
def test_p04_invalid_input_rejected_before_receiver(name):
    with QaRunner() as runner:
        result = runner.execute_case(load_pusch_profiles()[name])
        assert runner.pusch_adapter is None
    assert result.status == "PASS", result.failure_reason
    assert result.actual_result["invalid_input_status"] == "CONFIGURATION_ERROR"
    assert not result.actual_result["gpu_execution_started"]


def test_p04_json_validate_execute_collect(runner, tmp_path):
    case = load_pusch_profiles()["P04-MCS"]
    source = tmp_path / "case.json"
    source.write_text(json.dumps(case), encoding="utf-8")
    validated = validate_case(load_json_case(source))
    result = runner.execute_case(validated)
    assert result.status == "PASS", result.failure_reason
    assert runner.collect_result(validated) is result


@pytest.mark.parametrize(("update", "status"), [
    ({"mcs": "2"}, "CONFIGURATION_ERROR"),
    ({"tb_size": 13584}, "CONFIGURATION_ERROR"),
    ({"num_prbs": 274}, "CONFIGURATION_ERROR"),
    ({"channel_type": "tdl"}, "UNSUPPORTED"),
])
def test_invalid_external_cases_do_not_initialize_gpu(update, status):
    case = load_pusch_profiles()["P01-GOLDEN"] | update
    with QaRunner() as runner:
        result = runner.execute_case(case)
        assert runner.pusch_adapter is None
    assert result.status == status
    assert not result.actual_result["gpu_execution_started"]


def test_requirement_map_matches_profiles():
    requirements = read_yaml(ROOT / "specs/pusch_requirements.yaml")["requirements"]
    trace = read_yaml(ROOT / "specs/pusch_traceability.yaml")["test_cases"]
    assert set(load_pusch_profiles()) == set(trace)
    assert {(case, req) for case, reqs in trace.items() for req in reqs} == {
        (case, req["requirement_id"]) for req in requirements for case in req["test_case_ids"]
    }
