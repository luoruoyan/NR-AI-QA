import csv
import json

import pytest

from nr_ai_qa.cli import load_json_case
from nr_ai_qa.qa.reporting import save_evidence, write_reports
from nr_ai_qa.qa.runner import ROOT, read_yaml, validate_case


def test_all_requirements_and_profiles_link_both_directions(suite):
    _, profiles = suite
    records = read_yaml(ROOT / "specs/ldpc_requirements.yaml")["requirements"]
    trace = read_yaml(ROOT / "specs/ldpc_traceability.yaml")["cases"]
    assert set(profiles) == set(trace)
    for case_id, case in profiles.items():
        assert validate_case(case).case_id == case_id
        assert set(case["requirement_ids"]) == set(trace[case_id])
        assert set(trace[case_id]) == {r["requirement_id"] for r in records if case_id in r["test_case_ids"]}
    assert all(case_id in profiles for r in records for case_id in r["test_case_ids"])


def test_json_profile_uses_trusted_oracle_and_reports(suite, tmp_path):
    runner, _ = suite
    data = load_json_case(ROOT / "configs/ldpc_case_example.json")
    case = validate_case(data)
    result = runner.execute_case(case)
    assert runner.collect_result(data) == result
    assert result.status == "PASS", result.failure_reason
    save_evidence(tmp_path, result, {})
    write_reports(tmp_path, [result])
    assert json.loads((tmp_path / "results.json").read_text())[0]["case_id"] == case.case_id
    with (tmp_path / "results.csv").open() as stream:
        assert next(csv.DictReader(stream))["GPU"] == "NVIDIA H20"
    assert result.evidence_path in (tmp_path / "index.html").read_text()
    with pytest.raises(KeyError):
        runner.collect_result(data | {"seed": data["seed"] + 1})


@pytest.mark.parametrize("document", ['{"seed": 0, "seed": 1}', '{"code_rate":NaN}', '[]'])
def test_rejects_ambiguous_or_non_object_json(tmp_path, document):
    path = tmp_path / "case.json"
    path.write_text(document)
    with pytest.raises(ValueError):
        load_json_case(path)
