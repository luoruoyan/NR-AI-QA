from pathlib import Path
import json

from nr_ai_qa.ai.llm_client import LlmClient
from nr_ai_qa.ai.workflow import run_workflow
from nr_ai_qa.qa.coverage_registry import CoverageRegistry


class TestExecutor:
    __test__ = False
    backend = "TEST_DOUBLE"

    def generated(self, candidate, directory, deadline):
        row = {"case_id": candidate["case"]["case_id"], "status": "PASS",
               "execution_class": "PASS", "evidence_path": None, "actual_result": {},
               "test_parameters": candidate["case"], "requirement_ids": candidate["case"]["requirement_ids"]}
        return {"records": [row], "commands": [], "status": "COMPLETED"}

    def regression(self, suite, case_ids, directory, deadline):
        rows = []
        for case_id in case_ids:
            status = "EXPECTED_DETECTION" if case_id.endswith("-FAULT") else "PASS"
            rows.append({"case_id": case_id, "status": status, "execution_class": status})
        return {"records": rows, "commands": [], "status": "COMPLETED"}


def test_replay_workflow_is_repeatable_without_credentials(tmp_path, monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    registry = CoverageRegistry()
    config = registry.config
    replay = Path("tests/ai/fixtures/replay_pusch_v1.json")
    runs = []
    for name in ("replay-a", "replay-b"):
        client = LlmClient("replay", config["llm"], config["prompt_version"], replay=replay)
        path = tmp_path / name
        result = run_workflow(registry, client, TestExecutor(), path, suite="pusch", max_cases=2)
        assert result["workflow_id"] == name
        assert result["mode"] == "replay" and result["execution_backend"] == "TEST_DOUBLE"
        assert result["status"] == "COMPLETED"
        assert result["stages"]["regression_execution"] == "COMPLETED"
        accepted = json.loads((path / "generated_cases.json").read_text())["accepted"]
        runs.append([r["case"] for r in accepted])
    assert runs[0] == runs[1]
    assert len(runs[0]) == 2


def test_invalid_generation_stops_before_any_execution(tmp_path):
    registry = CoverageRegistry()

    class BadClient:
        mode = "mock"
        model_id = "bad-test-model"
        prompt_version = "test"
        calls = []

        def request(self, purpose, suite, context, schema):
            return {"advice": []} if purpose == "coverage" else {"candidates": [{"case": {"rv": 9}}]}

    summary = run_workflow(registry, BadClient(), TestExecutor(), tmp_path / "negative",
                           suite="ldpc", max_cases=2)
    assert summary["status"] == "BLOCKED"
    assert summary["stages"]["gpu_execution"] == "NOT_RUN"
    assert not (tmp_path / "negative" / "execution_results.json").exists()
