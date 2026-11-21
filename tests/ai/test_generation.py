import json

import pytest

from nr_ai_qa.ai.llm_client import LlmError, clean_json, strict_response_schema
from nr_ai_qa.ai.schemas import PuschGeneration
from nr_ai_qa.ai.test_generator import generate_test_cases
from nr_ai_qa.qa.coverage_registry import CoverageRegistry


@pytest.mark.parametrize("raw", ['{"candidates": [', '{"x": 1, "x": 2}', '{"v": NaN}', '[1]'])
def test_invalid_json_and_nonobject_are_rejected(raw):
    with pytest.raises(LlmError):
        clean_json(raw)


def test_provider_schema_supports_strict_object_mode():
    schema = strict_response_schema(PuschGeneration)
    assert schema["additionalProperties"] is False
    for record in schema["$defs"].values():
        if record.get("type") == "object":
            assert set(record["required"]) == set(record["properties"])
            assert record["additionalProperties"] is False
    assert json.dumps(schema).count('"default"') == 0


def test_generation_repairs_at_most_twice_and_does_not_fallback():
    registry = CoverageRegistry()

    class AllInvalid:
        mode = "replay"
        model_id = "fixture"
        prompt_version = "test"
        calls = 0

        def request(self, *args):
            self.calls += 1
            return {"candidates": [{"case": {"rv": 17}}]}

    from nr_ai_qa.ai.test_generator import make_request

    client = AllInvalid()
    request = make_request(registry, "ldpc", 2, 17)
    summary = generate_test_cases(request, registry, client)
    assert client.calls == 3
    assert not summary["accepted"]
    assert len(summary["rejected"]) == 3


def test_live_transport_requests_strict_schema_and_records_safe_telemetry(monkeypatch):
    from nr_ai_qa.ai.llm_client import LlmClient
    from nr_ai_qa.ai.schemas import AdviceResponse

    class FakeResponse:
        headers = {"x-request-id": "request-fixture-23"}

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self, _limit):
            return json.dumps({
                "choices": [{"message": {"content": "{\"advice\":[]}"}}],
                "usage": {"prompt_tokens": 30, "completion_tokens": 8},
            }).encode()

    class FakeOpener:
        def open(self, request, timeout):
            payload = json.loads(request.data)
            assert request.full_url == "https://provider.example/v1/chat/completions"
            assert request.headers["Authorization"] == "Bearer fake-unit-test-key"
            assert payload["model"] == "test-model" and timeout > 0
            assert payload["response_format"]["json_schema"]["strict"] is True
            return FakeResponse()

    monkeypatch.setenv("LLM_API_KEY", "fake-unit-test-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_BASE_URL", "https://provider.example/v1")
    registry = CoverageRegistry()
    client = LlmClient("live", registry.config["llm"], registry.config["prompt_version"])
    client._opener = FakeOpener()
    response = client.request("coverage", "ldpc", {"topics": []}, AdviceResponse)
    client.validated(True, response)
    assert client.calls[0]["request_id"] == "request-fixture-23"
    assert client.calls[0]["tokens"] == {"input": 30, "output": 8, "total": 38}
    assert client.calls[0]["structure_validation"] == "ACCEPTED"
    assert "fake-unit-test-key" not in json.dumps([client.calls, client.recording])


def test_live_token_budget_is_checked_before_network(monkeypatch):
    from nr_ai_qa.ai.llm_client import LlmClient
    from nr_ai_qa.ai.schemas import AdviceResponse

    monkeypatch.setenv("LLM_API_KEY", "fake-unit-test-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_BASE_URL", "https://provider.example/v1")
    settings = CoverageRegistry().config
    client = LlmClient("live", settings["llm"] | {"max_total_tokens": 1000}, settings["prompt_version"])
    client._opener = None
    with pytest.raises(LlmError, match="token budget"):
        client.request("coverage", "ldpc", {"topics": []}, AdviceResponse)
    assert client.calls[0]["structure_validation"] == "REJECTED"


@pytest.mark.parametrize("host", [
    "dashscope.aliyuncs.com",
    "dashscope-intl.aliyuncs.com",
    "workspace-test.cn-beijing.maas.aliyuncs.com",
])
def test_bailian_flash_uses_nonthinking_strict_output(monkeypatch, host):
    from nr_ai_qa.ai.llm_client import LlmClient
    from nr_ai_qa.ai.schemas import AdviceResponse

    class Stub:
        def open(self, request, timeout):
            payload = json.loads(request.data)
            assert payload["enable_thinking"] is False
            assert payload["response_format"]["json_schema"]["strict"] is True
            assert payload["max_completion_tokens"] > 0
            raise LlmError("End of provider contract test")

    for name in ("LLM_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "fake-unit-test-dashscope-key")
    monkeypatch.setenv("LLM_MODEL", "qwen3.8-flash")
    monkeypatch.setenv("LLM_BASE_URL", f"https://{host}/compatible-mode/v1")
    config = CoverageRegistry().config
    client = LlmClient("live", config["llm"], config["prompt_version"])
    client._opener = Stub()
    with pytest.raises(LlmError, match="End of provider contract test"):
        client.request("coverage", "pusch", {}, AdviceResponse)


def test_dashscope_key_is_scoped_to_the_official_compatible_endpoint(monkeypatch):
    from nr_ai_qa.ai.llm_client import LlmClient, LiveUnavailable

    for name in ("LLM_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "fake-unit-test-dashscope-key")
    monkeypatch.setenv("LLM_MODEL", "qwen3.8-flash")
    monkeypatch.setenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com.example.com/compatible-mode/v1")
    config = CoverageRegistry().config
    with pytest.raises(LiveUnavailable):
        LlmClient("live", config["llm"], config["prompt_version"])


def test_live_model_sets_requirement_ids_once_and_keeps_parent_constraints():
    from nr_ai_qa.ai.test_generator import make_request

    registry = CoverageRegistry()
    request = make_request(registry, "pusch", 1, 7713)

    class Model:
        mode = "live"
        model_id = "test-model"
        prompt_version = "test-version"

        def request(self, purpose, suite, context, schema):
            assert purpose == "generation" and suite == "pusch"
            wire = strict_response_schema(schema)
            assert "related_requirements" not in wire["$defs"]["WirePuschCandidate"]["properties"]
            parent = "P04-PRB"
            rid = "NR-PUSCH-MCS-TBS"
            assert rid in context["eligible_parent_requirements"][parent]
            proposed = context["qualified_parents"][parent] | {
                "case_id": "AI-LIVE-PROPOSAL", "requirement_ids": [rid], "seed": 7713,
                "mcs": 4, "tb_size": 2536, "risk_tags": ["PUSCH-MCS-PRB"],
            }
            return {"candidates": [{"case": proposed, "parent_case_id": parent,
                                    "risk_tags": ["PUSCH-MCS-PRB"], "risk_reason": "interaction"}]}

    result = generate_test_cases(request, registry, Model())
    assert not result["rejected"]
    accepted = result["accepted"][0]
    assert accepted["case"]["requirement_ids"] == accepted["metadata"]["related_requirements"]
    assert accepted["case"]["mcs"] == 4 and accepted["case"]["tb_size"] == 2536
