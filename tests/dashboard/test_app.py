from streamlit.testing.v1 import AppTest

from nr_ai_qa.dashboard import live
from nr_ai_qa.dashboard.data import ROOT

APP = ROOT / "src/nr_ai_qa/dashboard/app.py"
PAGES = {
    "Overview": "Overview",
    "Test Coverage & Risk": "Test Coverage & Risk",
    "AI Test Generation": "AI Test Generation",
    "Defect Analysis": "Defect Analysis",
    "Regression Results": "Regression Results",
}


def test_all_replay_pages_run_without_credentials_or_execution(monkeypatch):
    for var in ("LLM_API_KEY", "DASHSCOPE_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("Replay invoked Live execution")

    monkeypatch.setattr(live, "run_live", forbidden)
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    assert app.sidebar.radio(key="mode").value == "REPLAY"
    for label, title in PAGES.items():
        app.sidebar.radio(key="page").set_value(label).run()
        assert not app.exception
        assert app.title[0].value == title
    table = next(df.value for df in app.dataframe if "Current status" in df.value)
    assert "EXPECTED_DETECTION" in set(table["Current status"])


def test_defect_page_keeps_injection_status_separate():
    app = AppTest.from_file(str(APP)).run()
    app.sidebar.radio(key="page").set_value("Defect Analysis").run()
    app.selectbox(key="defect").set_value("P05-NOISE-FAULT").run()
    assert not app.exception
    assert {m.value for m in app.metric} >= {"INJECTED_DEFECT", "EXPECTED_DETECTION"}



def test_live_handoff_requires_confirmation_and_runs_once(monkeypatch):
    calls = []
    monkeypatch.setattr(live, "availability", lambda _: None)

    def record(*args, **kwargs):
        calls.append(args)
        return 2, None

    monkeypatch.setattr(live, "run_live", record)
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    app.sidebar.radio(key="mode").set_value("LIVE").run()
    app.sidebar.radio(key="page").set_value("AI Test Generation").run()
    assert not app.exception
    assert app.button(key="run_live").proto.disabled and not calls
    app.checkbox(key="live_confirm").set_value(True).run()
    assert not app.button(key="run_live").proto.disabled and not calls
    app.button(key="run_live").click().run()
    assert calls == [("pusch", "NR-PUSCH-MCS-TBS", "PUSCH-MCS-PRB")]
    app.run()
    assert len(calls) == 1
