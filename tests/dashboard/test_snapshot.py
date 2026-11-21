import json
from pathlib import Path

import pytest

from nr_ai_qa.dashboard.data import ROOT, load_snapshot, build_snapshot, source_path


def test_snapshot_has_provenance_and_actual_results():
    snapshot = load_snapshot()
    live = next(w for w in snapshot["workflows"] if w["live_verified"])
    assert live["id"] == "20261121T075721Z-live-pusch-774b8bcb"
    assert live["model_calls"] == 3
    assert len(live["regressions"]) == 7
    result = live["generated"][0]["execution"]
    assert result["status"] == "PASS" and result["observations"]["tb_bit_errors"] == [0]
    assert live["scenario_changes"][0]["before"] == "NOT_COVERED"
    assert live["scenario_changes"][0]["after"] == "COVERED"
    assert "runner_log" not in result
    assert "request_id" not in json.dumps(snapshot)


def test_snapshot_matches_local_sources_when_present():
    if not (ROOT / "reports/ldpc/results.json").exists():
        pytest.skip("Archive validation needs the original private sources")
    assert build_snapshot(ROOT) == load_snapshot()


@pytest.mark.parametrize("unsafe", ["/etc/passwd", "reports/../../etc/passwd", "https://example.com/a",
                                  "reports\\private.json", "reports/ldpc/../private.json"])
def test_provenance_rejects_external_paths(unsafe):
    with pytest.raises(ValueError):
        source_path(unsafe)
