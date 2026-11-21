"""Human- and machine-readable results from the same records."""

from dataclasses import asdict
import csv
from html import escape
import json
from pathlib import Path

import numpy as np


def json_text(data) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)


def save_evidence(root: Path, result, arrays: dict) -> None:
    path = root / "evidence" / f"{result.case_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    result.evidence_path = path.relative_to(root).as_posix()
    if arrays and result.status in {"EXPECTED_DETECTION", "POTENTIAL_SDK_DEFECT", "ERROR"}:
        raw = path.with_suffix(".npz")
        np.savez_compressed(raw, **arrays)
        result.actual_result["array_evidence"] = raw.relative_to(root).as_posix()
    path.write_text(json_text(asdict(result)) + "\n", encoding="utf-8")


def write_reports(root: Path, results: list, *, title: str = "NR-AI-QA LDPC",
                  extra_metrics: tuple[str, ...] = (), attachments: dict[str, str] | None = None,
                  summary: str = "") -> None:
    root.mkdir(parents=True, exist_ok=True)
    rows = [asdict(result) for result in results]
    (root / "results.json").write_text(json_text(rows) + "\n", encoding="utf-8")
    with (root / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = (list(rows[0]) + ["samples", "tb_errors", "ber", "bler", "bler_ci95", "ber_ci95"]
                  + list(extra_metrics))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            actual = row["actual_result"]
            metrics = {key: actual.get(key) for key in ("samples", "tb_errors", "ber", "bler")}
            metrics.update(bler_ci95=actual.get("bler_ci95_wilson"),
                           ber_ci95=actual.get("ber_ci95_tb_bootstrap"))
            metrics.update({key: actual.get(key, row["expected_result"].get(key))
                            for key in extra_metrics})
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                             for k, v in (row | metrics).items()})

    keys = ("case_id", "status", "GPU", "CUDA_version", "SDK_commit", "duration", "failure_reason")
    header = "".join(f"<th>{escape(k)}</th>" for k in keys) + "<th>Parameters and evidence</th>"
    body = []
    for row in rows:
        values = "".join(f"<td>{escape(str(row[k] if row[k] is not None else ''))}</td>" for k in keys)
        evidence = row.get("evidence_path")
        link = f'<a href="{escape(evidence, quote=True)}">JSON evidence</a>' if evidence else ""
        details = escape(json_text({k: row[k] for k in
                                    ("requirement_ids", "seed", "test_parameters", "expected_result", "actual_result")}))
        body.append(f"<tr>{values}<td>{link}<details><summary>Results</summary><pre>{details}</pre></details></td></tr>")
    links = "".join(f' · <a href="{escape(path, quote=True)}">{escape(label)}</a>'
                    for label, path in (attachments or {}).items())
    summary_html = f"<p>{escape(summary)}</p>" if summary else ""
    (root / "index.html").write_text(
        f'<!doctype html><meta charset="utf-8"><title>{escape(title)}</title>'
        '<style>body{font:16px system-ui;margin:2rem}table{border-collapse:collapse}'
        'td,th{border:1px solid #aaa;padding:.4rem;vertical-align:top}pre{white-space:pre-wrap}</style>'
        f'<h1>{escape(title)}</h1><p><a href="results.json">JSON</a> · <a href="results.csv">CSV</a>{links}</p>'
        f"{summary_html}"
        f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body)}</tbody></table>",
        encoding="utf-8",
    )
