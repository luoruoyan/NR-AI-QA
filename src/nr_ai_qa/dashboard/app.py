"""Streamlit portfolio for recorded NR-AI-QA results and opt-in Live runs."""

import json
import altair as alt
import pandas as pd
import streamlit as st

from nr_ai_qa.dashboard import live
from nr_ai_qa.dashboard.data import ROOT, load_snapshot, source_path

PAGES = (
    "Overview", "Test Coverage & Risk", "AI Test Generation", "Defect Analysis", "Regression Results",
)
BLUE, TEAL, MUTED = "#245EEA", "#008675", "#4B5D75"
COVERAGE_COLORS = {
    "COVERED": TEAL, "PARTIALLY_COVERED": "#A95613", "NOT_COVERED": "#B73846",
    "UNSUPPORTED": "#5D697B", "PENDING_SPEC_REVIEW": "#7660AB",
}
DEFECTS = {
    "BG-R067": "BG-R067 · Base graph threshold",
    "FAULT-RM-FLIP": "RM Bit Flip",
    "FAULT-CRC-FLIP": "CRC Bit Flip",
    "P05-DMRS-FAULT": "DMRS Configuration Mismatch",
    "P05-NOISE-FAULT": "Noise Variance Injection",
    "P03-SMOKE-LOW": "Low-SNR statistical smoke test",
}
STYLES = """
<style>
[data-testid="stMainBlockContainer"] { padding-top: 2.5rem; padding-bottom: 3rem; max-width: 1460px; }
[data-testid="stSidebar"] { border-right: 1px solid #dbe4f1; }
[data-testid="stSidebar"] h2 { letter-spacing: -.045em; }
h1,h2,h3 { letter-spacing: -.035em; }
[data-testid="stMetric"] { min-height: 93px; }
[data-testid="stMetricValue"] { font-weight: 650; letter-spacing: -.05em; }
[data-testid="stCaptionContainer"] p { color: #4b5d75; }
</style>
"""


def data_table(rows, *, key=None, height="auto"):
    st.dataframe(rows, hide_index=True, width="stretch", height=height, key=key)


def evidence(source):
    if not source:
        return
    path = source_path(source["path"])
    workflow_id = source.get("workflow_id") or "Standalone Stage 1/2 evidence"
    st.caption(f"Evidence source · {workflow_id}")
    st.code(path, language=None, wrap_lines=True)


def title(text, subtitle):
    st.title(text)
    st.caption(subtitle)


def format_workflow(report):
    label = report["mode"].upper()
    result = report["status"].replace("_", " ").title()
    stamp = report["id"].split("-", 1)[0]
    return f"{report['suite'].upper()} · {label} · {result} · {stamp[9:11]}:{stamp[11:13]} UTC"


def duration_chart(workflow):
    rows = workflow["timeline"]
    if not rows:
        st.info("This workflow has no captured activity durations.")
        return
    df = pd.DataFrame(rows)
    chart = alt.Chart(df).mark_bar(cornerRadius=3, size=17).encode(
        x=alt.X("from:Q", title="Cumulative recorded activity (seconds)", axis=alt.Axis(tickCount=6)),
        x2="to:Q",
        y=alt.Y("activity:N", title=None, sort=df["activity"].to_list(),
                axis=alt.Axis(labelLimit=280, labelOverlap=False)),
        color=alt.Color("kind:N", scale=alt.Scale(domain=["Model", "GPU"], range=[BLUE, TEAL]),
                        legend=alt.Legend(title=None, orient="top")),
        tooltip=[alt.Tooltip("activity:N", title="Activity"),
                 alt.Tooltip("duration:Q", title="Recorded seconds", format=".2f")],
    )
    st.altair_chart(chart, width="stretch", height=max(150, 30 * len(rows) + 65))
    st.caption("Reported API and container durations in execution order. Gaps between activities are omitted.")


def link_chart(snapshot, family, metric):
    rows = []
    metric_key = metric.lower()
    ci_key = "bler_ci95_wilson" if metric == "BLER" else "ber_ci95_tb_bootstrap"
    for row in snapshot["statistics"][family]:
        params, obs = row["parameters"], row["observations"]
        cfg = f"MCS {params['mcs']} · {params['num_prbs']} PRB · {params['tb_size']:,} bits"
        if params.get("num_rx_ant"):
            cfg += f" · {params['num_rx_ant']} RX · DMRS ID {params['dmrs_config']['scrambling_id']}"
        else:
            cfg += f" · RV {params['rv']}"
        low, high = obs[ci_key]
        rows.append({"dB": row["db"], "value": obs[metric_key], "low": low, "high": high,
                     "case": row["case_id"], "status": row["status"], "TBs": obs["samples"],
                     "seed": params["seed"], "config": cfg,
                     "source": row["evidence"]["path"]})
    data = pd.DataFrame(rows)
    x_title = "Es/N₀ (dB)" if family == "ldpc" else "SNR per receive branch (dB)"
    ci_label = "95% Wilson interval" if metric == "BLER" else "95% TB bootstrap interval"
    x = alt.X("dB:Q", title=x_title, scale=alt.Scale(zero=False), axis=alt.Axis(tickCount=5))
    y_scale = alt.Scale(domain=[0, 1.03])
    base = alt.Chart(data)
    line = base.transform_filter(alt.datum.status == "PASS").mark_line(color=BLUE, opacity=.65).encode(
        x=x, y=alt.Y("value:Q", title=metric, scale=y_scale))
    interval = base.mark_rule(color=MUTED, opacity=.7).encode(x=x, y=alt.Y("low:Q", scale=y_scale), y2="high:Q")
    points = base.mark_point(size=72, filled=True).encode(
        x=x, y=alt.Y("value:Q", title=metric, scale=y_scale),
        color=alt.Color("status:N", scale=alt.Scale(domain=["PASS", "INCONCLUSIVE"], range=[BLUE, "#A95613"]),
                        legend=alt.Legend(title="Oracle", orient="top")),
        shape=alt.Shape("status:N", scale=alt.Scale(domain=["PASS", "INCONCLUSIVE"],
                                                   range=["circle", "diamond"]), legend=None),
        tooltip=[alt.Tooltip("case:N", title="Case"), "status:N", "dB:Q",
                 alt.Tooltip("value:Q", title=metric, format=".5f"),
                 alt.Tooltip("low:Q", title=f"{ci_label}, lower", format=".4f"),
                 alt.Tooltip("high:Q", title=f"{ci_label}, upper", format=".4f"),
                 "TBs:Q", "seed:N", alt.Tooltip("config:N", title="PHY configuration")],
    )
    st.altair_chart(line + interval + points, width="stretch", height=230)
    st.caption(f"{snapshot['statistics'][family][0]['snr_definition']} · {ci_label}")
    with st.expander("Samples and source evidence"):
        data_table(data[["case", "dB", "TBs", "seed", "config", "status", "value", "low", "high", "source"]])


def overview(snapshot, workflow):
    title("Overview", "Recorded software QA results for NVIDIA pyAerial")
    gpu, host, calls, cases = st.columns(4, border=True)
    gpu.metric("GPU pytest", snapshot["test_counts"]["gpu"]["passed"])
    host.metric("AI host pytest", snapshot["test_counts"]["host"]["passed"])
    calls.metric("Model API calls", workflow["model_calls"])
    cases.metric("Regression cases executed", workflow.get("regression", {}).get("executed", 0))
    env = snapshot["environment"]
    with st.container(border=True):
        left, right = st.columns([1.25, 1])
        left.markdown("**Verified environment**")
        left.write(f"{env['GPU']} · SM {env['compute_capability']} · CUDA {env['CUDA']}")
        left.caption(f"Aerial {env['Aerial']} · pyAerial {env['pyAerial']}")
        right.markdown("**Applicable specifications · Release 17**")
        right.write(" · ".join(f"{spec} v{version}" for spec, version in snapshot["standards"].items()))
        right.caption(f"SDK commit {env['SDK_commit'][:12]}")
    st.caption("Stage 1 · LDPC verified    /    Stage 2 · PUSCH verified    /    Stage 3 · PUSCH Live verified")
    st.subheader("Link performance")
    col1, col2 = st.columns(2)
    with col1:
        col, control = st.columns([1, 1])
        col.markdown("**LDPC / rate matching**")
        metric = control.radio("LDPC link metric", ["BLER", "BER"], horizontal=True, label_visibility="collapsed")
        link_chart(snapshot, "ldpc", metric)
    with col2:
        st.markdown("**PUSCH receiver**")
        link_chart(snapshot, "pusch", "BLER")
    st.subheader("Execution timeline")
    st.caption(f"{workflow['mode'].upper()} recorded workflow · {workflow['id']}")
    duration_chart(workflow)
    with st.expander("Stage results and baseline logs"):
        data_table([
            {"Stage": "1 · LDPC / rate matching", "Status": "Verified", "Evidence": "reports/ldpc/results.json"},
            {"Stage": "2 · PUSCH receiver", "Status": "Verified", "Evidence": "reports/pusch/results.json"},
            {"Stage": "3 · AI QA", "Status": "PUSCH Live verified",
             "Evidence": f"reports/ai/{snapshot['demo_workflow_id']}/workflow_summary.json"},
        ])
        st.caption(f"Latest archived workflow ID · {snapshot['latest_workflow_id']}")
        st.caption("Workflow origins in the archive · Live, Replay and Mock")
        for count in snapshot["test_counts"].values():
            evidence(count["source"])


def coverage(snapshot, workflow):
    title("Test Coverage & Risk", "Requirement traceability and risk ranking from recorded evidence")
    family = st.radio("Suite", ["pusch", "ldpc"], format_func=str.upper, horizontal=True, key="coverage_suite")
    scope = snapshot["coverage"][family]
    metrics = scope["metrics"]["requirement"]
    a, b, c = st.columns(3, border=True)
    a.metric("Fully covered requirements", f"{metrics['fully_covered']} / {metrics['denominator']}")
    b.metric("Partially covered", metrics["partially_covered"])
    c.metric("Pending specification review", metrics["pending_spec_review"])
    st.caption(metrics["definition"])
    st.markdown("**Observed coverage changes**")
    changes = workflow["scenario_changes"] if workflow["suite"] == family else []
    if changes:
        data_table([{"Scenario": x["topic_id"], "Before": x["before"], "After": x["after"],
                     "GPU evidence cases": ", ".join(x["case_ids"])} for x in changes])
        with st.expander("Coverage change source"):
            evidence(changes[0]["source"])
    else:
        st.info("No change is recorded for this workflow and suite. Select the verified PUSCH Live record to see the MCS × PRB gap close.")

    left, right = st.columns([1.25, 1])
    with left:
        st.subheader("Requirement coverage matrix")
        rows = []
        for rid, value in scope["requirements"].items():
            rows.append({"Requirement": rid, "State": value["state"],
                         "Evidence": sum(e["qualifies"] for e in value["evidence"]),
                         "Linked cases": len(value["case_ids"]), "Clause": value["clause"]})
        matrix = pd.DataFrame(rows).style.map(
            lambda state: f"color: {COVERAGE_COLORS[state]}; font-weight: 600", subset=["State"],
        )
        st.dataframe(matrix, hide_index=True, width="stretch", height=min(410, 44 + 36 * len(rows)),
                     column_config={"Requirement": st.column_config.TextColumn(width=180),
                                    "State": st.column_config.TextColumn(width=170),
                                    "Evidence": st.column_config.NumberColumn(width=80),
                                    "Linked cases": st.column_config.NumberColumn(width=95),
                                    "Clause": st.column_config.TextColumn(width=240)})
        rid = st.selectbox("Inspect requirement evidence", list(scope["requirements"]), key="coverage_req")
        record = scope["requirements"][rid]
        st.caption(record["clause"])
        candidates = {e["case_id"]: e for e in record["evidence"]}
        if candidates:
            cid = st.selectbox("Evidence case", list(candidates), key=f"ev-{rid}")
            item = candidates[cid]
            st.write(f"{item['status']} · {item['basis']}")
            result = snapshot["cases"][family].get(cid)
            with st.expander("Recorded observations and source", expanded=True):
                evidence(item["source"])
                if result:
                    st.json(result["observations"], expanded=False)
        else:
            st.info("No qualifying evidence is recorded for this requirement.")
    with right:
        st.subheader("Risk ranking")
        ranked = [r for r in snapshot["ranking"] if r["family"] == family]
        df = pd.DataFrame([{"Topic": r["topic_id"], "Score": r["score"],
                            "Requirements": ", ".join(r["requirement_ids"])} for r in ranked])
        if not df.empty:
            chart = alt.Chart(df).mark_bar(color=BLUE, cornerRadiusEnd=4, size=18).encode(
                x=alt.X("Score:Q", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("Topic:N", sort="-x", title=None, axis=alt.Axis(labelLimit=190)),
                tooltip=["Topic:N", alt.Tooltip("Score:Q", format=".3f"), "Requirements:N"],
            ).properties(height=210)
            st.altair_chart(chart, width="stretch")
            topic = st.selectbox("Inspect risk factors", [r["topic_id"] for r in ranked], key="coverage_risk")
            details = next(r for r in ranked if r["topic_id"] == topic)
            st.caption("Risk weights are an engineering heuristic.")
            data_table([{"Factor": name.replace("_", " "), "Contribution": round(value, 3)}
                        for name, value in details["contributions"].items()])
            st.caption("Requirements · " + ", ".join(details["requirement_ids"]))
            evidence(details["source"])


def _scope(snapshot, family):
    return [row for row in snapshot["ranking"] if row["family"] == family]


def _live_form(snapshot, family):
    st.warning("Starting Live triggers a real model API request and, for a validated case, GPU execution.")
    address = st.get_option("server.address") or ""
    blocker = live.availability(address)
    if blocker:
        st.info(blocker)
    import yaml

    limits = yaml.safe_load((ROOT / "configs/ai_config.yaml").read_text(encoding="utf-8"))
    budget = limits["budget"]
    st.caption(f"Run budget · 1 candidate · {budget['max_generated_tb']} generated TB max per case · "
               f"{budget['max_regression_cases']} regressions max · {budget['wall_seconds']} s wall budget · "
               f"{limits['llm']['max_total_tokens']:,} reserved tokens max")
    parent_types = {"golden"} if family == "pusch" else {"golden", "boundary", "segmentation", "rv"}
    supported = {
        rid for case in snapshot["cases"][family].values()
        if case["status"] == "PASS" and case["parameters"]["test_type"] in parent_types
        for rid in case["requirement_ids"]
    }
    options = {r["topic_id"]: [rid for rid in r["requirement_ids"] if rid in supported]
               for r in _scope(snapshot, family) if supported.intersection(r["requirement_ids"])}
    chosen = st.selectbox("Live risk direction", list(options), key="live_topic")
    requirement = st.selectbox("Live requirement", options[chosen], key="live_req")
    confirmed = st.checkbox("Authorize one model and GPU workflow", key="live_confirm")
    if st.button("Run one Live workflow", type="primary", disabled=bool(blocker) or not confirmed, key="run_live"):
        if confirmed and blocker is None:
            try:
                with st.spinner("Running the existing Stage 3 workflow…"):
                    code, result = live.run_live(family, requirement, chosen, bind_address=address)
            except ValueError as exc:
                st.error(str(exc))
            except (OSError, KeyError):
                st.error("The local workflow could not start or produce a valid report.")
            else:
                if result:
                    st.session_state["dashboard_live_result"] = result
                    st.success(f"Workflow {result['id']} finished with {result['status']}; exit code {code}.")
                else:
                    st.error(f"The CLI returned exit code {code} before a workflow report was available.")
    result = st.session_state.get("dashboard_live_result")
    return result if result and result["suite"] == family else None


def generation(snapshot, selected, mode):
    title("AI Test Generation", "Structured proposals, independent validation and recorded GPU execution")
    family = st.radio("Test family", ["pusch", "ldpc"], index=0 if selected["suite"] == "pusch" else 1,
                      format_func=str.upper, horizontal=True, key="generation_suite")
    new_workflow = _live_form(snapshot, family) if mode == "LIVE" else None
    records = [r for r in snapshot["workflows"] if r["suite"] == family]
    if new_workflow:
        records = [new_workflow, *records]
    options = {r["id"]: r for r in records}
    default = (new_workflow or selected)["id"]
    keys = list(options)
    picked = st.selectbox("Generation record", keys, index=keys.index(default) if default in keys else 0,
                          format_func=lambda k: format_workflow(options[k]), key=f"gen_workflow-{family}")
    report = options[picked]
    counts = report.get("generation", {})
    a, b, c = st.columns(3, border=True)
    a.metric("Validation accepted", len(report["generated"]))
    b.metric("Validation rejected", len(report["rejected"]))
    c.metric("GPU cases executed", counts.get("real_gpu", 0))
    left, right = st.columns(2)
    req_ids = sorted({rid for row in _scope(snapshot, family) for rid in row["requirement_ids"]})
    rid = left.selectbox("Requirement filter", ["All", *req_ids], key=f"gen_req-{family}")
    risk_id = right.selectbox("Risk direction filter", ["All", *[r["topic_id"] for r in _scope(snapshot, family)]],
                              key=f"gen_risk-{family}")
    filtered = [r for r in report["generated"]
                if (rid == "All" or rid in r["case"]["requirement_ids"])
                and (risk_id == "All" or risk_id in r["metadata"]["risk_tags"])]
    if filtered:
        labels = {r["case"]["case_id"]: r for r in filtered}
        cid = st.selectbox("Generated case", list(labels), key=f"generated_case-{family}")
        row = labels[cid]
        source = row["metadata"]
        st.markdown(f"**{cid}** · {source['generation_source']} · {source['model_id']}")
        st.caption(f"Parent {source['parent_case_id']} · Prompt {source['prompt_version']}")
        lcol, rcol = st.columns([1, 1])
        with lcol:
            st.subheader("Validated case")
            st.json(row["case"], expanded=1)
            st.download_button("Download case JSON", json.dumps(row["case"], indent=2) + "\n",
                               file_name=f"{cid}.json", mime="application/json")
        with rcol:
            st.subheader("Execution evidence")
            if row["execution"]:
                obs = row["execution"]
                st.write(f"{obs['status']} · {obs['execution']} · {obs['GPU']}")
                st.json(obs["observations"], expanded=2)
                evidence(obs["evidence"])
            else:
                st.info("This proposal has no execution record.")
    else:
        st.info("No accepted proposal matches this record and filter.")
    st.subheader("Validation rejections")
    if report["rejected"]:
        data_table(report["rejected"])
    else:
        st.caption("No rejected proposals in this record.")
    evidence(report["source"])


def defects(snapshot):
    title("Defect Analysis", "Oracle observations with advisory triage and source evidence")
    choices = list(DEFECTS)
    cid = st.selectbox("Case", choices, format_func=DEFECTS.__getitem__, key="defect")
    row = snapshot["defects"][cid]
    category = "INJECTED_DEFECT" if row["failure_category"] == "CONTROLLED_INJECTION" else row["failure_category"]
    a, b = st.columns(2, border=True)
    a.metric("Defect classification", category)
    b.metric("Original Oracle status", row["execution_status"])
    st.write(row["observation"])
    st.caption("Requirements · " + ", ".join(row["requirement_ids"]))
    p, e, o = st.columns(3)
    with p:
        st.markdown("**Test parameters**")
        st.json(row["parameters"], expanded=1)
    with e:
        st.markdown("**Expected**")
        st.json(row["expected"], expanded=1)
    with o:
        st.markdown("**Actual**")
        st.json(row["actual"], expanded=1)
    with st.expander("Measured difference and source evidence"):
        if row["difference"] is not None:
            st.json(row["difference"], expanded=1)
        evidence(row["evidence"])
        for source in row["logs"]:
            evidence(source)
    st.subheader("AI root-cause hypotheses")
    if not row["assessments"]:
        st.info("No advisory assessment is recorded.")
    for item in row["assessments"]:
        st.caption(f"Confidence · {item['confidence']} · {item['model_review']}")
        for candidate in item["root_cause_hypotheses"]:
            st.write(f"• {candidate['hypothesis']}")
        for action in item["suggested_investigation"]:
            st.write(f"• {action['action']}")
        st.caption(item["uncertainty"])
        evidence(item["source"])


def regressions(workflow):
    title("Regression Results", "Selected cases, selection reasons and recorded Oracle outcomes")
    summary = workflow.get("regression", {})
    a, b, c = st.columns(3, border=True)
    a.metric("Selected", summary.get("selected", 0))
    b.metric("Executed on H20", summary.get("real_gpu", 0))
    c.metric("Historical status changes", len(summary.get("status_changes", [])))
    rows = workflow["regressions"]
    if not rows:
        st.info("This workflow stopped before regression execution.")
        evidence(workflow["source"])
        return
    counts = summary.get("status_counts", {})
    if counts:
        frame = pd.DataFrame([{"Status": k, "Cases": v} for k, v in counts.items()])
        chart = alt.Chart(frame).mark_bar(size=17, cornerRadiusEnd=4).encode(
            x=alt.X("Cases:Q", axis=alt.Axis(tickMinStep=1)),
            y=alt.Y("Status:N", title=None, sort="-x", axis=alt.Axis(labelLimit=225, labelOverlap=False)),
            color=alt.Color("Status:N", legend=None,
                            scale=alt.Scale(domain=["PASS", "EXPECTED_DETECTION", "INCONCLUSIVE",
                                                    "POTENTIAL_SDK_DEFECT"],
                                            range=[BLUE, TEAL, "#A95613", "#B73846"])),
            tooltip=["Status:N", "Cases:Q"],
        )
        st.altair_chart(chart, width="stretch", height=max(145, 36 * len(counts) + 70))
    data_table([{"Case": r["case_id"], "Priority": r["priority"], "Prior status": r["prior_status"],
                 "Current status": (r["execution"] or {}).get("status", "NOT_RUN"),
                 "Selection reason": r["selection_reason"], "Requirements": ", ".join(r["requirement_ids"])}
                for r in rows])
    options = {r["case_id"]: r for r in rows}
    cid = st.selectbox("Inspect case evidence", list(options), key="regression_case")
    row = options[cid]
    with st.expander("Parameters, observed outcome and references", expanded=True):
        if row["execution"]:
            st.json({"parameters": row["execution"]["parameters"],
                     "observations": row["execution"]["observations"]}, expanded=1)
            evidence(row["execution"]["evidence"])
        evidence(row["prior_evidence"])


def main():
    st.set_page_config(page_title="NR-AI-QA", layout="wide", initial_sidebar_state="expanded")
    st.html(STYLES)
    try:
        snapshot = load_snapshot()
    except (OSError, ValueError):
        st.error("The portable Replay snapshot could not be loaded.")
        st.stop()
    st.sidebar.markdown("## NR-AI-QA")
    st.sidebar.caption("5G NR software quality assurance")
    page = st.sidebar.radio("Page", PAGES, key="page", label_visibility="collapsed")
    st.sidebar.divider()
    mode = st.sidebar.radio("Mode", ["REPLAY", "LIVE"], horizontal=True, key="mode")
    if mode == "REPLAY":
        st.sidebar.info("Historical evidence · no API or GPU execution")
    else:
        st.sidebar.warning("Live execution requires your confirmation on AI Test Generation.")
    records = {r["id"]: r for r in snapshot["workflows"]}
    new = st.session_state.get("dashboard_live_result")
    if new:
        records[new["id"]] = new
    default = snapshot["demo_workflow_id"]
    wid = st.sidebar.selectbox("Workflow record", list(records), index=list(records).index(default),
                               format_func=lambda k: format_workflow(records[k]), key="workflow")
    selected = records[wid]
    st.sidebar.caption(wid)
    st.sidebar.caption(f"Latest archived ID · {snapshot['latest_workflow_id']}")
    if page == "Overview":
        overview(snapshot, selected)
    elif page == "Test Coverage & Risk":
        coverage(snapshot, selected)
    elif page == "AI Test Generation":
        generation(snapshot, selected, mode)
    elif page == "Defect Analysis":
        defects(snapshot)
    else:
        regressions(selected)


if __name__ == "__main__":
    main()
