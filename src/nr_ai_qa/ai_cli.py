"""Run a bounded NR-AI-QA workflow in live, replay or mock mode."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
from uuid import uuid4

from nr_ai_qa.ai.gpu_execution import GpuExecutor
from nr_ai_qa.ai.llm_client import LlmClient, LlmError, LiveUnavailable, read_object
from nr_ai_qa.ai.workflow import run_workflow, unavailable_summary
from nr_ai_qa.qa.coverage_registry import CoverageRegistry
from nr_ai_qa.qa.runner import ROOT


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, choices=("ldpc", "pusch"))
    parser.add_argument("--mode", required=True, choices=("live", "replay", "mock"))
    parser.add_argument("--replay", type=Path, help="response capture to replay; uses offline fixture if omitted")
    parser.add_argument("--seed", type=int, default=320601)
    parser.add_argument("--max-cases", type=int, default=2)
    parser.add_argument("--change-category", choices=("ldpc_processing", "configuration", "noise_variance"))
    parser.add_argument("--requirement", action="append", help="scope generation to a registered requirement")
    parser.add_argument("--risk-topic", action="append", help="scope generation to a registered risk topic")
    args = parser.parse_args(argv)
    if args.replay and args.mode != "replay":
        parser.error("--replay is only valid with --mode replay")
    if not 1 <= args.max_cases <= 8 or not 0 <= args.seed < 2**32:
        parser.error("Expected 1–8 cases and an unsigned 32-bit seed")
    if args.change_category and (args.change_category == "ldpc_processing") != (args.suite == "ldpc"):
        parser.error("Change category does not match the requested suite")
    replay = args.replay
    if args.mode == "replay" and replay is None:
        replay = ROOT / f"tests/ai/fixtures/replay_{args.suite}_v1.json"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = ROOT / "reports/ai" / f"{stamp}-{args.mode}-{args.suite}-{uuid4().hex[:8]}"
    try:
        registry = CoverageRegistry()
        config = registry.config
        prompt_version = config["prompt_version"]
        if args.mode == "live":
            prompt_version = config["live_prompt_version"]
        elif args.replay is not None:
            recorded_version = read_object(replay).get("prompt_version")
            if recorded_version == config["live_prompt_version"]:
                prompt_version = recorded_version
        try:
            client = LlmClient(args.mode, config["llm"], prompt_version, replay=replay)
        except LiveUnavailable as exc:
            summary = unavailable_summary(directory, args.suite, str(exc), prompt_version)
        else:
            summary = run_workflow(
                registry, client, GpuExecutor(registry), directory, suite=args.suite,
                max_cases=args.max_cases, seed=args.seed, change_category=args.change_category,
                requirement_ids=args.requirement, risk_topics=args.risk_topic,
            )
    except (LlmError, OSError, ValueError) as exc:
        print(f"CONFIGURATION_ERROR {exc}", file=sys.stderr)
        return 2

    print(f"{summary['workflow_id']} {summary['status']}")
    print(f"Report {directory / 'workflow_summary.json'}")
    if summary.get("blocker"):
        print(summary["blocker"], file=sys.stderr)
    if summary["status"] in {"BLOCKED", "LIVE_NOT_VERIFIED"}:
        return 2
    return int(bool(summary.get("new_or_unexpected_findings")))


if __name__ == "__main__":
    raise SystemExit(main())
