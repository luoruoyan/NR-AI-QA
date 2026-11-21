"""Run trusted profiles or a bounded JSON case using the same QA interface."""

import argparse
import json
from pathlib import Path
import sys

from nr_ai_qa.qa.reporting import write_reports
from nr_ai_qa.qa.runner import QaRunner, load_profiles

RESEARCH_STATUSES = frozenset({"PASS", "EXPECTED_DETECTION", "INCONCLUSIVE"})


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError(f"Invalid JSON numeric constant: {value}")


def load_json_case(path: Path) -> dict:
    with path.open("rb") as stream:
        data = stream.read(65537)
    if len(data) > 65536:
        raise ValueError("JSON case exceeds 64 KiB")
    case = json.loads(data, object_pairs_hook=unique_object, parse_constant=reject_constant)
    if not isinstance(case, dict):
        raise ValueError("The JSON root must be a case object")
    return case


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    choices = parser.add_mutually_exclusive_group()
    choices.add_argument("--case", action="append", help="run the named profile, repeatable")
    choices.add_argument("--json", type=Path, help="execute one structured JSON case")
    parser.add_argument("--suite", choices=("ldpc", "pusch"), default="ldpc")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.json:
            cases = [load_json_case(args.json)]
        else:
            if args.suite == "pusch":
                from nr_ai_qa.qa.pusch_oracle import load_pusch_profiles

                profiles = load_pusch_profiles()
            else:
                profiles = load_profiles()
            if args.case and len(args.case) != len(set(args.case)):
                raise ValueError("Duplicate profile argument")
            cases = [profiles[cid] for cid in args.case] if args.case else list(profiles.values())
    except (OSError, KeyError, UnicodeError, ValueError) as exc:
        print(f"CONFIGURATION_ERROR: {exc}", file=sys.stderr)
        return 2

    suite = "pusch" if cases[0].get("kind") == "pusch" else "ldpc"
    out = args.out or Path("reports") / suite
    with QaRunner(evidence_root=out) as runner:
        records = []
        for case in cases:
            result = runner.execute_case(case)
            records.append(result)
            reason = f" ({result.failure_reason})" if result.failure_reason else ""
            print(f"{result.case_id} {result.status}{reason}", flush=True)
        if suite == "pusch":
            from nr_ai_qa.qa.pusch_reporting import write_pusch_reports

            write_pusch_reports(out, records)
        else:
            write_reports(out, records)
    print(f"Reports: {out}")
    if any(r.status in {"CONFIGURATION_ERROR", "UNSUPPORTED_CONFIGURATION", "UNSUPPORTED"} for r in records):
        return 2
    return int(any(r.status not in RESEARCH_STATUSES for r in records))


if __name__ == "__main__":
    raise SystemExit(main())
