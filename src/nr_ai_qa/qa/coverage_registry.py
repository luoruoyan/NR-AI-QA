"""Version-checked evidence and bounded coverage denominators."""

import json
from pathlib import Path

from nr_ai_qa.qa.pusch_oracle import load_pusch_profiles
from nr_ai_qa.qa.runner import PINNED_COMMIT, ROOT, load_profiles, read_yaml, validate_case

QUALIFYING = {"PASS", "EXPECTED_DETECTION"}
IMAGE_DIGEST = "sha256:fe79b31fe1703d6dbca78c72fa4463b0192804c97fb45f97d8ce740513affa24"


def counts(rows: list[dict], definition: str) -> dict:
    states = [row["state"] for row in rows]
    denominator = sum(x not in {"UNSUPPORTED", "PENDING_SPEC_REVIEW"} for x in states)
    qualified = sum(x in {"COVERED", "PARTIALLY_COVERED"} for x in states)
    return {
        "definition": definition, "numerator": qualified, "denominator": denominator,
        "fully_covered": states.count("COVERED"), "partially_covered": states.count("PARTIALLY_COVERED"),
        "not_covered": states.count("NOT_COVERED"), "unsupported": states.count("UNSUPPORTED"),
        "pending_spec_review": states.count("PENDING_SPEC_REVIEW"),
    }


def evidence_is_gpu(row: dict) -> bool:
    actual = row.get("actual_result", {})
    environment = actual.get("environment", {})
    return bool(
        row.get("SDK_commit") == PINNED_COMMIT and row.get("GPU") == "NVIDIA H20" and
        environment.get("commit") == PINNED_COMMIT and environment.get("compute_capability") == "9.0"
        and row.get("CUDA_version") == "13.3" and environment.get("image_digest") == IMAGE_DIGEST
        and actual.get("gpu_execution_started") is True
    )


class CoverageRegistry:
    def __init__(self, root: Path = ROOT):
        self.root = root
        self.config = read_yaml(root / "configs/ai_config.yaml")
        self.profiles = {"ldpc": load_profiles(), "pusch": load_pusch_profiles()}
        self.requirements, self.trace, self.evidence = {}, {}, {}
        for family in self.profiles:
            records = read_yaml(root / f"specs/{family}_requirements.yaml")["requirements"]
            self.requirements[family] = {r["requirement_id"]: r for r in records}
            traces = read_yaml(root / f"specs/{family}_traceability.yaml")
            self.trace[family] = traces["cases" if family == "ldpc" else "test_cases"]
            self._check_trace(family, records)
            self.evidence[family] = self._load_evidence(family)

    def _check_trace(self, family: str, records: list[dict]) -> None:
        requirements, trace = self.requirements[family], self.trace[family]
        if len(records) != len(requirements) or set(trace) != set(self.profiles[family]):
            raise ValueError(f"Inconsistent {family} registry IDs")
        forward = {(cid, rid) for cid, ids in trace.items() for rid in ids}
        reverse = {(cid, r["requirement_id"]) for r in records for cid in r["test_case_ids"]}
        configured = {
            (cid, rid) for cid, case in self.profiles[family].items() for rid in case["requirement_ids"]
        }
        if forward != reverse or forward != configured:
            raise ValueError(f"Inconsistent {family} requirement references")

    def _load_evidence(self, family: str) -> dict:
        root = self.root / "reports" / family
        rows = json.loads((root / "results.json").read_text())
        evidence = {}
        for row in rows:
            case_id = row["case_id"]
            if case_id in evidence or case_id not in self.trace[family]:
                raise ValueError(f"Unknown or duplicate historical case ID {case_id}")
            if set(row["requirement_ids"]) != set(self.trace[family][case_id]):
                raise ValueError(f"Historical case {case_id} has mismatched requirements")
            path = (root / row["evidence_path"]).resolve()
            if not path.is_relative_to(root.resolve()) or not path.is_file():
                raise ValueError(f"Missing in-scope evidence for {case_id}")
            expected = validate_case(self.profiles[family][case_id]).model_dump(mode="json")
            if row["test_parameters"] != expected:
                raise ValueError(f"Historical input does not match the registered case {case_id}")
            if json.loads(path.read_text()) != row:
                raise ValueError(f"Historical evidence differs from the report for {case_id}")
            native = evidence_is_gpu(row)
            input_contract = (
                case_id in {"P04-BAD-IQ", "P04-BAD-TBS"} and row["SDK_commit"] == PINNED_COMMIT
                and row["actual_result"].get("invalid_input_status") == "CONFIGURATION_ERROR"
                and row["actual_result"].get("gpu_execution_started") is False
            )
            evidence[case_id] = {
                "case_id": case_id, "status": row["status"],
                "path": path.relative_to(self.root).as_posix(),
                "basis": "REAL_GPU" if native else "HOST_CONTRACT" if input_contract else "UNVERIFIED",
                "qualifies": (native or input_contract) and row["status"] in QUALIFYING,
                "row": row,
            }
        return evidence

    @staticmethod
    def _coverage(ids: list[str], evidence: dict) -> dict:
        linked = [evidence[cid] for cid in ids if cid in evidence]
        hits = sum(row["qualifies"] for row in linked)
        state = "COVERED" if ids and hits == len(ids) else "PARTIALLY_COVERED" if hits else "NOT_COVERED"
        return {"state": state, "case_ids": ids,
                "evidence": [{k: v for k, v in row.items() if k != "row"} for row in linked]}

    def snapshot(self, new_evidence: list[dict] | None = None) -> dict:
        """Add only actual generated GPU results to explicitly defined scenario gaps."""
        topics, suites = {}, {}
        for key, topic in self.config["topics"].items():
            family = topic["family"]
            if not set(topic["requirement_ids"]) <= self.requirements[family].keys():
                raise ValueError(f"Unknown requirements for topic {key}")
            if not set(topic["cases"]) <= self.profiles[family].keys():
                raise ValueError(f"Unknown cases for topic {key}")
            record = self._coverage(topic["cases"], self.evidence[family])
            if topic.get("parameters"):
                matches = [row for row in (new_evidence or []) if evidence_is_gpu(row)
                           and row["status"] == "PASS"
                           and row["test_parameters"].get("kind") == family
                           and set(topic["requirement_ids"]) <= set(row["requirement_ids"])
                           and all(row["test_parameters"].get(k) == v
                                   for k, v in topic["parameters"].items())]
                if matches:
                    record = {
                        "state": "COVERED", "case_ids": [row["case_id"] for row in matches],
                        "evidence": [{"case_id": row["case_id"], "status": row["status"],
                                      "basis": "REAL_GPU", "qualifies": True,
                                      "path": row["evidence_path"]} for row in matches],
                    }
            topics[key] = topic | record | {"state": topic.get("state", record["state"])}

        for family, requirements in self.requirements.items():
            evidence = self.evidence[family]
            req_coverage = {}
            for rid, requirement in requirements.items():
                available = self._coverage(requirement["test_case_ids"], evidence)
                if requirement["coverage"] != "implemented":
                    available["state"] = "PENDING_SPEC_REVIEW"
                req_coverage[rid] = {"clause": requirement["clause"],
                                     "kind": requirement.get("kind", requirement.get("category")),
                                     **available}
            family_topics = [row for row in topics.values() if row["family"] == family]
            regression = [self._coverage([cid], evidence) for cid in self.profiles[family]]
            suites[family] = {
                "requirements": req_coverage,
                "native_gpu_cases": sorted(cid for cid, row in evidence.items() if row["basis"] == "REAL_GPU"),
                "metrics": {
                    "requirement": counts(list(req_coverage.values()),
                                          "Verified in-scope requirements with qualifying evidence / implemented requirements"),
                    "parameter_boundary": counts([p for p in family_topics if p["axis"] == "boundary"],
                                                 "Boundary groups with qualifying evidence / curated applicable boundary groups"),
                    "supported_scenario": counts([p for p in family_topics if p["axis"] == "supported_scenario"],
                                                 "Scenarios with qualifying evidence / curated supported scenarios"),
                    "regression": counts(regression,
                                         "Registered profiles with qualifying baseline / registered regression profiles"),
                },
            }
        return {"SDK_commit": PINNED_COMMIT, "topics": topics, "suites": suites}
