"""Qualify model-proposed cases with the existing schema and a small verified domain."""

import json

from pydantic import ValidationError

from nr_ai_qa.ai.schemas import Candidate, GenerationRequest
from nr_ai_qa.qa.runner import validate_case
from nr_ai_qa.schemas.ldpc_case import UnsupportedConfiguration
from nr_ai_qa.qa.pusch_oracle import UnsupportedPuschConfiguration

DESCRIPTIVE = {"case_id", "requirement_ids", "risk_tags"}
LDPC_ALLOWED = {"golden", "boundary", "segmentation", "rv"}
PUSCH_MUTABLE = {"seed", "mcs", "tb_size", "num_prbs", "num_rx_ant", "dmrs_config",
                 "snr_db", "num_transport_blocks"}


class RejectedCase(ValueError):
    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


def normalized_parameters(case: dict, ignored=DESCRIPTIVE) -> dict:
    return {k: v for k, v in case.items() if k not in ignored}


def fingerprint(case: dict) -> str:
    return json.dumps(normalized_parameters(case), sort_keys=True, separators=(",", ":"))


def sample_count(case: dict) -> int:
    return case.get("num_transport_blocks", case.get("samples", 1))


def estimated_seconds(case: dict) -> float:
    # Scheduling estimate based on the observed runtime scale, with headroom.
    paths = 3 if case["test_type"] in {"consistency", "noise_fault"} else 1
    return round(0.5 + 0.04 * paths * sample_count(case), 3)


class CaseValidator:
    def __init__(self, registry, request: GenerationRequest):
        self.request = request
        self.profiles = registry.profiles[request.test_family]
        self.topics = {k for k, v in registry.config["topics"].items()
                       if v["family"] == request.test_family and v.get("state") != "UNSUPPORTED"}
        active = {k for k, value in registry.requirements[request.test_family].items()
                  if value["coverage"] == "implemented"}
        if not set(request.requirement_ids) <= active or not set(request.risk_focus) <= self.topics:
            raise ValueError("Generation request cites unknown or unsupported references")
        self.ids = set(self.profiles)
        self.fingerprints = {fingerprint(validate_case(raw).model_dump(mode="json"))
                             for raw in self.profiles.values()}
        self.geometry = {fingerprint(normalized_parameters(validate_case(raw).model_dump(mode="json"),
                                                           DESCRIPTIVE | {"seed"}))
                         for raw in self.profiles.values()}
        self.accepted_count, self.estimate = 0, 0.0

    def accept(self, raw: dict) -> dict:
        try:
            item = Candidate.model_validate(raw)
            if (item.case.get("kind") == "pusch") != (self.request.test_family == "pusch"):
                raise ValueError("Case family does not match request")
            case = validate_case(item.case).model_dump(mode="json")
        except (UnsupportedConfiguration, UnsupportedPuschConfiguration) as exc:
            raise RejectedCase("UNSUPPORTED", str(exc)) from None
        except ValidationError as exc:
            fields = [".".join(map(str, e["loc"])) or "case" for e in exc.errors(include_input=False)]
            raise RejectedCase("INVALID_CASE", f"Schema or PHY constraints failed at {', '.join(fields)}") from None
        except (TypeError, ValueError) as exc:
            raise RejectedCase("INVALID_CASE", str(exc)) from None

        parent = self.profiles.get(item.parent_case_id)
        if parent is None:
            raise RejectedCase("INVALID_CASE", "Parent case is not registered")
        reqs = set(case["requirement_ids"])
        if reqs != set(item.related_requirements) or len(reqs) != len(item.related_requirements):
            raise RejectedCase("INVALID_CASE", "Metadata requirement IDs differ from case.requirement_ids")
        eligible = set(parent["requirement_ids"]) & set(self.request.requirement_ids)
        if not reqs <= eligible:
            allowed = ", ".join(sorted(eligible))
            raise RejectedCase(
                "INVALID_CASE", f"For parent {item.parent_case_id}, use only requirement IDs from [{allowed}]",
            )
        if not set(item.risk_tags + case["risk_tags"]) <= self.topics:
            raise RejectedCase("INVALID_CASE", "Risk tags must identify registered topics")
        if not case["case_id"].startswith("AI-"):
            raise RejectedCase("INVALID_CASE", "Generated case IDs require the AI- prefix")
        key = fingerprint(case)
        if case["case_id"] in self.ids or key in self.fingerprints:
            raise RejectedCase("DUPLICATE", "Case ID or execution parameters already exist")

        limit = min(self.request.max_cases, self.request.execution_budget.max_cases)
        budget = self.request.execution_budget
        estimate = 8.0 + estimated_seconds(case)
        if (self.accepted_count >= limit or sample_count(case) > min(budget.max_tb_per_case, budget.max_generated_tb)
                or self.estimate + estimate > budget.estimated_gpu_seconds):
            raise RejectedCase("RESOURCE_BUDGET", "Exceeds case, TB, or estimated GPU budget")

        canonical_parent = validate_case(parent).model_dump(mode="json")
        if self.request.test_family == "ldpc":
            keep = DESCRIPTIVE | {"seed"}
            domain_ok = case["test_type"] in LDPC_ALLOWED and (
                normalized_parameters(case, keep) == normalized_parameters(canonical_parent, keep)
            )
        else:
            keep = DESCRIPTIVE | PUSCH_MUTABLE
            domain_ok = (
                case["test_type"] == "golden" and case["mcs"] in {2, 4} and case["num_prbs"] in {32, 273}
                and case["dmrs_config"]["scrambling_id"] in {41, 42} and 20 <= case["snr_db"] <= 35
                and normalized_parameters(case, keep) == normalized_parameters(canonical_parent, keep)
            )
        if not domain_ok:
            raise RejectedCase("UNSUPPORTED", "Outside previously verified generation domain")

        self.ids.add(case["case_id"])
        self.fingerprints.add(key)
        self.accepted_count += 1
        self.estimate += estimate
        geometry_key = fingerprint(normalized_parameters(case, DESCRIPTIVE | {"seed"}))
        return {
            "case": case, "metadata": item.model_dump(exclude={"case"}),
            "estimated_gpu_seconds": estimate, "novel_parameters_excluding_seed": geometry_key not in self.geometry,
        }
