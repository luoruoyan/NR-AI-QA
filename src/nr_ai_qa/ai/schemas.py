"""Structured model proposals. Oracle results are outside all model schemas."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from nr_ai_qa.schemas.ldpc_case import LdpcCase
from nr_ai_qa.schemas.pusch_case import PuschCase

Family = Literal["ldpc", "pusch"]
Mode = Literal["live", "replay", "mock"]
Identifier = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")]
Reason = Literal["boundary", "interaction", "missing_evidence", "reproduce_history", "seed_variation"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)


class Budget(StrictModel):
    max_cases: int = Field(ge=1, le=8)
    max_generated_tb: int = Field(ge=1, le=16)
    max_tb_per_case: int = Field(ge=1, le=96)
    max_regression_cases: int = Field(ge=1, le=12)
    estimated_gpu_seconds: float = Field(ge=10, le=180)
    wall_seconds: float = Field(ge=10, le=600)


class GenerationRequest(StrictModel):
    test_family: Family
    requirement_ids: list[Identifier] = Field(min_length=1, max_length=16)
    risk_focus: list[Identifier] = Field(max_length=12)
    max_cases: int = Field(ge=1, le=8)
    seed: int = Field(ge=0, le=2**32 - 1)
    execution_budget: Budget
    existing_coverage: dict


class Advice(StrictModel):
    topic_id: Identifier
    requirement_ids: list[Identifier] = Field(min_length=1, max_length=8)
    reason: Reason


class AdviceResponse(StrictModel):
    advice: list[Advice] = Field(max_length=12)


class WireCandidateMetadata(StrictModel):
    risk_tags: list[Identifier] = Field(max_length=8)
    risk_reason: Reason
    parent_case_id: Identifier


class Candidate(WireCandidateMetadata):
    related_requirements: list[Identifier] = Field(min_length=1, max_length=16)
    case: dict


class WireLdpcCandidate(WireCandidateMetadata):
    case: LdpcCase


class WirePuschCandidate(WireCandidateMetadata):
    case: PuschCase


class LdpcGeneration(StrictModel):
    candidates: list[WireLdpcCandidate]


class PuschGeneration(StrictModel):
    candidates: list[WirePuschCandidate]


class TriageProposal(StrictModel):
    case_id: Identifier
    hypothesis_ids: list[Identifier] = Field(max_length=5)
    evidence_ids: list[Identifier] = Field(min_length=1, max_length=8)
    investigation_ids: list[Identifier] = Field(max_length=5)
    regression_case_ids: list[Identifier] = Field(max_length=6)
    confidence: Literal["low", "medium", "high"]


class TriageResponse(StrictModel):
    assessments: list[TriageProposal] = Field(max_length=24)


class ChangeContext(StrictModel):
    family: Family
    category: Literal["configuration", "noise_variance", "ldpc_processing"]
    affected_requirements: list[Identifier] = Field(max_length=16)
    affected_modules: list[Identifier] = Field(max_length=12)
    known_failed_case_ids: list[Identifier] = Field(max_length=8)
    model_suggested_case_ids: list[Identifier] = Field(max_length=12)
    known_failure_parameters: dict = Field(default_factory=dict)
