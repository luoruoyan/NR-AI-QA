"""Untrusted JSON/YAML input boundary. Expected behavior labels are descriptive."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nr_ai_qa.qa.oracle import SMALL_TBS, geometry, mcs_parameters, scheduled_tbs


class UnsupportedConfiguration(ValueError):
    """A well-typed request outside the supported SDK test profile."""


class LdpcCase(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)

    case_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
    requirement_ids: list[str] = Field(min_length=1, max_length=16)
    test_type: Literal["golden", "boundary", "segmentation", "rv", "awgn", "fault", "bg_probe"]
    tb_size: int = Field(ge=24, le=16384, multiple_of=8)
    mcs: int | None = Field(default=None, ge=0, le=28)
    code_rate: float = Field(gt=0, le=0.95)
    rv: int = Field(default=0, ge=0, le=3)
    num_prbs: int | None = Field(default=None, ge=1, le=273)
    data_re_per_prb: int | None = Field(default=None, ge=12, le=156, multiple_of=6)
    channel_type: Literal["noiseless", "awgn"] = "noiseless"
    esno_db: float | None = Field(default=None, ge=-12, le=15)
    seed: int = Field(ge=0, le=2**32 - 1)
    samples: int = Field(default=1, ge=1, le=128)
    expected_behavior: Literal["round_trip", "geometry", "measure", "detect"]
    risk_tags: list[str] = Field(default_factory=list, max_length=12)
    fault: Literal["rm_bit_flip", "pre_crc_bit_flip"] | None = None

    @property
    def mod_order(self) -> int:
        return mcs_parameters(self.mcs)[0] if self.mcs is not None else 2

    @property
    def g(self) -> int:
        return self.mod_order * (self.num_prbs or 0) * (self.data_re_per_prb or 0)

    @model_validator(mode="after")
    def validate_profile(self):
        expected = {"awgn": "measure", "fault": "detect", "bg_probe": "geometry"}
        if self.expected_behavior != expected.get(self.test_type, "round_trip"):
            raise ValueError("expected_behavior must match the fixed test policy")
        if len(self.requirement_ids) != len(set(self.requirement_ids)):
            raise ValueError("Duplicate requirement IDs")
        if self.test_type == "bg_probe":
            if self.mcs is not None or self.num_prbs is not None or self.data_re_per_prb is not None:
                raise ValueError("A component BG probe must have no scheduled allocation")
            if self.tb_size not in SMALL_TBS and self.tb_size not in (3840, 3848):
                raise ValueError("Unsupported component probe TB size")
        else:
            if self.mcs is None or self.num_prbs is None or self.data_re_per_prb is None:
                raise ValueError("MCS, num_prbs and data_re_per_prb are required")
            if self.code_rate != mcs_parameters(self.mcs)[1]:
                raise ValueError("code_rate must equal the normalized rate from 38.214 table 1")
            if self.tb_size != scheduled_tbs(self.g, self.code_rate):
                raise ValueError("tb_size disagrees with 38.214 for the supplied allocation")
            if self.g <= self.tb_size:
                raise ValueError("Stage 1 requires G greater than the uncoded payload length")
        if self.test_type == "awgn":
            if self.channel_type != "awgn" or self.esno_db is None:
                raise ValueError("AWGN requires Es/N0 in dB")
        elif self.samples != 1 or self.channel_type != "noiseless" or self.esno_db is not None:
            raise ValueError("Deterministic cases require one noiseless sample")
        if (self.fault is not None) != (self.test_type == "fault"):
            raise ValueError("A controlled fault is required exactly for fault cases")
        return self


def validate_profile_budget(case: LdpcCase) -> LdpcCase:
    try:
        geo = geometry(case.tb_size, case.code_rate)
    except (ValueError, StopIteration) as exc:
        raise UnsupportedConfiguration("Unsupported CB geometry") from exc
    if geo.k_prime % 8:
        raise UnsupportedConfiguration("Stage 1 requires byte-aligned code block information")
    if geo.count > 4 or case.g > 131072 or case.g * case.samples > 2_000_000:
        raise UnsupportedConfiguration("Exceeds the Stage 1 CB, G or aggregate sample budget")
    if case.fault == "pre_crc_bit_flip" and geo.count != 1:
        raise UnsupportedConfiguration("Pre-CRC injection is qualified for one CB")
    return case
