"""Bounded JSON input for the notebook's single-layer, no-UCI receiver profile."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

Identifier = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")]


class DmrsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    scrambling_id: int = Field(default=41, ge=0, le=65535)
    mapping_type: Literal["A"] = "A"
    config_type: Literal[1] = 1
    symbol: Literal[2] = 2
    additional_positions: Literal[0] = 0
    cdm_groups_without_data: Literal[2] = 2


class PuschCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    kind: Literal["pusch"] = "pusch"
    case_id: Identifier
    requirement_ids: list[Identifier] = Field(min_length=1, max_length=12)
    test_type: Literal[
        "golden", "consistency", "awgn", "config_contract", "iq_contract", "dmrs_fault", "noise_fault",
    ] = "golden"
    mcs: int = Field(ge=0, le=28)
    tb_size: int = Field(ge=24, le=262144, description="Transport block size in bits")
    num_prbs: int = Field(ge=1, le=273)
    num_rx_ant: int = Field(default=2, ge=1, le=2)
    num_symbols: int = Field(default=12, ge=1, le=12)
    dmrs_config: DmrsConfig = Field(default_factory=DmrsConfig)
    channel_type: Literal["awgn", "tdl", "cdl"] = "awgn"
    snr_db: float = Field(ge=-30, le=60, allow_inf_nan=False)
    seed: int = Field(ge=0, le=2**32 - 1)
    num_transport_blocks: int = Field(default=1, ge=1, le=96)
    risk_tags: list[Identifier] = Field(default_factory=list, max_length=12)
