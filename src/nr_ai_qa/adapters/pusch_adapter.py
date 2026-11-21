"""Generate frequency-domain input and call the pinned NVIDIA GPU pipelines."""

from copy import copy
from dataclasses import dataclass
from typing import Literal

import cupy as cp
import numpy as np

from aerial.phy5g.config import (
    AerialPdschTxConfig, AerialPuschRxConfig, PdschConfig, PdschCwConfig, PdschUeConfig,
    PuschConfig, PuschUeConfig,
)
from aerial.phy5g.ldpc import get_mcs, get_tb_size
from aerial.phy5g.pdsch import PdschTxPipelineFactory
from aerial.phy5g.pusch import PuschRxPipelineFactory, SeparablePuschRxPipelineFactory
from aerial.pycuphy.types import PuschLdpcKernelLaunch
from aerial.util.cuda import CudaStream

from nr_ai_qa.qa.oracle import mcs_parameters
from nr_ai_qa.schemas.pusch_case import PuschCase

SUBCARRIERS, SLOT_SYMBOLS, RX_ANTENNAS = 3276, 14, 2
START_SYMBOL = 2


class ReceiverContractError(ValueError):
    """A caller supplied IQ that cannot be passed to the native receiver."""


def validate_rx_iq(iq, num_rx_ant: int = RX_ANTENNAS):
    shape = (SUBCARRIERS, SLOT_SYMBOLS, num_rx_ant)
    if iq.shape != shape or iq.dtype != np.complex64:
        raise ReceiverContractError(
            f"IQ must be complex64 with shape {shape}, got {iq.dtype} {iq.shape}"
        )


@dataclass
class Frame:
    slot: int
    payload: np.ndarray
    iq: cp.ndarray
    data_re_power: float
    noise_variance: float
    data_re_count: int
    num_rx_ant: int


@dataclass
class Reception:
    crc: np.ndarray
    payloads: list[np.ndarray]
    intermediate: dict[str, np.ndarray]


class _BiasedNoiseInput:
    """Delegate to the real equalizer, modifying only its pre-EQ noise input."""

    def __init__(self, equalizer, stream, offset_db: float):
        self.equalizer = equalizer
        self.stream = stream
        self.offset_db = np.float32(offset_db)
        self.applied_db = None

    def equalize(self, *, noise_var_pre_eq, **kwargs):
        with self.stream:
            adjusted = noise_var_pre_eq + self.offset_db
            self.applied_db = adjusted.get()
        return self.equalizer.equalize(noise_var_pre_eq=adjusted, **kwargs)


class PyAerialPuschAdapter:
    def __init__(self):
        self.stream = CudaStream()
        self.tx = PdschTxPipelineFactory().create(
            AerialPdschTxConfig(cell_id=41, num_tx_ant=1), self.stream
        )
        self.receivers = {}

    def close(self):
        try:
            self.stream.synchronize()
        finally:
            self.receivers.clear()
            self.tx = None
            self.stream = None

    @staticmethod
    def dmrs_symbols(case: PuschCase) -> list[int]:
        return [int(i == case.dmrs_config.symbol) for i in range(SLOT_SYMBOLS)]

    def configs(self, case: PuschCase, *, rx_dmrs_id: int | None = None):
        dmrs = case.dmrs_config
        dmrs_symbols = self.dmrs_symbols(case)
        order, rate = get_mcs(case.mcs, 1)
        ref_order, ref_rate = mcs_parameters(case.mcs)
        sdk_tbs = get_tb_size(
            mod_order=order, code_rate=rate, dmrs_syms=dmrs_symbols,
            num_prbs=case.num_prbs, start_sym=START_SYMBOL,
            num_symbols=case.num_symbols, num_layers=1,
        )
        if (order, rate / 1024, sdk_tbs) != (ref_order, ref_rate, case.tb_size):
            raise RuntimeError(
                f"SDK MCS/TBS {(order, rate, sdk_tbs)} disagrees with independent profile"
            )

        cw = dict(mcs_table=0, mcs_index=case.mcs, code_rate=int(rate * 10),
                  mod_order=order, rv=0)
        ue = dict(scid=0, data_scid=0, layers=1, dmrs_ports=1, rnti=1234)
        allocation = dict(
            num_dmrs_cdm_grps_no_data=dmrs.cdm_groups_without_data,
            start_prb=0, num_prbs=case.num_prbs, dmrs_syms=dmrs_symbols,
            start_sym=START_SYMBOL, num_symbols=case.num_symbols,
        )
        tx_config = PdschConfig(
            ue_configs=[PdschUeConfig(
                cw_configs=[PdschCwConfig(**cw)], dmrs_scrm_id=dmrs.scrambling_id,
                precoding_matrix=None, **ue,
            )], **allocation,
        )
        rx_config = PuschConfig(
            ue_configs=[PuschUeConfig(tb_size=case.tb_size // 8, ndi=1, **cw, **ue)],
            dmrs_scrm_id=dmrs.scrambling_id if rx_dmrs_id is None else rx_dmrs_id,
            dmrs_max_len=1, dmrs_add_ln_pos=dmrs.additional_positions, **allocation,
        )
        return tx_config, rx_config

    def prepare(self, case: PuschCase, index: int, tx_config) -> Frame:
        """Unity gain per branch and independent complex AWGN on the FFT grid."""
        rng = np.random.default_rng(np.random.SeedSequence([case.seed, index]))
        payload = rng.integers(0, 256, size=case.tb_size // 8, dtype=np.uint8)
        slot = index % 20
        with self.stream:
            tb = cp.array(payload, dtype=cp.uint8, order="F")
        tx_grid = self.tx(slot=slot, tb_inputs=[tb], config=[tx_config])
        if tx_grid.shape != (SUBCARRIERS, SLOT_SYMBOLS, 1) or tx_grid.dtype != cp.complex64:
            raise RuntimeError(f"Unexpected TX grid {tx_grid.shape} {tx_grid.dtype}")
        data_sc = slice(0, 12 * case.num_prbs)
        data_symbols = [i for i in range(START_SYMBOL, START_SYMBOL + case.num_symbols)
                        if i != case.dmrs_config.symbol]
        with self.stream:
            data_grid = tx_grid[data_sc, data_symbols, 0]
            data_re_count = int(cp.count_nonzero(data_grid).get())
            signal_power = float(cp.mean(cp.abs(data_grid)**2).get())
            if not np.isfinite(signal_power) or signal_power <= 0:
                raise RuntimeError(f"Invalid TX data-RE power {signal_power}")
            variance = signal_power * 10**(-case.snr_db / 10)
            noise = rng.standard_normal((SUBCARRIERS, SLOT_SYMBOLS, case.num_rx_ant, 2),
                                        dtype=np.float32)
            complex_noise = (noise[..., 0] + 1j * noise[..., 1]) * np.float32(np.sqrt(variance / 2))
            iq = cp.array(complex_noise, dtype=cp.complex64, order="F")
            allocated = slice(START_SYMBOL, START_SYMBOL + case.num_symbols)
            iq[data_sc, allocated, :] += tx_grid[data_sc, allocated, :]
        self.stream.synchronize()
        return Frame(slot, payload, iq, signal_power, variance, data_re_count, case.num_rx_ant)

    def receiver(self, path: Literal["fused", "separable"], num_rx_ant: int):
        key = (path, num_rx_ant)
        if key not in self.receivers:
            factory = {"fused": PuschRxPipelineFactory,
                       "separable": SeparablePuschRxPipelineFactory}[path]
            config = AerialPuschRxConfig(
                cell_id=41, num_rx_ant=num_rx_ant, enable_pusch_tdi=1, eq_coeff_algo=1,
                ldpc_kernel_launch=PuschLdpcKernelLaunch.PUSCH_RX_LDPC_STREAM_SEQUENTIAL,
            )
            self.receivers[key] = factory().create(config, self.stream)
        return self.receivers[key]

    def receive(self, frame: Frame, rx_config, *, path: Literal["fused", "separable"] = "fused",
                capture: bool = False, noise_offset_db: float | None = None) -> Reception:
        validate_rx_iq(frame.iq, frame.num_rx_ant)
        with self.stream:
            iq = frame.iq.copy(order="F")
        pipeline = self.receiver(path, frame.num_rx_ant)
        noise_input = None
        if noise_offset_db is not None:
            if path != "separable" or not np.isfinite(noise_offset_db):
                raise ReceiverContractError("Noise offset requires the Separable path and finite dB")
            pipeline = copy(pipeline)
            noise_input = _BiasedNoiseInput(pipeline.channel_equalizer, self.stream, noise_offset_db)
            pipeline.channel_equalizer = noise_input
        crc, payloads = pipeline(slot=frame.slot, rx_slot=iq, config=[rx_config])
        self.stream.synchronize()
        intermediate = {}
        if capture and path == "separable":
            equalizer = noise_input.equalizer if noise_input is not None else pipeline.channel_equalizer
            order = rx_config.ue_configs[0].mod_order
            with self.stream:
                intermediate = {
                    "llrs": equalizer.llrs[0][:order].get(),
                    "eq_symbols": equalizer.eq_sym[0].get(),
                    "estimated_noise_db": pipeline.noise_intf_estimator.noise_var_pre_eq.get(),
                }
            intermediate["applied_noise_db"] = (noise_input.applied_db if noise_input is not None
                                                else intermediate["estimated_noise_db"])
        return Reception(np.asarray(crc).copy(), [np.asarray(tb).copy() for tb in payloads],
                         intermediate)
