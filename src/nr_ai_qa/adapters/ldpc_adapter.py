"""Own one GPU stream and the pinned Aerial modules for serial frame execution."""

from dataclasses import dataclass

import cupy as cp
import numpy as np
from aerial.phy5g.algorithms import ModulationMapper
from aerial.phy5g.ldpc import (
    CrcChecker, CrcEncoder, LdpcDecoder, LdpcDeRateMatch, LdpcEncoder, LdpcRateMatch, get_mcs,
)
from aerial.phy5g.ldpc.util import get_base_graph
from aerial.util.cuda import CudaStream

from nr_ai_qa.qa.fault_injection import flip_bit
from nr_ai_qa.schemas.ldpc_case import LdpcCase


@dataclass
class Frame:
    payload: np.ndarray
    blocks: np.ndarray
    encoded: np.ndarray
    sdk_helper_bg: int
    matched: np.ndarray | None = None
    matched_before_fault: np.ndarray | None = None
    rm_lengths: tuple[int, ...] = ()
    symbols: np.ndarray | None = None
    llr_shape: tuple[int, ...] = ()
    finite_llrs: bool = False
    derated_shape: tuple[int, ...] = ()
    decoded: np.ndarray | None = None
    received: np.ndarray | None = None
    tb_crc: int | None = None
    cb_crc: tuple[int, ...] = ()
    fault: dict | None = None


class PyAerialLdpcAdapter:
    def __init__(self):
        cp.cuda.Device(0).use()
        self.stream = CudaStream()
        self.crc = CrcEncoder(max_num_tbs=1, cuda_stream=self.stream)
        self.encoder = LdpcEncoder(puncturing=True, max_num_code_blocks=4, cuda_stream=self.stream)
        self.rate_matcher = LdpcRateMatch(enable_scrambling=False, max_num_tbs=1,
                                          max_num_code_blocks=4, cuda_stream=self.stream)
        self.derate_matcher = LdpcDeRateMatch(enable_scrambling=False, cuda_stream=self.stream)
        self.decoder = LdpcDecoder(num_iterations=10, cuda_stream=self.stream)
        self.checker = CrcChecker(cuda_stream=self.stream)
        self.mappers = {}

    def close(self):
        self.stream.synchronize()

    def execute(self, case: LdpcCase, rng: np.random.Generator, *, inject: bool = True) -> Frame:
        payload = rng.integers(0, 256, size=case.tb_size // 8, dtype=np.uint8)
        common = {"tb_sizes": [case.tb_size], "code_rates": [case.code_rate]}
        rv = {"redundancy_versions": [case.rv]}
        blocks = self.crc.encode(tb_inputs=[payload], **common)
        encoded = self.encoder.encode(code_blocks=blocks, **common, **rv)
        self.stream.synchronize()
        frame = Frame(payload, blocks[0], encoded[0], get_base_graph(case.tb_size, case.code_rate))
        if case.test_type == "bg_probe":
            return frame

        q, scaled_rate = get_mcs(case.mcs, table_idx=1)
        if (q, scaled_rate / 1024) != (case.mod_order, case.code_rate):
            raise RuntimeError("SDK get_mcs disagrees with the validated allocation")
        air = {"mod_orders": [q], "num_layers": [1], "cinits": [0]}
        matched = self.rate_matcher.rate_match(coded_blocks=encoded, rate_match_lens=[case.g],
                                              **common, **rv, **air)[0]
        self.stream.synchronize()
        frame.rm_lengths = tuple(self.rate_matcher.pycuphy_ldpc_rate_match.get_num_rm_bits())
        frame.matched = frame.matched_before_fault = matched
        if inject and case.fault == "rm_bit_flip":
            frame.matched, change = flip_bit(matched)
            frame.fault = {"location": "after_rate_matching", **change}

        # The mapper has no CudaStream argument. Use the shared stream for its CuPy work.
        # Do not wrap SDK calls which already enter the same non-reentrant stream.
        with self.stream:
            if q not in self.mappers:
                self.mappers[q] = ModulationMapper(q)
            mapper = self.mappers[q]
            symbols = mapper.map(cp.asarray(frame.matched))
            frame.symbols = symbols.get()
            variance = 10 ** (-case.esno_db / 10) if case.esno_db is not None else 0.1
            if case.channel_type == "awgn":
                noise = rng.standard_normal((2, symbols.size), dtype=np.float32)
                noise = (noise[0] + 1j * noise[1]) * np.float32(np.sqrt(variance / 2))
                received = symbols + cp.asarray(noise)
            else:
                received = symbols
            llr = mapper.demap(received, np.float32(1 / variance))
            llr = llr.T.reshape(-1, 1).get()
        self.stream.synchronize()
        frame.llr_shape = llr.shape
        frame.finite_llrs = bool(np.isfinite(llr).all())
        if llr.shape != (case.g, 1) or not frame.finite_llrs:
            raise RuntimeError("Invalid demapper output length or non-finite LLR")
        derated = self.derate_matcher.derate_match(input_llrs=[llr], rate_match_lengths=[case.g],
                                                   ndis=[1], **common, **rv, **air)
        frame.derated_shape = derated[0].shape
        decoded = self.decoder.decode(input_llrs=derated, rate_match_lengths=[case.g], **common, **rv)
        self.stream.synchronize()
        frame.decoded = decoded[0]
        if inject and case.fault == "pre_crc_bit_flip":
            decoded[0], change = flip_bit(decoded[0], column=0)
            frame.fault = {"location": "after_ldpc_decode_before_crc", **change}
        received, tb_crc = self.checker.check_crc(input_bits=decoded, **common)
        self.stream.synchronize()
        frame.received = received[0]
        frame.tb_crc = int(tb_crc[0].item())
        frame.cb_crc = tuple(map(int, self.checker.cb_crcs[0]))
        return frame
