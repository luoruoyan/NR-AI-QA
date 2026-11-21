"""Independent Release 17 calculations for the tested profiles."""

import math
from dataclasses import dataclass

import numpy as np

# TS 38.214 Table 5.1.3.1-1, target rate numerator with denominator 1024.
MCS_RATES = (
    120, 157, 193, 251, 308, 379, 449, 526, 602, 679,
    340, 378, 434, 490, 553, 616, 658, 438, 466, 517, 567, 616, 666,
    719, 772, 822, 873, 910, 948,
)
SMALL_TBS = (
    24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144,
    152, 160, 168, 176, 184, 192, 208, 224, 240, 256, 272, 288, 304, 320,
    336, 352, 368, 384, 408, 432, 456, 480, 504, 528, 552, 576, 608, 640,
    672, 704, 736, 768, 808, 848, 888, 928, 984, 1032, 1064, 1128, 1160,
    1192, 1224, 1256, 1288, 1320, 1352, 1416, 1480, 1544, 1608, 1672,
    1736, 1800, 1864, 1928, 2024, 2088, 2152, 2216, 2280, 2408, 2472,
    2536, 2600, 2664, 2728, 2792, 2856, 2976, 3104, 3240, 3368, 3496,
    3624, 3752, 3824,
)
LIFTING = sorted(i * (2 ** e) for i in (2, 3, 5, 7, 9, 11, 13, 15)
                 for e in range(8) if i * (2 ** e) <= 384)
RV_FACTORS = {1: (0, 17, 33, 56), 2: (0, 13, 25, 43)}


@dataclass(frozen=True)
class Geometry:
    bg: int
    tb_crc: int
    cb_crc: int
    count: int
    z: int
    k_prime: int
    k: int
    n: int

    @property
    def filler(self) -> int:
        return self.k - self.k_prime


def mcs_parameters(mcs: int) -> tuple[int, float]:
    q = 2 if mcs < 10 else 4 if mcs < 17 else 6
    return q, MCS_RATES[mcs] / 1024


def scheduled_tbs(g: int, rate: float) -> int:
    """38.214 §5.1.3.2 with G=Qm*N_RE for our single-layer allocation."""
    info = g * rate
    if info <= 3824:
        quantum = 2 ** max(3, math.floor(math.log2(info)) - 6)
        target = max(24, quantum * math.floor(info / quantum))
        return next(value for value in SMALL_TBS if value >= target)
    quantum = 2 ** (math.floor(math.log2(info - 24)) - 5)
    target = max(3840, quantum * math.floor((info - 24) / quantum + 0.5))
    if rate <= 0.25:
        count = math.ceil((target + 24) / 3816)
    elif target > 8424:
        count = math.ceil((target + 24) / 8424)
    else:
        count = 1
    return 8 * count * math.ceil((target + 24) / (8 * count)) - 24


def base_graph(a: int, rate: float) -> int:
    return 2 if a <= 292 or (a <= 3824 and rate <= 0.67) or rate <= 0.25 else 1


def geometry(a: int, rate: float, *, bg: int | None = None) -> Geometry:
    bg = base_graph(a, rate) if bg is None else bg
    tb_crc = 16 if a <= 3824 else 24
    b = a + tb_crc
    limit = 8448 if bg == 1 else 3840
    cb_crc = 0 if b <= limit else 24
    count = math.ceil(b / (limit - cb_crc))
    if b % count:
        raise ValueError("Non-integral K-prime is outside the Stage 1 profile")
    k_prime = b // count + cb_crc
    if bg == 1:
        kb = 22
    else:
        kb = 10 if b > 640 else 9 if b > 560 else 8 if b > 192 else 6
    z = next(value for value in LIFTING if value * kb >= k_prime)
    return Geometry(bg, tb_crc, cb_crc, count, z, k_prime,
                    z * (22 if bg == 1 else 10), z * (66 if bg == 1 else 50))


def crc_bits(bits: np.ndarray, width: int, polynomial: int) -> np.ndarray:
    remainder = 0
    mask = (1 << width) - 1
    for bit in bits:
        feedback = (remainder >> (width - 1)) ^ int(bit)
        remainder = (remainder << 1) & mask
        if feedback:
            remainder ^= polynomial
    return np.array([(remainder >> i) & 1 for i in range(width - 1, -1, -1)], dtype=np.uint8)


def segmented_reference(payload: np.ndarray, geo: Geometry) -> np.ndarray:
    # The pinned cuPHY CRC bridge reverses storage bits to present each input
    # payload byte in MSB-first protocol order.
    bits = np.unpackbits(payload, bitorder="big")
    tb_poly = 0x1021 if geo.tb_crc == 16 else 0x864CFB
    tb = np.concatenate((bits, crc_bits(bits, geo.tb_crc, tb_poly)))
    blocks = np.split(tb, geo.count)
    if geo.cb_crc:
        blocks = [np.concatenate((cb, crc_bits(cb, 24, 0x800063))) for cb in blocks]
    return np.column_stack(blocks)


def matched_indices(geo: Geometry, g: int, q: int, rv: int) -> tuple[np.ndarray, ...]:
    symbols, extra = divmod(g // q, geo.count)
    k0 = RV_FACTORS[geo.bg][rv] * geo.z
    circular = (np.arange(geo.n) + k0) % geo.n
    # Filler is marked by its position after puncturing, never by the stored bit.
    first_null = geo.k_prime - 2 * geo.z
    end_null = geo.k - 2 * geo.z
    circular = circular[(circular < first_null) | (circular >= end_null)]
    return tuple(np.resize(circular, q * (symbols + (cb >= geo.count - extra)))
                 .reshape(q, -1).T.ravel() for cb in range(geo.count))


def matched_reference(encoded: np.ndarray, indices: tuple[np.ndarray, ...]) -> np.ndarray:
    return np.concatenate([encoded[positions, cb] for cb, positions in enumerate(indices)])


def modulation_reference(bits: np.ndarray, q: int) -> np.ndarray:
    signed = 1 - 2 * bits.astype(np.float32).reshape(-1, q)
    result = signed[:, :2]
    if q == 4:
        result = result * (2 - signed[:, 2:4])
    elif q == 6:
        result = result * (4 - signed[:, 2:4] * (2 - signed[:, 4:6]))
    return (result[:, 0] + 1j * result[:, 1]) / math.sqrt({2: 2, 4: 10, 6: 42}[q])


def first_difference(expected: np.ndarray, observed: np.ndarray) -> dict | None:
    if expected.shape != observed.shape:
        return {"expected_shape": list(expected.shape), "observed_shape": list(observed.shape)}
    different = np.flatnonzero(expected.ravel() != observed.ravel())
    if not different.size:
        return None
    index = int(different[0])
    return {"flat_index": index, "coordinate": [int(i) for i in np.unravel_index(index, expected.shape)],
            "expected": int(expected.ravel()[index]), "observed": int(observed.ravel()[index]),
            "count": int(different.size)}


def intervals(errors: list[int], size: int, seed: int) -> dict:
    n = len(errors)
    observed = np.asarray(errors)
    tb_errors = int(np.count_nonzero(observed))
    p = tb_errors / n
    z = 1.959963984540054
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    bler_ci = [0.0 if tb_errors == 0 else max(0.0, center - radius),
               1.0 if tb_errors == n else min(1.0, center + radius)]
    sample = np.random.default_rng(seed).integers(0, n, size=(2000, n))
    ber_ci = np.quantile(observed[sample].mean(axis=1) / size, [0.025, 0.975]).tolist()
    return {"samples": n, "tb_errors": tb_errors, "bit_errors": int(observed.sum()),
            "bits_tested": n * size, "ber": float(observed.mean() / size), "bler": p,
            "bler_ci95_wilson": bler_ci, "ber_ci95_tb_bootstrap": ber_ci,
            "ber_ci_degenerate": ber_ci[0] == ber_ci[1],
            "ber_conservative_upper_from_bler_ci95": bler_ci[1]}
