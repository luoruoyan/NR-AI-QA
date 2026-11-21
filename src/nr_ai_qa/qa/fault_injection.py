"""Only the test adapter invokes these hooks. Arrays are copied before alteration."""

import numpy as np

BIT_INDEX = 7


def flip_bit(bits: np.ndarray, *, column: int | None = None) -> tuple[np.ndarray, dict]:
    index = BIT_INDEX if column is None else (BIT_INDEX, column)
    altered = bits.copy()
    before = int(altered[index])
    altered[index] = 1 - before
    return altered, {"offset": BIT_INDEX, "column": column, "before": before,
                     "after": int(altered[index])}
