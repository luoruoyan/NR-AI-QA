"""Curated regression families, required cases and change-specific priorities."""

GOLDEN = {"ldpc": "GOLD-QPSK", "pusch": "P01-GOLDEN"}
RULES = {
    "ldpc_processing": (
        "BG-A3824", "FAULT-CRC-FLIP", "RV-BG2-REPEAT", "RV-BG1-3", "FAULT-RM-FLIP",
    ),
    "configuration": (
        "P02-CONSISTENCY", "P04-DMRS", "P05-DMRS-FAULT", "P04-MCS", "P04-PRB", "P04-SISO",
    ),
    "noise_variance": (
        "P02-CONSISTENCY", "P05-NOISE-FAULT", "P03-MID", "P04-DMRS", "P05-DMRS-FAULT", "P04-PRB",
    ),
}
PRIORITY_MODULES = {
    "base_graph": ("BG-A3824", "BG-A3840"),
    "segmentation": ("SEG-BG2-FOUR", "SEG-BG1-TWO"),
    "rate_matching": ("RV-BG2-REPEAT", "FAULT-RM-FLIP"),
    "rv": ("RV-BG1-3", "RV-BG2-3"),
    "crc_bit_order": ("FAULT-CRC-FLIP", "GOLD-BG1"),
    "mcs": ("P04-MCS",),
    "prb": ("P04-PRB",),
    "dmrs": ("P04-DMRS", "P05-DMRS-FAULT"),
    "noise_variance": ("P05-NOISE-FAULT", "P03-MID"),
    "receivers": ("P02-CONSISTENCY", "P04-SISO"),
    "mcs_prb": ("P04-MCS", "P04-PRB"),
}
