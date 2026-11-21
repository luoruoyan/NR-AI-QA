"""Write PUSCH reports and a BLER plot from measured payload errors."""

from pathlib import Path

from nr_ai_qa.qa.reporting import write_reports


def write_pusch_reports(root: Path, results: list) -> None:
    points = sorted(
        (r for r in results if r.test_parameters.get("test_type") == "awgn" and
         r.status == "PASS" and "bler_ci95_wilson" in r.actual_result),
        key=lambda r: r.actual_result["snr_db"],
    )
    attachments = {}
    if len(points) >= 2:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt

        snr = [r.actual_result["snr_db"] for r in points]
        bler = [r.actual_result["bler"] for r in points]
        ci = [r.actual_result["bler_ci95_wilson"] for r in points]
        samples = [r.actual_result["samples"] for r in points]
        fig, ax = plt.subplots(figsize=(6.5, 4.0), layout="constrained")
        ax.errorbar(snr, bler, yerr=[[p - bound[0] for p, bound in zip(bler, ci)],
                                   [bound[1] - p for p, bound in zip(bler, ci)]],
                    color="#064f84", marker="o", capsize=5, linestyle="none",
                    label="Observed BLER and 95% Wilson CI")
        ax.set(title="PUSCH payload BLER on NVIDIA H20",
               xlabel="Data-RE SNR per receive branch (dB)", ylabel="Payload block error rate",
               xticks=snr, ylim=(-0.08, 1.08))
        for x, p, count in zip(snr, bler, samples):
            ax.annotate(f"n={count}", (x, p), xytext=(8, -14 if p > 0.7 else 6),
                        textcoords="offset points", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="center right", frameon=False)
        root.mkdir(parents=True, exist_ok=True)
        fig.savefig(root / "bler.svg")
        plt.close(fig)
        attachments["Measured BLER"] = "bler.svg"

    counts = {
        status: sum(r.status == status for r in results)
        for status in ("PASS", "FAIL", "INCONCLUSIVE", "UNSUPPORTED", "EXPECTED_DETECTION",
                       "POTENTIAL_SDK_DEFECT", "CONFIGURATION_ERROR")
    }
    write_reports(
        root, results, title="NR-AI-QA PUSCH",
        extra_metrics=("crc_failures", "payload_errors", "crc_bler", "crc_bler_ci95_wilson",
                       "oracle_type", "reproduction_command"),
        attachments=attachments, summary=" · ".join(f"{k} {v}" for k, v in counts.items()),
    )
