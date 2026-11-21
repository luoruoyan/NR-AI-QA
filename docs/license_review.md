# Licence review

**Recommendation pending owner approval:** consider Apache-2.0 for the original NR-AI-QA source and original documentation. It is a permissive option that includes an express patent grant and notice requirements. The project owner should confirm copyright ownership, employer or internship obligations, and third-party attribution before adopting it. A final root `LICENSE` has not been assigned. [Apache-2.0 text](https://www.apache.org/licenses/LICENSE-2.0)

| Material | Finding and release treatment |
| --- | --- |
| NR-AI-QA dashboard, test harness and docs | Candidate material. Confirm the author can grant the selected licence. |
| NVIDIA Aerial SDK at the pinned commit | Its root licence specifies Apache-2.0 for the repository and calls out separate terms for materials obtained with the container and build. The portfolio links upstream and excludes the SDK tree, image, binaries and wheel artifacts. |
| ETSI / 3GPP publications | Confirm rights for any extracted normative table entries before publishing the affected source. The candidate links to the PDFs and excludes the PDFs and the standalone BG probe with directly transcribed parity-table rows. QA summaries and normative references remain attributable to their source. |
| Python dependencies | Streamlit 1.51.0 declares Apache-2.0, Pandas 2.3.3 declares BSD-3-Clause, and the installed Altair and NumPy distributions include their upstream BSD notices. The candidate lists dependencies for installation. No dependency wheel, browser binary or OS package is bundled. |
| Evidence and screenshots | Project-generated JSON summaries and genuine screenshots are included for owner review. The candidate excludes raw IQ, NPZ, credentials, operational logs and request IDs. |

Review any substantive code or tables adapted from upstream and retain applicable notices if redistribution is approved. The packaged Release 17 requirements are test mappings, and their presence does not assign a licence to the source publications.

[NVIDIA pinned licence](https://github.com/NVIDIA/aerial-cuda-accelerated-ran/blob/4f65f97c1d5f701ce911f7dda8f1b1f3f0c7693c/LICENSE) · [NVIDIA Third-Party Acknowledgements](https://docs.nvidia.com/aerial/cuda-accelerated-ran/latest/acknowledgements.html) · [ETSI IPR and copyright](https://www.etsi.org/resources/intellectual-property-rights/)
