# Known limitations

- **BG-R067.** At A=3824 and R=0.67, the native encoder selected BG2 using `0.67f`. The Python helper selected BG1 with `2/3`. Four bounded GPU probes supported the native boundary result. The broader decoder and final TB impact remains unverified. The recorded verdict is `POTENTIAL_SDK_DEFECT`.
- **Test scope.** The release covers selected LDPC/rate matching and single-layer PUSCH profiles on H20. Live generation was verified for PUSCH. LDPC exercised the offline model workflow and real GPU regression. TDL, UCI, HARQ, multi-user and full runtime operation are outside the verified scope.
- **Transmitter model.** The NVIDIA PUSCH example uses the SDK's PDSCH transmitter to emulate selected uplink signals. Fused and Separable comparison covers observable TB, CRC and shapes. Their internal tensors are unavailable for a direct pairwise comparison.
- **Statistics.** Link curves represent finite AWGN samples and the stated Es/N0 or per-branch data-RE SNR conventions. A 0/64 error count only bounds the corresponding rate. Noise injection changed observed LLRs, with BLER impact still inconclusive.
- **AI triage.** A suggested explanation is advisory. Controlled injections and statistical uncertainty remain separate from the possible SDK finding. No improvement percentage or product-level defect count was established.
- **Portability.** Replay has no GPU dependencies. Full tests were qualified on one H20 configuration. Public evidence contains selected summaries and original relative references. Raw IQ, NPZ and operational logs remain local.
