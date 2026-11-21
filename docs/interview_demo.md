# Interview demo

Use the Streamlit dashboard in **REPLAY** with the verified PUSCH Live record selected. It displays the saved H20 and provider evidence. A live API call is unnecessary during the interview.

| Time | Show | Say |
| --- | --- | --- |
| 0:00–0:35 | Overview | Introduce software test development for GPU PHY. Point to the pinned SDK, H20, 66 GPU tests and 37 AI host tests. |
| 0:35–1:10 | Test Coverage & Risk | Open the MCS × PRB risk factor. Show its requirement ID and the recorded scenario transition from NOT_COVERED to COVERED. |
| 1:10–1:55 | AI Test Generation | Open the verified Live record. Show MCS 4, 32 PRBs and seed 320601, the validated JSON, zero bit errors and CRC 0. Switch briefly to the first blocked Live record to show why rejected proposals never reached GPU execution. |
| 1:55–2:35 | Defect Analysis | Open RM Bit Flip. Explain how independent RM still catches the flip even when final CRC is zero. Then open Noise Variance Injection and distinguish the LLR evidence from its inconclusive BLER impact. |
| 2:35–3:15 | Regression Results | Show the seven cases, the selection reasons and the original outcomes. Use BG-R067 to explain how a reproducible helper discrepancy remains a potential finding. |
| 3:15–3:45 | Overview | Point to the BLER intervals and explain hardware coverage, finite samples and the independently replayable record. |

[Overview screenshot](screenshots/overview.png) · [Coverage screenshot](screenshots/coverage.png) · [Generation screenshot](screenshots/generation.png)

## 60–90 second introduction

> I built NR-AI-QA to explore how a software test engineer can qualify a GPU PHY component when a passing CRC tells only part of the story. The project runs NVIDIA pyAerial and cuPHY on an H20 and uses independent checks for selected Release 17 LDPC, rate-matching, and PUSCH behavior. One controlled rate-matching flip was caught at the intermediate output even though the decoder recovered the payload and the final CRC passed.
>
> I added an AI workflow for coverage and test selection. A Python risk score highlights gaps, and the model proposes structured cases. The existing validator and oracle own execution and verdicts. In a real qwen3.8-flash run, the model proposed a PUSCH case with MCS 4 and 32 PRBs. It passed on the H20, and the workflow actually ran seven selected regressions. The final baseline passed 66 GPU tests and 37 AI host tests.
>
> The evidence covers selected profiles. I retain BG-R067 as a helper boundary discrepancy with unverified final transport-block impact, and use confidence intervals to show what our AWGN samples support.
