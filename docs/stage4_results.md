# Stage 4 local acceptance

The portfolio adds five Streamlit views, a sanitized historical dataset, a GPU reproduction guide and interview material. The Dashboard defaults to Replay and presents recorded evidence. The Live start action is available on a configured local host and delegates to the existing workflow.

## Verified work

- The initial baseline passed 66 GPU tests and 37 AI host tests. The GPU host was the recorded NVIDIA H20 with SDK commit `4f65f97c1d5f701ce911f7dda8f1b1f3f0c7693c`.
- The Replay dataset is a deterministic projection of historical reports. It preserves coverage state, risk factors, validation rejections, Oracle status, statistical uncertainty and original relative provenance.
- The verified PUSCH Live record has three model calls, one passing generated GPU case and seven exercised regressions. The UI labels this as historical evidence in Replay.
- The Live click and scope routing are tested with a double. Stage 4 does not submit another model or GPU workflow through that button. Stage 3 supplies the real Live data presented in the dashboard.
- The release audit and candidate allowlist precede owner approval. A final licence is pending rights review.

## Verification results

| Check | Observed result |
| --- | --- |
| Baseline before dashboard work | 66 GPU tests and 37 AI host tests passed. |
| Final GPU regression | 66 passed in 13.21 s on the pinned H20 container. |
| Host AI and Dashboard regression | 50 passed, comprising the existing 37 AI tests and 13 dashboard tests. |
| Isolated portfolio candidate | 11 dashboard tests passed. Two comparisons needing original private reports skipped as designed. |
| Streamlit HTTP and browser | The loopback server passed its health check. Chromium rendered all five pages from the candidate with API credentials unset and non-loopback requests blocked. |
| Historical projection | The portable JSON equals the deterministic projection from the selected original reports. The archive retains the rejected Live attempt and the successful Live PUSCH case and regressions. |
| Live control | The dashboard tested the confirmation, fixed CLI handoff and suite/requirement/risk scope with a double. No additional paid Live run was started from the dashboard. |
| SDK integrity | The pinned SDK commit and worktree remained clean. |

The baseline host and GPU logs, final host and GPU logs, and browser transcripts are retained locally under `logs/`. Final GPU evidence is `logs/stage1-container-VuTyfdD4.log`; final host and portable browser logs are `logs/stage4-final-host.log` and `logs/stage4-portable-browser-final.log`. The portfolio disables Streamlit file watching to avoid the observed host inotify limit. A fresh server passed browser checks with that configuration. The screenshots below were captured with Chromium from the local candidate after all widgets and charts rendered.

## Publication status

The explicit allowlist stages only project sources, dependency pins, selected docs and sanitized evidence. Original reports and host files remain local. GitHub publication and final licence selection remain pending owner approval and the rights items noted in the licence review.

## Delivery

[README preview](../README.md) · [Dashboard architecture](architecture.md) · [GPU reproduction](gpu_reproduction.md) · [Interview demo](interview_demo.md) · [Known limitations](known_limitations.md) · [Release audit](github_release_audit.md) · [Release manifest](github_release_manifest.txt)

## Replay views

[Overview](screenshots/overview.png) · [Coverage and risk](screenshots/coverage.png) · [AI generation](screenshots/generation.png) · [Defect analysis](screenshots/defect.png) · [Regressions](screenshots/regression.png)
