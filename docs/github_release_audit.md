# GitHub release audit

**Local candidate only.** No repository was created, no history was rewritten, and nothing was pushed. The project root had no Git repository, commits, or local Git history to scan. Its independent NVIDIA SDK checkout at `4f65f97c1d5f701ce911f7dda8f1b1f3f0c7693c` was clean. The checkout is shallow with one commit available locally. That checkout and its Git data are outside the release allowlist.

## Findings

- A path-and-line-only scan of 556 project text files (7.55 MB), including historical reports, logs, configs and source, found no candidate credentials or cloud instance IDs matching the configured signatures. It checked API and NGC keys, literal authorization headers, bearer or JWT tokens, SSH private-key blocks, proxy secrets and authenticated URLs. Values were never copied to the audit output.
- Host paths and private or loopback address matches appeared in the original worktree. The infrastructure matches included the historical Docker and network logs, plus the intentional localhost check in the model client. Raw logs, deployment notes and full local report trees are excluded. The release uses a field-allowlisted projection of recorded evidence.
- The earlier credential posted in chat is outside this repository scan. The subsequent Live verification read its key through local terminal input. Ensure the previously posted key is revoked in the provider console. No submitted key value was included in the candidate.
- SDK storage was about 12 GB, original reports were about 60 MB, and the dependency directories are local. The ignore file excludes these directories, local secrets, wheels, references, binary evidence and raw IQ. The explicit manifest is the publication boundary.
- This is a signature-based text review, with binary and vendor trees excluded from release. Run a hosted secret scanner against the Git history after creating the eventual repository history. If a new exposure is found, hold publication and revoke or rotate the affected credential.

## Publication boundary

The 88-file local candidate follows [the candidate manifest](github_release_manifest.txt). The snapshot retains relative provenance and selected report values. The screenshot views run against the same sanitized snapshot. The manifest omits the standalone BG parity probe, full specifications, SDK, image layers, browser dependencies and original operational reports. [Licence review and pending decisions](license_review.md)

Before approval, confirm the project licence, third-party rights and any employer disclosure requirements. Choose eventual repository visibility and owner, then review the first Git commit tree and secret scan before publishing.
