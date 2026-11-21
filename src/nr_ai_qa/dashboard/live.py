"""Submit one opt-in request to the existing Stage 3 CLI from the dashboard."""

import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from nr_ai_qa.dashboard.data import ROOT, project_workflow

ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")


def live_arguments(suite: str, requirement: str, risk_topic: str) -> list[str]:
    if suite not in {"ldpc", "pusch"} or not all(ID.fullmatch(v) for v in (requirement, risk_topic)):
        raise ValueError("Invalid dashboard request scope")
    return [sys.executable, "-m", "nr_ai_qa.ai_cli", "--suite", suite, "--mode", "live", "--max-cases", "1",
            "--requirement", requirement, "--risk-topic", risk_topic]


def availability(bind_address: str, root: Path = ROOT) -> str | None:
    if bind_address not in {"127.0.0.1", "::1", "localhost"}:
        return "Run the server on loopback before enabling local Live execution."
    if not shutil.which("docker") or not (root / "sdk/aerial-cuda-accelerated-ran").is_dir():
        return "Live execution needs Docker and the pinned Aerial SDK."
    if not all((root / f"reports/{suite}/results.json").is_file() for suite in ("ldpc", "pusch")):
        return "Generate the local LDPC and PUSCH baseline reports first."
    if not any(os.environ.get(key) for key in ("LLM_API_KEY", "DASHSCOPE_API_KEY", "OPENAI_API_KEY")):
        return "Set the model credential in the server environment."
    if not any(os.environ.get(key) for key in ("LLM_MODEL", "OPENAI_MODEL")):
        return "Set the model ID in the server environment."
    return None


def run_live(suite, requirement, risk_topic, *, bind_address, root=ROOT) -> tuple[int, dict | None]:
    """Run the fixed CLI once; expose only the allowlisted evidence projection."""
    blocked = availability(bind_address, root)
    if blocked:
        raise ValueError(blocked)

    import fcntl  # Live uses the Linux CUDA host. Replay stays portable.

    args = live_arguments(suite, requirement, risk_topic)
    lock_path = root / "reports" / ".dashboard-live.lock"
    lock_path.parent.mkdir(exist_ok=True)
    with lock_path.open("a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("A Live dashboard workflow is already running.") from None
        env = os.environ.copy()
        search = [str(root / "src"), str(root / "environment/stage3-python")]
        if env.get("PYTHONPATH"):
            search.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(search)
        process = subprocess.run(args, cwd=root, env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, check=False)
        match = re.search(r"(?m)^(\d{8}T\d{6}Z-live-(?:ldpc|pusch)-[a-f0-9]{8}) ", process.stdout)
        if match is None:
            return process.returncode, None
        path = root / "reports" / "ai" / match.group(1)
        return process.returncode, project_workflow(path)
