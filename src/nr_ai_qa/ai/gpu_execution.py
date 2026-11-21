"""Run only allowlisted QA CLI operations with deadline and provenance checks."""

import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

from nr_ai_qa.ai.defect_analyzer import outcome_category
from nr_ai_qa.qa.coverage_registry import evidence_is_gpu
from nr_ai_qa.qa.reporting import json_text
from nr_ai_qa.qa.runner import ROOT, validate_case


def execution_class(row: dict) -> str:
    if row["status"] == "EXPECTED_DETECTION":
        return "EXPECTED_DETECTION"
    category = outcome_category(row)
    if category == "CONTROLLED_INJECTION":
        return ("SDK_RUNTIME_ERROR" if row.get("error_class") == "SDK_ERROR" else
                "INCONCLUSIVE" if row["status"] == "INCONCLUSIVE" else "ORACLE_MISMATCH")
    return {"POTENTIAL_SDK_DEFECT": "ORACLE_MISMATCH",
            "STATISTICAL_UNCERTAINTY": "INCONCLUSIVE"}.get(category, category)


class GpuExecutor:
    backend = "AERIAL_GPU_CONTAINER"

    def __init__(self, registry):
        self.registry = registry
        self.root = registry.root

    def _relative(self, path: Path) -> str:
        target = path.resolve()
        expected = (self.root / "reports/ai").resolve()
        if not target.is_relative_to(expected):
            raise ValueError("GPU evidence must live under reports/ai")
        return target.relative_to(self.root).as_posix()

    def generated(self, candidate, directory, deadline):
        case = validate_case(candidate["case"]).model_dump(mode="json")
        output = directory / case["case_id"]
        source = output / "input.json"
        self._relative(source)
        output.mkdir(parents=True, exist_ok=False)
        source.write_text(json_text(case) + "\n", encoding="utf-8")
        return self._run(["--json", self._relative(source)], output, [case["case_id"]], deadline,
                         expected_cases={case["case_id"]: case})

    def regression(self, suite, case_ids, directory, deadline):
        profiles = self.registry.profiles[suite]
        if not case_ids or len(case_ids) != len(set(case_ids)) or not set(case_ids) <= profiles.keys():
            raise ValueError("Only unique registered regression cases can execute")
        arguments = ["--suite", suite]
        for case_id in case_ids:
            arguments += ["--case", case_id]
        expected = {cid: validate_case(profiles[cid]).model_dump(mode="json") for cid in case_ids}
        return self._run(arguments, directory, case_ids, deadline, expected_cases=expected)

    def _run(self, arguments, directory, case_ids, deadline, expected_cases):
        started = time.monotonic()
        output = self._relative(directory)
        command = ["scripts/run_qa_container.sh", "-m", "nr_ai_qa.cli", *arguments, "--out", output]
        directory.mkdir(parents=True, exist_ok=True)
        log, cidfile = directory / "container.log", directory / "container.cid"
        env = {name: os.environ[name] for name in ("PATH", "HOME", "LANG", "LC_ALL") if name in os.environ}
        env["NRQA_CONTAINER_CIDFILE"] = str(cidfile)
        ran = {"argv": command, "log": self._relative(log), "returncode": None}
        timeout = max(0, deadline - time.monotonic())
        try:
            if not timeout:
                raise TimeoutError
            with log.open("w", encoding="utf-8") as stream:
                with subprocess.Popen(command, cwd=self.root, env=env, stdout=stream,
                                      stderr=subprocess.STDOUT, start_new_session=True) as process:
                    try:
                        ran["returncode"] = process.wait(timeout=timeout)
                    except (subprocess.TimeoutExpired, KeyboardInterrupt):
                        self._cancel(process, cidfile, stream, env)
                        raise
        except (TimeoutError, subprocess.TimeoutExpired):
            return {"status": "ENVIRONMENT_ERROR", "reason": "GPU wall time budget exhausted",
                    "records": [], "commands": [ran]}
        except OSError:
            return {"status": "ENVIRONMENT_ERROR", "reason": "Unable to start QA container process",
                    "records": [], "commands": [ran]}

        ran["wall_seconds"] = round(time.monotonic() - started, 6)
        try:
            rows = json.loads((directory / "results.json").read_text(encoding="utf-8"))
            if [r["case_id"] for r in rows] != case_ids:
                raise ValueError("Unexpected or absent executed case IDs")
            if ran["returncode"] not in {0, 1, 2}:
                raise ValueError("Unexpected runner process exit code")
            for row in rows:
                evidence = (directory / row["evidence_path"]).resolve()
                if not evidence.is_relative_to(directory.resolve()):
                    raise ValueError("Result evidence outside the run directory")
                if json.loads(evidence.read_text(encoding="utf-8")) != row:
                    raise ValueError("Result and evidence disagree")
                if row["test_parameters"] != expected_cases[row["case_id"]]:
                    raise ValueError("GPU input differs from validated input")
                classification = execution_class(row)
                native = evidence_is_gpu(row)
                if row["status"] in {"PASS", "EXPECTED_DETECTION", "POTENTIAL_SDK_DEFECT"} and not native:
                    raise ValueError("Successful or suspect test lacks qualified native GPU evidence")
                row.update(execution_class=classification, runner_log=ran["log"],
                           evidence_path=self._relative(evidence), real_gpu=native)
        except (OSError, TypeError, ValueError, KeyError) as exc:
            return {"status": "ENVIRONMENT_ERROR", "reason": f"GPU evidence check failed ({type(exc).__name__})",
                    "records": [], "commands": [ran]}
        errors = [r["execution_class"] for r in rows if r["execution_class"] not in
                  {"PASS", "EXPECTED_DETECTION", "INCONCLUSIVE"}]
        return {"status": "COMPLETED_WITH_FINDINGS" if errors else "COMPLETED", "records": rows,
                "commands": [ran], "expected_case_ids": case_ids}

    @staticmethod
    def _cancel(process, cidfile, stream, env):
        try:
            cid = cidfile.read_text(encoding="utf-8").strip()
            if re.fullmatch(r"[a-f0-9]{64}", cid):
                subprocess.run(["docker", "kill", cid], check=False, stdout=stream,
                               stderr=subprocess.STDOUT, env=env, timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            pass
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
