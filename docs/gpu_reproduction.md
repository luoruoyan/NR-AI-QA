# GPU reproduction

## Environment

The verified Linux x86-64 system was Ubuntu 22.04 with an NVIDIA H20 (SM 9.0), driver 580.126.09, Docker Engine and NVIDIA Container Toolkit. Aerial 26-2 provided CUDA 13.3, Python 3.12.3, CMake 3.28.3, Ninja 1.11.1 and the native SDK build dependencies. The image consumed roughly 42 GB on disk on the verified host. Allow additional space for the SDK, build and test evidence. Follow the upstream support and licence terms for your hardware and image access. [NVIDIA Getting Started](https://docs.nvidia.com/aerial/cuda-accelerated-ran/latest/pyaerial/getting_started.html) · [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

Run from the project directory on a configured GPU host with Git LFS installed. Docker must be available to the current user.

```bash
mkdir -p sdk environment/stage1-python reports/ai
git clone https://github.com/NVIDIA/aerial-cuda-accelerated-ran.git sdk/aerial-cuda-accelerated-ran
git -C sdk/aerial-cuda-accelerated-ran checkout --detach 4f65f97c1d5f701ce911f7dda8f1b1f3f0c7693c
git -C sdk/aerial-cuda-accelerated-ran submodule update --init --recursive
git -C sdk/aerial-cuda-accelerated-ran lfs pull

image=nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:26-2-cubb
docker pull --platform linux/amd64 "$image"
docker image inspect --platform linux/amd64 "$image" --format '{{.Id}}'
```

Expect `sha256:fe79b31fe1703d6dbca78c72fa4463b0192804c97fb45f97d8ce740513affa24`. Configure any required NGC authentication through Docker's normal credential mechanism. Inside the official image, build from the fixed SDK at the same mount path used by the test runner.

```bash
docker run --rm --platform linux/amd64 --runtime=nvidia --gpus all \
  --user "$(id -u):$(id -g)" --shm-size=1g \
  --mount "type=bind,src=$(pwd)/sdk/aerial-cuda-accelerated-ran,dst=/opt/nvidia/cuBB" \
  --mount "type=bind,src=$(pwd)/environment/stage1-python,dst=/qa-deps" \
  --mount "type=bind,src=$(pwd)/requirements-gpu.txt,dst=/qa-requirements.txt,readonly" \
  --workdir /opt/nvidia/cuBB --entrypoint /bin/bash "$image" -lc '
    set -e
    cmake --preset pyaerial-x86
    cmake --build --preset pyaerial-x86 --parallel 8 --target pyaerial_setup
    pyaerial/.venv/bin/python -m pip install --target /qa-deps -r /qa-requirements.txt
  '
```

The build may require outbound access for upstream dependencies. The test entry point checks the image digest and SDK commit, then runs with networking disabled and a read-only SDK mount. The verified QA path uses neither a real-time RAN launch nor GDRCopy devices.

## Native tests and baseline inputs

```bash
scripts/run_qa_container.sh -m pytest -q -p no:cacheprovider tests/ldpc tests/pusch
scripts/run_qa_container.sh -m nr_ai_qa.cli --suite ldpc --out reports/ldpc
scripts/run_qa_container.sh -m nr_ai_qa.cli --suite pusch --out reports/pusch
```

Run the reports independently. The historical LDPC profile returned exit code 1 for `BG-R067`; keep that recorded discrepancy and check for any other failures. The Live planner reads these two baseline reports and their per-case evidence. Exact link results can vary under the finite AWGN sample and native floating-point behavior. The published snapshot preserves the recorded outcomes.

## Host Live controller

```bash
python3 -m pip install --target environment/stage3-python -r environment/requirements-stage3.txt
export PYTHONPATH=src:environment/stage3-python
python3 -m pytest -q -p no:cacheprovider tests/ai
```

Set `LLM_BASE_URL`, `LLM_MODEL` and `LLM_API_KEY` in the server environment through your local secret handling. Verified Alibaba Cloud Model Studio model is `qwen3.8-flash`; use the OpenAI-compatible address for the API key's region. An SSH shell can use `read -rs -p 'API Key: ' LLM_API_KEY` followed by `export LLM_API_KEY` without echoing the credential. Then run a single scoped case through the dashboard or CLI.

```bash
python3 -m nr_ai_qa.ai_cli --suite pusch --mode live --max-cases 1 \
  --requirement NR-PUSCH-MCS-TBS --risk-topic PUSCH-MCS-PRB
```

The historic recording is visible in the portable dashboard with its exact seed and parameters. A new model call may propose different inputs. To run a fixed validated input, export the case JSON from Replay and use `scripts/run_qa_container.sh -m nr_ai_qa.cli --json` with a project-relative file below `configs/` or `reports/`, and a new `--out reports/...` directory.
