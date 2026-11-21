#!/usr/bin/env bash
set -euo pipefail
project=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
sdk="$project/sdk/aerial-cuda-accelerated-ran"
image=nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:26-2-cubb
digest=sha256:fe79b31fe1703d6dbca78c72fa4463b0192804c97fb45f97d8ce740513affa24
commit=4f65f97c1d5f701ce911f7dda8f1b1f3f0c7693c
mkdir -p "$project/logs" "$project/reports/ldpc"
log=$(mktemp "$project/logs/stage1-container-XXXXXXXX.log")
printf 'Log: %s\n' "$log"
{
    observed=$(docker image inspect --platform linux/amd64 "$image" --format '{{.Id}}')
    [[ "$observed" == "$digest" ]] || { echo "Unexpected image: $observed" >&2; exit 1; }
    [[ "$(git -C "$sdk" rev-parse HEAD)" == "$commit" ]] || { echo 'Unexpected SDK commit' >&2; exit 1; }
    git -C "$sdk" diff --exit-code HEAD
    [[ -x "$sdk/pyaerial/.venv/bin/python" && -d "$project/environment/stage1-python" ]] || {
        echo 'Missing existing pyAerial venv or project Python dependencies' >&2; exit 1;
    }
    if (( $# == 0 )); then set -- -m pytest -q -p no:cacheprovider tests/ldpc; fi
    printf 'Python arguments:'; printf ' %q' "$@"; printf '\n'
    docker_options=()
    if [[ -n "${NRQA_CONTAINER_CIDFILE:-}" ]]; then
        docker_options+=(--cidfile "$NRQA_CONTAINER_CIDFILE")
    fi
    docker run --rm --pull=never --network none --runtime=nvidia --gpus all "${docker_options[@]}" \
        --user "$(id -u):$(id -g)" --shm-size=1g \
        --env CUDA_LAUNCH_BLOCKING=1 --env CUDA_MODULE_LOADING=EAGER \
        --env "NRQA_SDK_COMMIT=$commit" --env "NRQA_IMAGE_DIGEST=$digest" \
        --env PYTHONPATH=/workspace/src:/qa-deps --env PYTHONDONTWRITEBYTECODE=1 \
        --mount "type=bind,src=$sdk,dst=/opt/nvidia/cuBB,readonly" \
        --mount "type=bind,src=$project/src,dst=/workspace/src,readonly" \
        --mount "type=bind,src=$project/specs,dst=/workspace/specs,readonly" \
        --mount "type=bind,src=$project/configs,dst=/workspace/configs,readonly" \
        --mount "type=bind,src=$project/tests/ldpc,dst=/workspace/tests/ldpc,readonly" \
        --mount "type=bind,src=$project/tests/pusch,dst=/workspace/tests/pusch,readonly" \
        --mount "type=bind,src=$project/environment/stage1-python,dst=/qa-deps,readonly" \
        --mount "type=bind,src=$project/reports,dst=/workspace/reports" \
        --workdir /workspace --entrypoint /bin/bash "$image" -euc '
        nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv
        python=/opt/nvidia/cuBB/pyaerial/.venv/bin/python
        extension=$("$python" -B -c "from aerial.pycuphy import _pycuphy; print(_pycuphy.__file__)")
        [[ "$extension" == /opt/nvidia/cuBB/pyaerial/src/aerial/pycuphy/_pycuphy*.so ]]
        links=$(ldd "$extension")
        printf "%s\n" "$links" | grep -E "libcuphy|libcuda|not found"
        [[ "$links" == *"libcuphy.so => /opt/nvidia/cuBB/build-pyaerial-x86/"* &&
           "$links" == *"libcuphy_ldpc.so => /opt/nvidia/cuBB/build-pyaerial-x86/"* &&
           "$links" != *"=> not found"* ]]
        exec "$python" -B "$@"
    ' nrqa "$@"
} 2>&1 | tee "$log"
