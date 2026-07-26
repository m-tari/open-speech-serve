# RunPod (no nested Docker)

RunPod starts a Pod as a container, so nested Docker (`make vllm-up`,
`make v2-comparison`) usually does not work inside it. Use a custom template
for one backend, then run the benchmark client in another shell in that same
Pod. On a real Docker GPU VM (not nested), use plain `docker` via
`make v2-comparison` — Compose is not required.

Use one NVIDIA L4, at least 50 GB container disk, and a persistent network
volume mounted at `/workspace`. Do not expose backend ports unless a client
outside the Pod needs them.

## vLLM

Create a custom Pod template with image `vllm/vllm-openai:v0.25.1`. Keep the
container alive with an interactive shell, connect in two terminals, and run:

```bash
# terminal 1
cd /workspace
git clone https://github.com/m-tari/open-speech-serve.git
cd open-speech-serve
./deploy/runpod/launch_vllm.sh

# terminal 2
cd /workspace/open-speech-serve
./deploy/runpod/setup_client.sh
prepare --with-librispeech --n 25
cell configs/cells/vllm_turbo_l4_c8_serialized.yaml
cell configs/cells/vllm_turbo_l4_c8_concurrent.yaml
```

The vLLM cells default to `http://localhost:8000`.

## SGLang

Create a custom Pod template with image `lmsysorg/sglang:v0.5.15`. Then:

```bash
# terminal 1
cd /workspace/open-speech-serve
./deploy/runpod/launch_sglang.sh

# terminal 2
cd /workspace/open-speech-serve
./deploy/runpod/setup_client.sh
prepare --with-librispeech --n 25
cell configs/cells/sglang_turbo_l4_c8_serialized.yaml
cell configs/cells/sglang_turbo_l4_c8_concurrent.yaml
```

The SGLang cells default to `http://localhost:30000`. Its Whisper route is
experimental; check WER before interpreting performance.

## TensorRT-LLM/Triton

Build `docker/Dockerfile.triton` on a Docker machine, push it to a registry,
and use that image in a RunPod custom template. RunPod cannot build the image
inside a Pod.

On the target L4, prepare architecture-specific engines:

```bash
cd /workspace/open-speech-serve
ARTIFACT_DIR=/workspace/artifacts/triton \
  ./scripts/triton/build_whisper_engines.sh
```

Start Triton:

```bash
python3 /opt/TensorRT-LLM/tensorrt_llm/triton_backend/scripts/launch_triton_server.py \
  --world_size 1 \
  --model_repo /workspace/artifacts/triton/model_repo \
  --tensorrt_llm_model_name tensorrt_llm,whisper_bls \
  --multimodal_gpu0_cuda_mem_pool_bytes 300000000
```

In another terminal:

```bash
cd /workspace/open-speech-serve
pip install -e '.[triton]'
prepare --with-librispeech --n 25
cell configs/cells/trtllm_turbo_l4_c8_serialized.yaml
cell configs/cells/trtllm_turbo_l4_c8_concurrent.yaml
```

The Triton cells default to `localhost:8001` (gRPC).

## Persistence and transfer

Keep model caches, data, results, and TensorRT engines under `/workspace`:

```bash
export HF_HOME=/workspace/.cache/huggingface
export XDG_CACHE_HOME=/workspace/.cache
tar -czf /workspace/open-speech-results.tar.gz \
  -C /workspace/open-speech-serve results
```

Download the archive through RunPod's file browser, `scp`, or `rsync`.
