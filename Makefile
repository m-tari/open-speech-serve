DOCKER := ./scripts/docker.sh

CELL ?= configs/cells/fw_turbo_c1.yaml
SWEEP ?= configs/sweeps/v1.yaml
N ?= 25
RESULTS_DIR ?= results/v2_comparison
GPU_TELEMETRY_DIR ?= results/gpu_telemetry
SKIP_PREP ?= 0
SKIP_TRT_PREP ?= 0

.PHONY: build prepare smoke mock server ttfs stop-server analyze plot \
	gpu-build gpu-prepare gpu-cell gpu-sweep gpu-server gpu-ttfs stop-gpu-server \
	build-vllm vllm-up vllm-cell vllm-sweep stop-vllm \
	sglang-up sglang-cell sglang-sweep stop-sglang \
	triton-build triton-prepare triton-up triton-cell triton-sweep stop-triton \
	stop-backends v2-comparison gpu-telemetry plot-gpu-telemetry

build-vllm:
	$(DOCKER) build-vllm

build:
	$(DOCKER) build-cpu

prepare:
	$(DOCKER) run-cpu prepare

mock:
	$(DOCKER) run-cpu cell configs/cells/smoke_mock_c1.yaml

smoke: prepare
	$(DOCKER) run-cpu cell configs/cells/smoke_mock_c1.yaml
	$(DOCKER) run-cpu cell configs/cells/smoke_fw_tiny_c1.yaml

server: build
	$(DOCKER) up-server

stop-server:
	$(DOCKER) stop-server

ttfs:
	$(DOCKER) run-cpu ttfs --url ws://127.0.0.1:8000/v1/stream --n 3

analyze:
	$(DOCKER) run-cpu analyze

plot:
	plot --results-dir results --out-dir results/published

# GPU in-process baselines
gpu-build:
	$(DOCKER) build-gpu

gpu-prepare:
	$(DOCKER) run-gpu prepare --with-librispeech --n $(N)

gpu-cell:
	$(DOCKER) run-gpu cell $(CELL)

gpu-sweep:
	$(DOCKER) run-gpu sweep $(SWEEP)

gpu-server: gpu-build
	$(DOCKER) up-gpu-server

stop-gpu-server:
	$(DOCKER) stop-gpu-server

gpu-ttfs:
	$(DOCKER) run-gpu ttfs --url ws://127.0.0.1:8000/v1/stream --n 3

# Remote serving backends — one GPU at a time
vllm-up:
	$(DOCKER) build-bench
	$(DOCKER) up-vllm

vllm-cell:
	$(DOCKER) cell-vllm $(CELL)

vllm-sweep:
	$(DOCKER) sweep-vllm configs/sweeps/v2_vllm.yaml

stop-vllm:
	$(DOCKER) stop-vllm

sglang-up:
	$(DOCKER) build-bench
	$(DOCKER) up-sglang

sglang-cell:
	$(DOCKER) cell-sglang $(CELL)

sglang-sweep:
	$(DOCKER) sweep-sglang configs/sweeps/v2_sglang.yaml

stop-sglang:
	$(DOCKER) stop-sglang

triton-build:
	$(DOCKER) build-triton

triton-prepare:
	$(DOCKER) triton-prepare

triton-up:
	$(DOCKER) build-bench
	$(DOCKER) up-triton

triton-cell:
	$(DOCKER) cell-triton $(CELL)

triton-sweep:
	$(DOCKER) sweep-triton configs/sweeps/v2_trtllm.yaml

stop-triton:
	$(DOCKER) stop-triton

stop-backends:
	$(DOCKER) stop-backends

v2-comparison:
	N=$(N) RESULTS_DIR=$(RESULTS_DIR) \
		SKIP_PREP=$(SKIP_PREP) SKIP_TRT_PREP=$(SKIP_TRT_PREP) \
		./scripts/run_v2_comparison.sh

# GPU util vs concurrency (HF + vLLM + TensorRT-LLM, nvidia-smi during timed passes)
gpu-telemetry:
	N=$(N) RESULTS_DIR=$(GPU_TELEMETRY_DIR) \
		SKIP_PREP=$(SKIP_PREP) SKIP_TRT_PREP=$(SKIP_TRT_PREP) \
		./scripts/run_gpu_telemetry.sh

plot-gpu-telemetry:
	$(DOCKER) run-bench -- plot-gpu-telemetry \
		--results-dir $(GPU_TELEMETRY_DIR) \
		--out-dir $(GPU_TELEMETRY_DIR)/published
