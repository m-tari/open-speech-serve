COMPOSE_GPU := docker compose -f docker-compose.yml -f docker-compose.gpu.yml
COMPOSE_BACKENDS := docker compose -f docker-compose.backends.yml

CELL ?= configs/cells/fw_turbo_l4_c1.yaml
SWEEP ?= configs/sweeps/v1.yaml
N ?= 25
RESULTS_DIR ?= results/v2_comparison
SKIP_PREP ?= 0
SKIP_TRT_PREP ?= 0

.PHONY: build prepare smoke mock server ttfs analyze plot \
	gpu-build gpu-prepare gpu-cell gpu-sweep gpu-server gpu-ttfs \
	vllm-up vllm-cell vllm-sweep sglang-up sglang-cell sglang-sweep \
	triton-build triton-prepare triton-up triton-cell triton-sweep \
	v2-comparison

build:
	docker compose build

prepare:
	docker compose run --rm --entrypoint prepare server

mock:
	docker compose run --rm --entrypoint cell server configs/cells/smoke_mock_c1.yaml

smoke: prepare
	docker compose run --rm --entrypoint cell server configs/cells/smoke_mock_c1.yaml
	docker compose run --rm --entrypoint cell server configs/cells/smoke_fw_tiny_c1.yaml

server:
	docker compose up --build server

ttfs:
	docker compose run --rm --entrypoint ttfs server \
		--url ws://server:8000/v1/stream --n 3

analyze:
	docker compose run --rm --entrypoint analyze server

plot:
	plot --results-dir results --out-dir results/published

# GPU overlay (docker-compose.gpu.yml) — rented L4 path
gpu-build:
	$(COMPOSE_GPU) build

gpu-prepare:
	$(COMPOSE_GPU) run --rm --entrypoint prepare server --with-librispeech --n $(N)

gpu-cell:
	$(COMPOSE_GPU) run --rm --entrypoint cell server $(CELL)

gpu-sweep:
	$(COMPOSE_GPU) run --rm --entrypoint sweep server $(SWEEP)

gpu-server:
	$(COMPOSE_GPU) up --build server

gpu-ttfs:
	$(COMPOSE_GPU) run --rm --entrypoint ttfs server \
		--url ws://server:8000/v1/stream --n 3

# Remote serving backends — run one GPU profile at a time.
vllm-up:
	$(COMPOSE_BACKENDS) --profile vllm up vllm

vllm-cell:
	$(COMPOSE_BACKENDS) --profile vllm run --rm \
		-e OSS_BASE_URL=http://vllm:8000 bench-client cell $(CELL)

vllm-sweep:
	$(COMPOSE_BACKENDS) --profile vllm run --rm \
		-e OSS_BASE_URL=http://vllm:8000 bench-client \
		sweep configs/sweeps/v2_vllm.yaml

sglang-up:
	$(COMPOSE_BACKENDS) --profile sglang up sglang

sglang-cell:
	$(COMPOSE_BACKENDS) --profile sglang run --rm \
		-e OSS_BASE_URL=http://sglang:30000 bench-client cell $(CELL)

sglang-sweep:
	$(COMPOSE_BACKENDS) --profile sglang run --rm \
		-e OSS_BASE_URL=http://sglang:30000 bench-client \
		sweep configs/sweeps/v2_sglang.yaml

triton-build:
	$(COMPOSE_BACKENDS) --profile triton build triton

triton-prepare: triton-build
	$(COMPOSE_BACKENDS) --profile triton run --rm \
		--entrypoint bash triton \
		/workspace/open-speech-serve/scripts/triton/build_whisper_engines.sh

triton-up:
	$(COMPOSE_BACKENDS) --profile triton up triton

triton-cell:
	$(COMPOSE_BACKENDS) --profile triton run --rm \
		-e OSS_GRPC_URL=triton:8001 bench-client cell $(CELL)

triton-sweep:
	$(COMPOSE_BACKENDS) --profile triton run --rm \
		-e OSS_GRPC_URL=triton:8001 bench-client \
		sweep configs/sweeps/v2_trtllm.yaml

v2-comparison:
	N=$(N) RESULTS_DIR=$(RESULTS_DIR) \
		SKIP_PREP=$(SKIP_PREP) SKIP_TRT_PREP=$(SKIP_TRT_PREP) \
		./scripts/run_v2_comparison.sh
