COMPOSE_GPU := docker compose -f docker-compose.yml -f docker-compose.gpu.yml

CELL ?= configs/cells/fw_turbo_l4_c1.yaml
SWEEP ?= configs/sweeps/v1.yaml
N ?= 25

.PHONY: build prepare smoke mock server ttfs analyze \
	gpu-build gpu-prepare gpu-cell gpu-sweep gpu-server gpu-ttfs

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
