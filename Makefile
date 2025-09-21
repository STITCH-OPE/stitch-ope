IMAGE ?= stitch-gpu
DOCKERFILE ?= Dockerfile

build:
	docker build -f $(DOCKERFILE) -t $(IMAGE) .

sanity:
	docker run --rm -t --gpus all \
	  -e MUJOCO_GL=egl \
	  -v $$PWD:/workspace -w /workspace \
	  $(IMAGE) python3 opelab/sanity.py --steps 5

shell:
	docker run --rm -it --gpus all --user $(shell id -u):$(shell id -g) \
	  -e MUJOCO_GL=egl -v $$PWD:/workspace -w /workspace \
	  $(IMAGE) bash

ZIP_URL ?= https://drive.google.com/file/d/1jPmqwmBSkkIuSoIFFpiRe8YvNh0ul2R6/view?usp=sharing
STAMP   ?= .cache/download.stamp

.PHONY: download
download:
	docker run --rm -t --gpus all --user $(shell id -u):$(shell id -g) \
	  -e MUJOCO_GL=egl -e ZIP_LOCAL=$(ZIP_LOCAL) -e ZIP_URL=$(ZIP_URL) -e ZIP_ID=$(ZIP_ID) \
	  -v $$PWD:/workspace -w /workspace \
	  $(IMAGE) bash -lc 'bash scripts/download_assets.sh'

.PHONY: refresh
refresh:
	docker run --rm -t --gpus all --user $(shell id -u):$(shell id -g) \
	  -e MUJOCO_GL=egl \
	  -e ZIP_URL=$(ZIP_URL) \
	  -e ZIP_LOCAL=$(ZIP_LOCAL) \
	  -e STAMP=$(STAMP) \
	  -e REFRESH=1 \
	  -v $$PWD:/workspace -w /workspace \
	  $(IMAGE) bash -lc 'bash scripts/download_assets.sh'

.PHONY: clean
clean:
	@echo "[clean] removing downloaded assets (models/policies/dataset) …"
	@set -e; \
	for d in \
	  opelab/examples/d4rl/models \
	  opelab/examples/d4rl/policy \
	  opelab/examples/gym/models  \
	  opelab/examples/gym/policy  \
	  opelab/examples/diffusion_policy/policy \
	  dataset \
	; do \
	  if [ -d "$$d" ]; then \
	    find "$$d" -mindepth 1 -maxdepth 1 -exec rm -rf {} +; \
	  fi; \
	done; \
	rm -f .cache/download.stamp || true
	@echo "[clean] done."
