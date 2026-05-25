PY ?= python
PIP ?= uv pip
CFG ?= configs/default.yaml

.PHONY: help venv install install-tflite synth data augment train eval export stream smoke clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

venv: ## Create .venv (Python 3.11)
	uv venv -p 3.11 .venv

install: ## Install runtime deps (training + ONNX export)
	$(PIP) install -e ".[tts,dev]"

install-tflite: ## Install TFLite export deps (heavy: TF + onnx2tf)
	$(PIP) install -e ".[tflite]"

synth: ## Generate "Hey Bubba" positives via TTS
	$(PY) -m whistle.cli synth --config $(CFG) --count 400

data: ## Download negatives + noise + RIRs (~few GB)
	$(PY) -m whistle.cli download --config $(CFG)

manifests: ## Rebuild train/val/test manifests from data/ contents
	$(PY) -m whistle.cli manifests --config $(CFG)

train: ## Train the wake-word model
	$(PY) -m whistle.cli train --config $(CFG)

eval: ## Evaluate latest checkpoint, write ROC/DET curves
	$(PY) -m whistle.cli eval --config $(CFG) --ckpt checkpoints/best.pt

export-onnx: ## Export ONNX (FP32 + INT8)
	$(PY) -m whistle.cli export --config $(CFG) --ckpt checkpoints/best.pt --format onnx

export-tflite: ## Export TFLite (FP32 + INT8). Needs `make install-tflite`.
	$(PY) -m whistle.cli export --config $(CFG) --ckpt checkpoints/best.pt --format tflite

stream-file: ## Run streaming detector on a WAV
	$(PY) -m whistle.cli stream --ckpt checkpoints/best.pt --wav $(WAV)

stream-mic: ## Live mic demo
	$(PY) -m whistle.cli stream --ckpt checkpoints/best.pt --mic

smoke: ## Tiny synthetic end-to-end run (no downloads)
	$(PY) -m whistle.cli smoke

clean: ## Remove caches and __pycache__
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
