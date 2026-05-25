# baseline-v2 — AGC/EQ-augmented "Hey Bubba" model

Second acceptable model. Materially better than v1 in real-world use —
same architecture (BC-ResNet τ=3, 39k params, 686 KB INT8) but trained
with augmentations that simulate device-side DSP processing.

## Eval (test set, FP32)
- AUC: 0.9999
- FRR: 0.027 @ FAR: 0.30/h
- Operating threshold: 0.59 (FP32) / ~0.45 (INT8 runtime)
- Test set: 1347 positives, 10037 negatives, 3.35 h

## What's different from v1
| | baseline-v1 | baseline-v2 |
|--|--:|--:|
| Compression (AGC sim) prob | 0 | **0.65** |
| Random shelf EQ prob       | 0 | **0.50** |
| Bandpass prob              | 0.20 | **0.55** |
| Codec roundtrip prob       | 0.15 | **0.40** |
| Per-epoch training time    | 127s | 285s (CPU-bound) |
| FP32 operating threshold   | 0.26 | 0.59 (wider score range = more robust) |
| INT8 runtime threshold     | 0.35 | 0.45 |

Training data and architecture unchanged from v1.

## On-device behavior (Mac mic, ONNX runtime)
- v1: works at threshold 0.35 with raw mic, brittle to AGC-style processing
- v2: holds up cleanly through both raw and processed audio paths

## Files
- `exports/wakeword_audio.int8.onnx` — what's in the v0.9.0-agc APK (686 KB)
- `exports/wakeword_audio.onnx` + `.data` — FP32 reference
- `exports/wakeword_mel.int8.onnx` — mel-in variant
- `exports/mel_filterbank.npy` — filterbank for parity testing
- `exports/metadata.json`
- `checkpoints/best.pt` — PyTorch state dict (EMA weights)
- `checkpoints/eval_test.json` — full ROC sweep
- `whistle-demo-v0.9.0-agc.apk` — Android test app shipping this model
