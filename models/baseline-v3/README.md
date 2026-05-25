# baseline-v3 — UrbanSound8K fine-tune

Third acceptable model. Fine-tuned from `baseline-v2` for 20 epochs (LR 5e-4)
with UrbanSound8K added to the noise augmentation pool. Same architecture
(BC-ResNet τ=3, 39k params, 686 KB INT8) — only training data/noise changed.

## Eval (test set, FP32)
| | v1 baseline | v2 (AGC/EQ) | **v3 (+UrbanSound8K)** |
|--|--:|--:|--:|
| AUC                  | 0.9999 | 0.9999 | **1.0000** |
| FRR @ FAR=0.30/h     | 0.0074 | 0.027  | **0.003**  |
| FP32 op threshold    | 0.26   | 0.59   | **0.47**   |
| INT8 runtime threshold | 0.35 | 0.45   | **0.35**   |
| Model size (INT8)    | 686 KB | 686 KB | **686 KB** |

The big win is FRR — 9× fewer missed wake words at the same FAR vs v2.
Same APK size, same inference cost.

## What changed from v2
- Initialized from `models/baseline-v2/checkpoints/best.pt` (kept all of v2's
  AGC/EQ learning).
- 20 epochs fine-tune at LR 5e-4 (vs 3e-3 for cold start).
- Noise pool expanded from ~128 MS-SNSD clips to **~8,860 clips** (added
  UrbanSound8K's 8,732 urban sounds: car_horn, engine_idling, jackhammer,
  siren, drilling, street_music, children_playing, dog_bark, AC, gunshot).
- Hard-negative mining ran 4 times (epochs 4, 9, 14, 19), pulling in UrbanSound-
  flavored confusables.

## Files
- `exports/wakeword_audio.int8.onnx` — what's in the v0.10.0-urbansound APK
- `exports/wakeword_audio.onnx` + `.data` — FP32 reference
- `exports/wakeword_mel.int8.onnx` — mel-in variant
- `exports/mel_filterbank.npy` + `metadata.json`
- `checkpoints/best.pt` — PyTorch state dict, EMA weights
- `checkpoints/eval_test.json` — full ROC sweep
- `whistle-demo-v0.10.0-urbansound.apk` — Android test app
