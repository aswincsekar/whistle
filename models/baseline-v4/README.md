# baseline-v4 — openWakeWord embedding + small classifier

Fundamentally different architecture from v1/v2/v3. Instead of training a
mel-spectrogram BC-ResNet from scratch, we use openWakeWord v0.5.1's
pretrained `speech_embedding` model (originally Google's TFHub
`speech_embedding/1`) as a frozen feature extractor and train a small
Conv1D classifier on top of the 96-dim embedding stream.

Why this matters: the front-end was self-supervised on ~10k hours of speech
plus phoneme/speaker aux tasks. Its features are already invariant to
speaker, loudness, microphone, and ambient noise — exactly the things our
mel-feature baselines had to learn from scratch and never fully mastered.

## Eval (test set, FP32)
- AUC: 1.0000
- FRR: 0.010 @ FAR: 0.37/h
- Op threshold: 0.46

## Comparison
| | v1 | v2 AGC | v3 noise | **v4 OWW** |
|--|--:|--:|--:|--:|
| Front-end           | log-mel (40, 25/10 ms) | same | same | **mel32 + speech_embedding** |
| Trained params      | 39 k | 39 k | 39 k | **111 k (classifier only)** |
| Frozen params       | 0 | 0 | 0 | **~330 k (embedding) + melspec** |
| AUC                 | 0.9999 | 0.9999 | 1.0000 | **1.0000** |
| FRR @ FAR ≈ 0.3/h   | 0.7% | 2.7% | 0.3% | 1.0% |
| Mobile bundle (INT8)| 686 KB | 686 KB | 686 KB | **2.5 MB (3 ONNX)** |

## Architecture
```
audio (16 kHz, 1.96 s = 31360 samples)
   │
   ▼  melspectrogram.onnx   (frozen, ~1.0 MB)
mel features (T=196, 32 bins)
   │
   ▼  slide 76-frame windows, stride 8 → N=15 windows
   │
   ▼  embedding_model.onnx  (frozen, ~1.3 MB)
embeddings (15, 96)
   │
   ▼  classifier.int8.onnx  (TRAINED HERE, ~120 KB)
logits (1, 2)
```

## Files
- `exports/melspectrogram.onnx` — frozen openWakeWord front-end
- `exports/embedding_model.onnx` — frozen openWakeWord speech embedding
- `exports/classifier.onnx` + `.data` — FP32 trained classifier
- `exports/classifier.int8.onnx` — static QDQ INT8 classifier
- `exports/metadata.json` — chain spec
- `checkpoints/best.pt` — PyTorch state dict (classifier only)
- `checkpoints/eval_test.json` — full ROC sweep
- `whistle-demo-v0.13.0-owww.apk` — Android test app
