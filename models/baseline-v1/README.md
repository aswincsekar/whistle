# baseline-v1 — first acceptable "Hey Bubba" model

This is the first model that actually fires reliably on a real phone (with
the source-toggle APK, v0.8.0-srctoggle). Save here as a reference point —
do **not** overwrite when later training runs land.

## Eval (test set)
- AUC: 0.9999
- FRR: 0.0074 @ FAR: 0.299/h
- Threshold: 0.26 (computed for FP32; INT8 deployment threshold ≈ 0.35)
- Test set: 1347 positives, 10037 negatives, 3.35 h of negative audio

## Architecture
- BC-ResNet, tau=3 (~39k params)
- Window: 1.2 s @ 16 kHz, 100 ms hop
- Features: log-mel, 40 mels, 25 ms / 10 ms STFT

## Training recipe
- 60 epochs, batch 256, AdamW + cosine, EMA shadow weights
- Hard-negative mining every 5 epochs after warmup
- Positives: ~6.7 k (cloud TTS: OpenAI + Gemini across 40+ voices, with
  4-way speed variants 0.7/0.85/1.15/1.3 and tempo-explicit prompts)
- Negatives: Speech Commands v0.02 + ~8 k same-voice cloud-TTS confusables
  ("Hey buddy", "Hey Bobby", "Hey Siri", "How are you", etc.)
- Augmentation: noise mix (MS-SNSD, prob 0.9, SNR -10 to +25 dB), RIR
  convolution (OpenSLR-26, prob 0.4), bandpass, mu-law codec, gain

## On-device behavior
- Works well on macOS mic via `python -m whistle.cli stream --mic` (FP32 .pt)
- Works on Android with **Raw / UNPROCESSED** mic source on Huawei (low
  threshold ≈ 0.07, energy gate ≈ -54 dBFS)
- Less reliable on Android with **Processed / VOICE_RECOGNITION** — the
  device's AGC + noise suppression compresses dynamics in a way the model
  wasn't trained on. Fix is in flight (v2 training run with AGC/EQ
  simulation in augmentation).

## Files
- `exports/wakeword_audio.int8.onnx` — what shipped in the APK (686 KB, INT8 QDQ)
- `exports/wakeword_audio.onnx` + `.data` — FP32 reference (use for parity tests)
- `exports/wakeword_mel.int8.onnx` — mel-in variant (152 KB; mobile computes mel)
- `exports/wakeword_mel.onnx` + `.data` — FP32 mel-in
- `exports/mel_filterbank.npy` — filterbank used in the front-end (for parity)
- `exports/metadata.json` — sample rate, window, mel params
- `checkpoints/best.pt` — PyTorch state dict (FP32, EMA shadow weights)
- `checkpoints/eval_test.json` — full ROC sweep
- `whistle-demo-v0.8.0.apk` — the Android test app that wrapped this model
