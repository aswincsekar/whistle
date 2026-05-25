# whistle - "Hey Bubba" wake-word detector

End-to-end pipeline for a tiny on-device wake-word model. Trains on synthesized
positives + downloaded negatives + augmentation, exports to **ONNX** and
**TFLite** with INT8 quantization, runs on iOS, Android, desktop.

```
audio (16 kHz, mono) ─► log-mel (40 × ~120) ─► BC-ResNet ─► sigmoid(P("Hey Bubba"))
```

| Component             | Default                                    |
|-----------------------|--------------------------------------------|
| Front-end             | Log-mel, 40 mels, 25 ms / 10 ms, 1.2 s win |
| Model                 | BC-ResNet (tau=2)                          |
| Params (tau=2)        | ~30 k                                      |
| Loss                  | BCE with hard-negative mining + EMA        |
| Window / hop (stream) | 1.2 s / 100 ms                             |
| Export targets        | ONNX (opset 18) + TFLite (INT8)            |

---

## 1. Setup

```bash
make venv                   # creates .venv with Python 3.11
source .venv/bin/activate
make install                # core deps (training, ONNX export)
make install-tflite         # only if you also need TFLite export
```

## 2. Smoke test (no downloads, ~30 s)

```bash
python -m whistle.cli smoke
```

This synthesizes a handful of "Hey Bubba" positives via macOS `say`
(falls back to tone-bursts if unavailable), generates noise negatives,
trains 2 epochs, evaluates, exports ONNX, and runs the streaming detector.
Useful as a regression test after edits.

## 3. Full training run

### 3a. Generate positives

```bash
# macOS (no install needed - uses built-in `say` voices):
make synth   # ~400 utterances across ~18 voices

# or cross-platform with Piper:
pip install piper-tts
# download voices: https://github.com/rhasspy/piper/blob/master/VOICES.md
python -m whistle.cli synth \
    --piper-voice voices/en_US-libritts-high.onnx \
    --piper-voice voices/en_GB-alan-medium.onnx \
    --count 800
```

You can also drop your own field recordings into `data/positives/` -
anything `.wav/.flac/.ogg/.mp3` is picked up.

### 3b. Download negatives + noise + RIRs

```bash
make data
# Sources (configurable in src/whistle/data/download.py):
#   - Speech Commands v0.02 (Google) -> negatives/      ~2.4 GB
#   - MS-SNSD noise corpus (Microsoft) -> noise/        ~230 MB
#   - OpenSLR-26 simulated RIRs        -> rirs/         ~1.3 GB
```

Skip any with `--skip speech_commands`, etc. You can also drop any extra
audio into `data/negatives` / `data/noise` / `data/rirs` and it'll be used.

### 3c. Build manifests, train, evaluate

```bash
make manifests
make train                  # writes checkpoints/{best,last}.pt + runs/<ts>/
make eval                   # writes checkpoints/eval_test.json with FAR/FRR curve
```

Training reports per-epoch BCE loss, validation AUC, and FRR at 1% FPR.
Hard negatives get re-mined every 5 epochs after warm-up.

### 3d. Export

```bash
make export-onnx            # exports/wakeword_audio.onnx + .int8.onnx
make export-tflite          # exports/wakeword_audio.{fp32,int8}.tflite
```

Each export verifies PyTorch vs ONNX outputs match to ~1e-5.
`exports/metadata.json` holds everything the mobile runtime needs.

## 4. Local testing

```bash
# Offline (WAV file)
python -m whistle.cli stream --ckpt checkpoints/best.pt --wav my_test.wav

# Live mic
python -m whistle.cli stream --ckpt checkpoints/best.pt --mic --threshold 0.6
```

---

## Repository layout

```
configs/default.yaml          all hyperparameters
src/whistle/
  features.py                 log-mel front-end (torch, ONNX-exportable)
  models/bc_resnet.py         the classifier
  models/wakeword.py          audio-in composite (mel + classifier)
  data/
    synth.py                  TTS positives (say / piper)
    download.py               external corpora
    augment.py                noise mix, RIR, pitch/time, codec, gain
    dataset.py                PyTorch dataset + balanced sampler
    manifests.py              speaker-disjoint splits
  train.py                    trainer with EMA + hard-negative mining
  eval.py                     FAR/FRR/AUC at thresholds
  stream.py                   sliding-window detector
  export.py                   ONNX + TFLite + INT8 + verify
  smoke.py                    end-to-end smoke test
```

---

## Mobile integration

### Two integration shapes

`exports/` contains two graph variants - pick whichever fits your stack.

| File                       | Input                              | When to use                              |
|----------------------------|------------------------------------|------------------------------------------|
| `wakeword_audio.onnx`      | float32 PCM `(1, 19200)`           | One-file deploy; mobile does only audio capture. |
| `wakeword_mel.onnx`        | float32 log-mel `(1, 40, ~120)`    | Mobile already has a mel front-end (CoreML AudioFeaturePrint, NNAPI, …). Smaller graph. |

Both variants ship FP32 and dynamic INT8 versions. `metadata.json` describes
exact shapes, the mel-filterbank parameters, and the audio-pipeline expectations.

### Streaming loop (any platform)

```
1. Capture 16 kHz mono float32 PCM (or int16, scale by /32768).
2. Append into a ring buffer of size `input_samples` (19200 = 1.2 s).
3. Every `hop_seconds` (100 ms / 1600 samples):
     a. Run inference on the current window.
     b. p = sigmoid(logits[1] - logits[0])         # the positive logit
     c. Smooth p over the last 3 windows (mean).
     d. If smoothed p >= threshold and >= cooldown since last fire: WAKE.
```

The reference Python implementation lives in `src/whistle/stream.py` -
mirror its semantics exactly on device.

### iOS

Recommended runtime: **ONNX Runtime for iOS** (CocoaPods `onnxruntime-c` or
SwiftPM `onnxruntime-objc`). It's the lowest-friction path because no model
conversion is needed beyond what `make export-onnx` already produces.

```swift
let modelURL = Bundle.main.url(forResource: "wakeword_audio.int8", withExtension: "onnx")!
let session = try ORTSession(env: env, modelPath: modelURL.path, sessionOptions: nil)
let input = try ORTValue(tensorData: NSMutableData(bytes: pcm, length: pcm.count * 4),
                         elementType: .float, shape: [1, 19200])
let outputs = try session.run(withInputs: ["audio": input],
                              outputNames: ["logits"], runOptions: nil)
```

Alternative: convert `wakeword_audio.onnx` -> Core ML via
[`coremltools.convert`](https://apple.github.io/coremltools/). Coremltools
supports the front-end ops (rFFT was added in iOS 17). For older iOS, use the
`wakeword_mel.onnx` variant and feed mel features computed with
`MLAudioFeaturePrint`.

### Android

Recommended runtime: **TFLite** (`tensorflow-lite` + `tensorflow-lite-support`)
or **ONNX Runtime Android** (`com.microsoft.onnxruntime:onnxruntime-mobile`).

For TFLite:
```kotlin
val interpreter = Interpreter(loadModelFile("wakeword_audio.int8.tflite"))
val input = ByteBuffer.allocateDirect(4 * 19200).order(ByteOrder.nativeOrder())
pcm.forEach { input.putFloat(it) }
val output = Array(1) { FloatArray(2) }
interpreter.run(input, output)
val logits = output[0]
val prob = 1f / (1f + exp(-(logits[1] - logits[0])))
```

For audio capture use `AudioRecord` with `ENCODING_PCM_16BIT` at 16 kHz; divide
by 32768 to get float32 in `[-1, 1]`.

### Parity test before shipping

Sanity-check that the mobile pipeline produces the same logits as the desktop
ONNX model:

1. Save a `(1, 19200)` float32 PCM buffer from your mobile capture layer.
2. Run `wakeword_audio.onnx` on it in Python:
   ```python
   import numpy as np, onnxruntime as ort
   sess = ort.InferenceSession("exports/wakeword_audio.onnx")
   pcm = np.load("captured.npy")
   print(sess.run(["logits"], {"audio": pcm}))
   ```
3. Compare against the logits your mobile runtime produced. Should match
   to within ~1e-4 (FP32) or ~1e-2 (INT8).

If they diverge, the usual culprits are:
- Different audio scaling (int16 vs float32, missing `/32768`).
- Wrong sample rate (resampling on the mobile side that introduces aliasing).
- Padding/window misalignment.

---

## Tuning tips

| Symptom                           | Knob                                                   |
|-----------------------------------|--------------------------------------------------------|
| Too many false fires in cafes     | Bump `noise_prob` to 0.95, broaden `noise_snr_db` low end to -10. |
| Missing softly-spoken activations | Drop `gain_db` lower bound (-15), raise `positive_ratio` to 0.3. |
| Activations on TV/news            | Add long-form speech (LibriSpeech, Common Voice) into `data/negatives`. |
| Too big a model for embedded MCU  | `model.tau: 1`. ~7k params, ~0.3 MB INT8.              |
| Want closer to SOTA               | `model.tau: 6`, more positives (≥2000), more noise diversity. |

Operating point (the threshold + cooldown you ship) is set from the
`eval.target_far_per_hour` config — `make eval` prints the recommended
threshold for your target false-accept rate.

---

## Troubleshooting

* **`No TTS engine available`** during `make synth` — install `piper-tts`
  and pass `--piper-voice <model.onnx>`, or run on macOS where `say` is
  built in.
* **`UserWarning: pin_memory ... not supported on MPS`** — harmless;
  Apple Silicon training still works.
* **ONNX verify "MISMATCH"** — usually means the front-end you compiled
  into ONNX uses different STFT parameters than your runtime. Re-run
  `make export-onnx` after any change in `configs/default.yaml:features`.
* **Apple Silicon training stalls** — set `train.amp: false` in the
  config; AMP on MPS is still flaky in some torch versions.
