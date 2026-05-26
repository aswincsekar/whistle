# Training a wake-word model that actually works on your phone

*What I learned shipping "Hey Bubba" — five model architectures, two
training pipelines, three subtle deployment bugs, and the lesson I should
have absorbed from day one.*

---

If you ask Claude to "train a wake-word model for *Hey Bubba* and ship it
on Android," it can. The pipeline I built is on
[GitHub](https://github.com/aswincsekar/whistle). The final APK works.

But the path there was not a clean line. I trained five different models
over a few days. Four of them looked perfect on synthetic test metrics
and failed in some specific real-world way. The one that finally worked
required throwing out most of my architecture and using someone else's
training pipeline.

This post is the honest writeup — what I tried, what broke, what I would
do differently. The TL;DR is at the bottom.

---

## The problem

A wake-word detector is a tiny always-on binary classifier. It listens
to 16 kHz audio in 80-100 ms chunks and answers a single question:
*"Did the user just say the phrase?"*

The constraints make it interesting:

- **Small.** It runs continuously on a phone battery. Sub-1 MB model is
  the target. Inference needs to fit in ~10 ms.
- **High specificity.** A false fire every few minutes is unacceptable —
  the wake word triggers a downstream voice assistant.
- **Robust.** It has to work across voices, microphones, distances,
  ambient noise, AGC-processed audio paths, the user mumbling, the user
  speaking the words run-together as one syllable.

I targeted Android + iOS, so the output format had to be
[ONNX](https://onnx.ai) or TFLite — something both platforms have a
working runtime for.

## Attempt 1: BC-ResNet from scratch on mel-spectrograms

The textbook architecture for tiny keyword spotting is
[BC-ResNet](https://arxiv.org/abs/2106.04140) — a depthwise-separable
ConvNet that factorizes 2-D convolutions into a frequency branch and a
time branch. At τ=3 (a width scale factor), it's ~40k parameters. Most
of the original "Hey Siri" / "Hey Google" papers used variants of this.

**Front-end.** 1.2 s of audio → log-mel spectrogram, 40 mels × 121
frames at 25/10 ms STFT.

**Data.** This is where it got interesting. I generated positives via
cloud TTS — OpenAI's `gpt-4o-mini-tts` and Google's Gemini TTS, ~3000
utterances across ~40 voices with prosody and accent variation prompts
(Indian, Eastern European, regional American, fast, slow, whispered).
I added offline speed variants at ±15% to teach tempo invariance.

For negatives I used Google's Speech Commands v0.02 (~100k clips of short
isolated non-trigger words) plus ~3000 same-voice cloud TTS samples of
phonetic neighbors like "Hey buddy", "Hey Bobby", "Hey Siri".

I trained the classic recipe: BCE loss, hard-negative mining every 5
epochs, EMA shadow weights, balanced batches.

**Results.** Test set: AUC=0.9999, FRR=0.7% at 0.3 false fires per hour.
Looked beautiful.

**Reality.** On my real phone microphone the model would fire on
*literally anything that sounded like "Hey ___"* — "hey there", "hey
buddy", "hey um" all triggered. With aggressive thresholding it would
miss real "Hey Bubba" instead.

**Lesson 1: Synthetic test metrics are a lower bound on the gap to
production, not the gap itself.** My test negatives were drawn from the
same Speech Commands distribution as my training negatives — both
isolated speech words at clean recording quality. The real world is
continuous conversational speech through a phone mic with AGC. Different
distribution entirely.

## Attempts 2–3: closing the obvious gaps

I knew domain mismatch was the issue, so I cranked the augmentation:

- **AGC simulation:** soft-knee compression at random threshold/ratio,
  to mimic the dynamic-range compression Android's
  `MediaRecorder.AudioSource.VOICE_RECOGNITION` applies.
- **Random shelf EQ:** to simulate different microphone frequency
  responses.
- **Heavier bandpass + codec roundtrips:** for cheap-mic and codec
  artifacts.
- **UrbanSound8K** mixed into the noise pool: 8732 clips of traffic,
  car horns, drilling, engines, AC units.

Retrained, re-exported. Test metrics held: AUC 1.0000, FRR 0.3% at 0.3
FAR/h. On the phone, **noticeably better** — survived noisier
environments, handled different microphone gains. But still:

- Triggered on "Hey" alone if I said it with confidence
- Missed quiet "Hey Bubba" almost completely
- Required different default thresholds for "Raw" vs "Processed" audio
  source on Android

The model had learned the *acoustic signature* of cloud TTS positives,
not the *phonetic content* of "Hey Bubba." More augmentation kept
pushing the boundary out but it never closed.

## Attempt 4: switch to a pretrained speech embedding

This is where I should have started.

[openWakeWord](https://github.com/dscripka/openWakeWord) does the obvious
thing: instead of training a classifier on mel features directly, it
uses Google's pretrained `speech_embedding` model (originally on
[TFHub](https://tfhub.dev/google/speech_embedding/1)) as a frozen
feature extractor. The embedding model was trained
self-supervised on ~10k hours of speech with phoneme and speaker
auxiliary tasks. Its 96-dim output is already invariant to speaker,
loudness, microphone — the things my BC-ResNet had to learn from scratch
and never fully mastered.

```
audio (16 kHz)
   │
   ▼ melspectrogram.onnx   (frozen, ~1 MB)
mel features (32 bins/frame)
   │
   ▼ embedding_model.onnx  (frozen, ~1.3 MB)
96-dim speech embeddings
   │
   ▼ classifier (trained)  (~200 KB)
score
```

I converted the embedding model to PyTorch with `onnx2torch`, swapped my
mel front-end out, built a small 111k-parameter Conv1D classifier on
top of the 96-dim feature stream, and retrained on the same data.

Test set: AUC 1.0000, FRR 1.0%. On the phone:

**It fired on pure silence.**

Like, p=0.68 on zero-padded silence. p=0.90 on quiet noise floor. Real
"Hey Bubba" got p=0.77, *lower* than noise.

**Lesson 2: Beware sneaky training-data artifacts.** The bug was in my
padding logic. The openWakeWord pipeline expects ~2 s audio windows. My
positives were ~700 ms of "Hey Bubba" — so I padded to 2 s with zeros.
My negatives (Speech Commands clips) were ~1 s — they got less padding.
Per-frame, my positives contained *more silence than my negatives*. The
classifier learned "lots of zero-energy frames ≈ positive." The
pretrained embedding actually made this worse, because its silence
embedding is highly consistent — perfectly reinforcing whatever pattern
I'd trained.

I could have fixed it (pad with noise, add explicit silence negatives).
Instead, I did the more pragmatic thing.

## Attempt 5: openWakeWord's actual training pipeline

openWakeWord ships not just an inference runtime but a complete training
pipeline that has produced shipped models for "alexa", "hey jarvis",
"hey mycroft", etc. Their YAML-driven `train.py`:

1. Generates positive clips via `piper-sample-generator` (Linux-only,
   fork of [piper](https://github.com/rhasspy/piper))
2. Generates **adversarial phoneme-overlap negatives** automatically —
   the system enumerates phrases that share phonemes with your target
   and synthesizes them, instead of you guessing which confusables
   matter
3. Augments with real RIRs and a noise pool
4. Trains a classifier head against a pre-computed feature dump of
   **~2000 hours** of continuous background speech (ACAV100M)
5. Auto-tunes negative weighting to hit a target false-positive rate

The third bullet is the one that mattered. The reason BC-ResNet kept
firing on "hey ___" wasn't that it learned the wrong features — it was
that it had never seen what *real continuous human speech that isn't the
wake word* sounds like. Speech Commands was the wrong negative
distribution. AudioSet / LibriSpeech / ACAV100M is the right one.

I provisioned a GCP n1-standard-4 with a T4, set up their environment,
fed in my existing cloud TTS positives, downloaded the 17 GB
pre-computed feature file, and ran their pipeline.

```yaml
target_phrase:
  - "hey bubba"
  - "heybubba"      # for the run-together pronunciation

custom_negative_phrases:
  - "hey buddy"
  - "hey bobby"
  - "hey siri"
  - "hi bubba"
  # ...

batch_n_per_class:
  ACAV100M_sample: 1024     # 2000h of conversational speech
  adversarial_negative: 50
  positive: 50

target_false_positives_per_hour: 0.2
```

Their training script auto-found a `negative_weight` that hit (close to)
that target. Reported metrics on the auto test set: Recall 0.48, FPR/h
0.62. The recall sounds scary but the test set is dominated by
phoneme-overlap adversarials that are genuinely hard.

I ran it on my Mac mic via openWakeWord's own `Model.predict()`. It
worked. Fired cleanly on "Hey Bubba" and "Heybubba", stayed dark on
silence, conversational speech, "Hey there".

## Porting to Android: two last bugs

Now I just needed to wrap it in an APK. I already had the ORT chain
working — I'd been shipping it for previous attempts. Same code, swap
the classifier ONNX. Done, right?

The APK shipped. Score was **0.000 on everything.**

I verified the Python streaming chain still worked. I checked the
ONNX file hash matched what shipped in the APK. The bug had to be in
the Kotlin code.

I went back to openWakeWord's `_get_melspectrogram` source. Two
preprocessing steps I'd glossed over:

```python
# openwakeword/utils.py
if x.dtype != np.int16:
    raise ValueError("Input data must be 16-bit integers...")
x = x.astype(np.float32) if x.dtype != np.float32 else x

# ... call mel ONNX with this audio ...

# Arbitrary transform of melspectrogram
spec = melspec_transform(spec)   # = lambda x: x/10 + 2
```

Two things:

1. **The mel ONNX expects audio in int16 *magnitude* range** — float32
   values from `-32768` to `32767`. Not the canonical `[-1, 1]` you get
   from `AudioRecord` with `ENCODING_PCM_FLOAT`. The model takes float
   but expects you to have multiplied by 32767 first.

2. **There's a post-transform on the mel output**: `mel = mel/10 + 2`.
   This is a hack the openWakeWord author applied to make the ONNX
   melspec output match the original TFHub TensorFlow implementation,
   which is what `speech_embedding` was trained against.

Skip either step and the embedding model sees out-of-distribution input.
The classifier, in turn, sees garbage and outputs ~0 for everything.

Added both fixes, rebuilt:

```kotlin
val scaled = FloatArray(windowSamples)
for (i in 0 until windowSamples) scaled[i] = audioRing[i] * 32767f
// ... mel.run(...) ...
val src = melArr[0][0][T - take + row]
val dst = FloatArray(MEL_BINS)
for (b in 0 until MEL_BINS) dst[b] = src[b] / 10f + 2f    // the missing transform
```

The APK now works.

**Lesson 3: Reproduce the reference implementation byte-for-byte.**
Pretrained models often have invisible preprocessing — input scaling
quirks, output transforms, expected ranges. The reference code is the
spec. If you're not getting the same scores as the reference on the
same input, something in your pipeline is different from theirs.

A useful diagnostic: take the *same* audio buffer, run it through
the reference (`openwakeword.Model.predict()`) and your port, compare
scores. If they diverge, bisect the chain — feed identical audio to
each ONNX step, compare outputs at each stage.

## Practical takeaways

### 1. Use a pretrained speech embedding. Don't train mel features end-to-end.

This is the biggest lesson. A frozen embedding pretrained on thousands
of hours of speech is doing work you can't replicate with a small dataset
no matter how clever your augmentation is. The classifier on top of it
can be tiny (~100k params) and still generalize. The classifier on top
of raw mel features needs to be larger and still won't generalize as
well, because it's also implicitly learning *what speech sounds like*.

### 2. Your negatives need to match the distribution of "audio that isn't the wake word in the wild."

Not "isolated short words." Not "noise." Real conversational speech,
real music in cars, real podcast playback. openWakeWord uses
~2000 hours of ACAV100M for this. You don't have to be that thorough,
but Speech Commands alone (or any small short-form dataset) is
strictly insufficient.

### 3. Be paranoid about preprocessing parity at the deployment boundary.

The model that works in Python won't work in Kotlin unless every step
of preprocessing is identical. Audio scale, sample rate, byte order,
window size, padding strategy, normalization constants, post-transforms.
Print the same buffer through both pipelines and compare numerically.
A diff at any layer means something silent is going wrong.

### 4. Test against the actual deployment audio path.

Android has several audio sources (`MIC`, `VOICE_RECOGNITION`,
`UNPROCESSED`). They each go through different DSP chains. Your model
trained on clean TTS audio will hit each of them differently. Test all
of them on a real device. (On Huawei devices, `UNPROCESSED` is silently
broken — you get a valid `AudioRecord` that captures effectively zero.)

### 5. Synthetic test metrics will lie to you.

`AUC = 1.0` on the synth test set predicts almost nothing about real
performance. Always validate on actual recorded human voice in real
acoustic conditions before declaring victory. Build a small "real test
set" early — 20–50 recordings of your wake word from different speakers
in different environments — and grade every model against it.

### 6. Streaming inference is not the same as one-shot inference.

If a model was trained with a rolling feature buffer (most KWS systems
are), giving it a one-shot 2-second audio window may produce
out-of-distribution embeddings. Match the streaming pattern: maintain
the same rolling buffers the reference does.

---

## What this looks like in code

The final architecture in one diagram:

```
audio @ 16 kHz, 80 ms chunks (1280 samples)
  │
  ▼ append to ring buffer (2.04 s)
  ▼ run melspectrogram.onnx on ring (after *32767 scaling)
  ▼ apply mel_out = mel_out / 10 + 2
  ▼ take the newest 8 mel frames, push to mel deque (cap 196)
  ▼
  ▼ if deque has ≥76 frames: run embedding_model.onnx on the latest 76
  ▼   → push 96-d embedding to embedding deque (cap 16)
  ▼
  ▼ if embedding deque has 16: run hey_bubba_v0.1.onnx
  ▼   → sigmoid score
  ▼
  ▼ smooth over last 3 scores, fire when > threshold + cooldown elapsed
```

Three ONNX models, total ~2.5 MB shipped in the APK. CPU inference, no
GPU needed on device. Lives comfortably alongside an always-listening
service.

---

## Conclusion

Most of the effort in this project went into the wrong things — training
classifiers on mel features, hand-curating "hard negative" phrases, fine-
tuning augmentation probabilities. The thing that actually moved the
needle was switching to a pretrained speech embedding and using
openWakeWord's training pipeline against a properly-distributed
negative set.

If you're building a custom wake word: start with openWakeWord. If you
have a specific reason to roll your own (model size constraints,
non-English languages, exotic deployment targets), at least use a
pretrained speech embedding as your front-end. The "train end-to-end on
mel features" path is a tar pit, even if you have a lot of compute.

And before you ship: do a numerical parity check between your Python
reference and your on-device port. Two of my five models broke not in
the model itself but in the boring preprocessing code wrapping it.

---

*Repo and training scripts at [github.com/aswincsekar/whistle](https://github.com/aswincsekar/whistle).
Final working APK at [Releases v0.16.0](https://github.com/aswincsekar/whistle/releases/tag/v0.16.0).*
