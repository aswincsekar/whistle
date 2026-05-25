"""Record real-voice 'Hey Bubba' positives for fine-tuning.

Captures N short clips, varies prompts to cover prosody/distance/speed.
Saves to data/positives_real/<speaker>_<idx>.wav (16 kHz mono).

Usage:
    python scripts/record_positives.py --speaker aswin --count 30
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

PROMPTS = [
    ("Hey Bubba",            "normal pace, neutral tone"),
    ("Hey Bubba",            "louder, like calling out"),
    ("Hey Bubba",            "quieter, conversational"),
    ("Hey Bubba",            "fast (one word almost)"),
    ("Hey Bubba",            "slow and deliberate"),
    ("Hey Bubba",            "questioning - 'Hey Bubba?'"),
    ("Hey Bubba",            "excited - 'Hey Bubba!'"),
    ("Hey Bubba",            "flat / monotone"),
    ("Hey Bubba",            "back away ~3 feet from mic"),
    ("Hey Bubba",            "lean in close to mic"),
    ("Hey Bubba",            "say at end of an exhale"),
    ("Hey Bubba",            "say at start of an inhale"),
    ("Hey Bubba",            "look away from mic while saying it"),
    ("Hey Bubba",            "with a little smile in your voice"),
    ("Hey Bubba",            "stern / firm tone"),
]


def record_clip(sr: int, duration: float) -> np.ndarray:
    audio = sd.rec(int(sr * duration), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    return audio.reshape(-1)


def trim_to_speech(x: np.ndarray, sr: int, top_db: float = 30.0) -> np.ndarray:
    """Trim leading/trailing silence with a tiny margin so the wake word lands ~centered."""
    abs_x = np.abs(x)
    if abs_x.max() < 1e-4:
        return x
    threshold = abs_x.max() * 10 ** (-top_db / 20.0)
    above = np.where(abs_x > threshold)[0]
    if above.size == 0:
        return x
    margin = int(0.1 * sr)   # 100 ms padding on each side
    start = max(0, above[0] - margin)
    end = min(x.size, above[-1] + margin)
    return x[start:end]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--speaker", required=True, help="speaker id (lowercase, no spaces)")
    ap.add_argument("--count", type=int, default=30)
    ap.add_argument("--out-dir", default="data/positives_real")
    ap.add_argument("--seconds", type=float, default=2.0, help="recording window per clip")
    ap.add_argument("--device", type=int, default=None, help="sounddevice input device id")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sr = 16000

    print(f"\nRecording {args.count} 'Hey Bubba' samples for speaker '{args.speaker}'.")
    print(f"Each recording is {args.seconds}s. You'll be prompted on style.")
    print(f"Files -> {out_dir.resolve()}/{args.speaker}_NNN.wav\n")
    print("Tip: vary distance, volume, prosody. The more diverse, the more robust.\n")

    if args.device is not None:
        sd.default.device = args.device
    print(f"input device: {sd.query_devices(sd.default.device[0] if isinstance(sd.default.device, (list, tuple)) else sd.default.device)['name']}\n")

    for i in range(args.count):
        prompt_text, style = PROMPTS[i % len(PROMPTS)]
        print(f"[{i+1:2d}/{args.count}]  Say \"{prompt_text}\"  ({style})")
        input("           press Enter then say it ...")
        print("           RECORDING...", end="", flush=True)
        clip = record_clip(sr, args.seconds)
        clip = trim_to_speech(clip, sr)

        rms = float(np.sqrt(np.mean(clip ** 2) + 1e-12))
        db = 20 * np.log10(rms + 1e-9)

        out = out_dir / f"{args.speaker}_{i:03d}.wav"
        sf.write(str(out), clip, sr)
        print(f"  saved {out.name}  ({len(clip)/sr:.2f}s, {db:+.1f} dBFS)")

        if db < -45:
            print("           ! looks very quiet - move closer or raise mic gain")
        if db > -3:
            print("           ! clipping risk - back off the mic")

    print(f"\nDone. {args.count} clips in {out_dir.resolve()}\n")
    print("Next: scp to VM and fine-tune.")


if __name__ == "__main__":
    main()
