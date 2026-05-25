"""Record audio + dump per-window wake-word probabilities.

Usage:
    python scripts/probe_mic.py --ckpt checkpoints/best.pt --seconds 8
Press Enter, say "Hey Bubba" a few times in the recording window,
then we'll print the model's scores per hop.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from whistle.eval import ckpt_config, _load_ckpt
from whistle.features import FeatureConfig, LogMelSpectrogram
from whistle.models import build_model
from whistle.train import _positive_logit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--wav", help="If set, score this file instead of recording.")
    args = ap.parse_args()

    cfg = ckpt_config(args.ckpt)
    sr = cfg.audio.sample_rate
    window = int(sr * cfg.audio.window_seconds)
    hop = int(sr * cfg.audio.hop_seconds)

    feat_cfg = FeatureConfig.from_cfg(cfg)
    frontend = LogMelSpectrogram(feat_cfg).eval()
    model = build_model(cfg).eval()
    _load_ckpt(model, args.ckpt, use_ema=True)

    if args.wav:
        audio, file_sr = sf.read(args.wav, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if file_sr != sr:
            import librosa
            audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)
        print(f"[probe] scoring {args.wav} ({len(audio)/sr:.1f}s)")
    else:
        import sounddevice as sd
        input(f"[probe] press Enter then say 'Hey Bubba' a few times ({args.seconds}s recording)...")
        print("[probe] RECORDING...")
        audio = sd.rec(int(sr * args.seconds), samplerate=sr, channels=1, dtype="float32")
        sd.wait()
        audio = audio.reshape(-1)
        out = "probe.wav"
        sf.write(out, audio, sr)
        print(f"[probe] saved -> {out}")

    if audio.shape[0] < window:
        audio = np.pad(audio, (0, window - audio.shape[0]))

    print(f"\n  {'t (s)':>6}  {'mic dBFS':>9}  {'p':>6}  bar")
    print("  " + "-" * 60)
    peak_p = 0.0
    peak_t = 0.0
    with torch.no_grad():
        for start in range(0, audio.shape[0] - window + 1, hop):
            win = audio[start:start + window]
            rms = float(np.sqrt(np.mean(win ** 2) + 1e-12))
            db = 20 * np.log10(rms + 1e-9)
            x = torch.from_numpy(win).unsqueeze(0)
            mel = frontend(x)
            logits = _positive_logit(model(mel))
            p = float(torch.sigmoid(logits).item())
            if p > peak_p:
                peak_p, peak_t = p, start / sr
            n = int(min(40, max(0, p * 40)))
            bar = "#" * n + "-" * (40 - n)
            print(f"  {start/sr:6.2f}  {db:+9.1f}  {p:6.3f}  |{bar}|")

    print(f"\n  peak p = {peak_p:.3f} at t = {peak_t:.2f}s")
    print(f"  (threshold 0.84 = WAKE)  |"
          f"{'WOULD FIRE' if peak_p >= 0.84 else 'no fire at thr=0.84':^16}|")


if __name__ == "__main__":
    main()
