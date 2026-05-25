"""Single CLI entrypoint: `whistle <subcommand>`."""
from __future__ import annotations

from pathlib import Path

import click

from .config import load_config, repo_root

# Load .env from the repo root so API keys land in os.environ before any
# subcommand runs. Silent if the file doesn't exist or python-dotenv isn't
# installed (only needed for cloud-synth).
try:
    from dotenv import load_dotenv
    _env_path = repo_root() / ".env"
    if _env_path.exists():
        load_dotenv(_env_path, override=False)
except Exception:
    pass


@click.group()
def main():
    """whistle - 'Hey Bubba' wake-word pipeline."""


@main.command()
@click.option("--config", "cfg_path", default="configs/default.yaml")
@click.option("--count", default=400, help="Number of positive utterances to synthesize.")
@click.option("--piper-voice", multiple=True, help="Path(s) to piper .onnx voice models.")
def synth(cfg_path: str, count: int, piper_voice: tuple[str, ...]):
    """Generate 'Hey Bubba' positives via TTS."""
    from .data.synth import synth_positives, write_manifest
    cfg = load_config(cfg_path)
    root = repo_root()
    out_dir = root / cfg.data.positives_dir
    phrases = list(cfg.wakeword.variants)
    records = synth_positives(
        out_dir=out_dir,
        phrases=phrases,
        count=count,
        sample_rate=cfg.audio.sample_rate,
        piper_voices=list(piper_voice) if piper_voice else None,
    )
    write_manifest(records, root / cfg.data.manifest_dir / "positives_synth.jsonl")
    click.echo(f"synthesized {len(records)} positives -> {out_dir}")


@main.command("cloud-synth")
@click.option("--config", "cfg_path", default="configs/default.yaml")
@click.option("--elevenlabs", "n_eleven", default=0, type=int, help="ElevenLabs utterance count.")
@click.option("--openai", "n_openai", default=0, type=int, help="OpenAI tts utterance count.")
@click.option("--gemini", "n_gemini", default=0, type=int, help="Gemini TTS utterance count.")
@click.option("--out", "out_dir", default=None,
              help="Output directory (defaults to data/positives_cloud or data/negatives_cloud).")
@click.option("--workers", default=8, type=int, help="Parallel API requests.")
@click.option("--seed", default=0, type=int)
@click.option("--speed-variants/--no-speed-variants", default=True,
              help="Also write fast (1.15x) and slow (0.85x) copies of each clip.")
@click.option("--negatives", is_flag=True,
              help="Generate same-voice NEGATIVES (phonetic neighbors + other wake words + filler).")
def cloud_synth(cfg_path: str, n_eleven: int, n_openai: int, n_gemini: int,
                out_dir: str | None, workers: int, seed: int, speed_variants: bool,
                negatives: bool):
    """Generate 'Hey Bubba' positives (or matched negatives) via cloud TTS."""
    from pathlib import Path
    from .data.synth_cloud import default_plans, run_plan, add_speed_variants, NEGATIVE_PHRASES
    from .data.synth import write_manifest
    cfg = load_config(cfg_path)
    root = repo_root()

    if negatives:
        phrases = list(NEGATIVE_PHRASES)
        out_default = "data/negatives_cloud"
        manifest_name = "negatives_cloud.jsonl"
        label = 0
        kind = "negatives"
    else:
        phrases = list(cfg.wakeword.variants)
        out_default = "data/positives_cloud"
        manifest_name = "positives_cloud.jsonl"
        label = 1
        kind = "positives"

    plans = default_plans({"elevenlabs": n_eleven, "openai": n_openai, "gemini": n_gemini})
    out_path = root / (out_dir or out_default)
    records = run_plan(plans, phrases, out_path, max_workers=workers, seed=seed)
    # Override label (run_plan defaults to label=1).
    for r in records:
        r["label"] = label
    if speed_variants and records:
        before = len(records)
        # 4-way speed coverage: 0.7x (very slow) -> 1.3x (very fast). Combined with
        # the tempo prompts in OPENAI_STYLES / GEMINI_STYLES this gets us robust
        # coverage of fast and stretched utterances.
        records = add_speed_variants(records, factors=(0.7, 0.85, 1.15, 1.3))
        click.echo(f"  +{len(records) - before} speed-variant copies (0.70x, 0.85x, 1.15x, 1.30x)")
    manifest = root / cfg.data.manifest_dir / manifest_name
    write_manifest(records, manifest)
    click.echo(f"cloud-synth ({kind}): {len(records)} utterances -> {out_path}  (manifest: {manifest})")


@main.command()
@click.option("--config", "cfg_path", default="configs/default.yaml")
@click.option("--force", is_flag=True)
@click.option("--skip", multiple=True, help="Source keys to skip (speech_commands|ms_snsd_noise|mit_rirs).")
def download(cfg_path: str, force: bool, skip: tuple[str, ...]):
    """Download negatives, noise, and RIRs."""
    from .data.download import fetch_all
    root = repo_root()
    fetched = fetch_all(root / "data", force=force, skip=list(skip))
    for k, p in fetched.items():
        click.echo(f"  {k}: {p}")


@main.command()
@click.option("--config", "cfg_path", default="configs/default.yaml")
def manifests(cfg_path: str):
    """Rebuild train/val/test manifests."""
    from .data.manifests import build_manifests
    cfg = load_config(cfg_path)
    root = repo_root()
    paths = build_manifests(
        positives_dir=root / cfg.data.positives_dir,
        negatives_dir=root / cfg.data.negatives_dir,
        out_dir=root / cfg.data.manifest_dir,
        val_fraction=cfg.data.val_fraction,
        test_fraction=cfg.data.test_fraction,
        seed=cfg.train.seed,
    )
    for split, p in paths.items():
        click.echo(f"  {split}: {p}")


@main.command()
@click.option("--config", "cfg_path", default="configs/default.yaml")
@click.option("--init-from", "init_from", default=None,
              help="Start from an existing checkpoint (fine-tune mode).")
@click.option("--lr", "lr_override", default=None, type=float,
              help="Override config train.lr (useful with --init-from, e.g. 5e-4).")
@click.option("--epochs", "epochs_override", default=None, type=int,
              help="Override config train.epochs.")
def train(cfg_path: str, init_from: str | None, lr_override: float | None,
          epochs_override: int | None):
    """Train the wake-word model. With --init-from, fine-tune from a checkpoint."""
    from .train import train as _train
    result = _train(cfg_path, init_from=init_from,
                    lr_override=lr_override, epochs_override=epochs_override)
    click.echo(f"best ckpt: {result['best_ckpt']}  (AUC={result['best_val_auc']:.4f}, params={result['params']:,})")


@main.command()
@click.option("--config", "cfg_path", default="configs/default.yaml")
@click.option("--ckpt", "ckpt_path", required=True)
@click.option("--split", default="test")
def eval(cfg_path: str, ckpt_path: str, split: str):
    """Evaluate a checkpoint, write ROC/DET, FAR-per-hour."""
    from .eval import eval_checkpoint
    eval_checkpoint(cfg_path, ckpt_path, split=split)


@main.command()
@click.option("--config", "cfg_path", default="configs/default.yaml")
@click.option("--ckpt", "ckpt_path", required=True)
@click.option("--format", "fmt", default="onnx", type=click.Choice(["onnx", "tflite"]))
@click.option("--verify/--no-verify", default=True)
def export(cfg_path: str, ckpt_path: str, fmt: str, verify: bool):
    """Export the trained model to ONNX (and TFLite if requested)."""
    from .export import export as _export, verify_onnx
    meta = _export(cfg_path, ckpt_path, fmt=fmt)
    if verify:
        verify_onnx(cfg_path, ckpt_path, meta["outputs"]["onnx_audio"])


@main.command()
@click.option("--config", "cfg_path", default="configs/default.yaml")
@click.option("--ckpt", "ckpt_path", required=True)
@click.option("--wav", "wav_path", default=None)
@click.option("--mic", is_flag=True)
@click.option("--threshold", default=0.6, type=float)
@click.option("--verbose", is_flag=True, help="Print mic level + live probability bar (debug).")
@click.option("--min-dbfs", default=-45.0, type=float,
              help="Skip the model when the input window is quieter than this. Default -45 dBFS.")
@click.option("--cooldown", default=1.5, type=float,
              help="Seconds to suppress after a fire (prevents double-trigger). Default 1.5.")
def stream(cfg_path: str, ckpt_path: str, wav_path: str | None, mic: bool,
           threshold: float, verbose: bool, min_dbfs: float, cooldown: float):
    """Run the streaming detector on a WAV or live mic."""
    from .stream import stream_wav, stream_mic
    if mic:
        stream_mic(cfg_path, ckpt_path, threshold=threshold, verbose=verbose,
                   min_dbfs=min_dbfs, cooldown_s=cooldown)
    elif wav_path:
        stream_wav(cfg_path, ckpt_path, wav_path, threshold=threshold)
    else:
        raise click.UsageError("provide --wav PATH or --mic")


@main.command()
def smoke():
    """End-to-end smoke test on a tiny synthetic dataset (no downloads)."""
    from .smoke import run_smoke
    ok = run_smoke()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
