#!/usr/bin/env python3
"""
Batch voice conversion with a Seed VC checkpoint.

Converts every audio file in --source-dir to the voice of the --target
reference audio, loading the models only once (inference.py is invoked
in-process per file instead of as a new subprocess per file).

Output files keep the original names (and folder structure, if the
source folder has subfolders).

Example:
  python batch_convert.py \
    --source-dir /path/to/source_wavs \
    --target /path/to/reference_voice.wav \
    --checkpoint runs/my_run/ft_model.pth \
    --config runs/my_run/config_dit_mel_seed_uvit_whisper_small_wavenet.yml \
    --output-dir /path/to/output
"""

import argparse
import os
import sys
import traceback

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch-convert a folder of audio files with a Seed VC checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-dir", type=str, required=True,
                        help="Folder with source audio files (searched recursively).")
    parser.add_argument("--target", type=str, required=True,
                        help="Reference audio whose voice the output will have.")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Folder where converted files are written.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Fine-tuned checkpoint. Falls back to the pretrained HF model.")
    parser.add_argument("--config", type=str, default=None,
                        help="Config matching the checkpoint.")
    parser.add_argument("--diffusion-steps", type=int, default=25)
    parser.add_argument("--length-adjust", type=float, default=1.0)
    parser.add_argument("--inference-cfg-rate", type=float, default=0.7)
    parser.add_argument("--f0-condition", action="store_true",
                        help="Enable F0 conditioning (must match the config).")
    parser.add_argument("--auto-f0-adjust", action="store_true")
    parser.add_argument("--semi-tone-shift", type=int, default=0)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit", type=int, default=None,
                        help="Convert at most N files (useful for testing).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-convert even if the output already exists.")
    return parser.parse_args()


def collect_sources(source_dir):
    files = []
    for root, _, names in os.walk(source_dir):
        for name in sorted(names):
            if name.lower().endswith(AUDIO_EXTENSIONS):
                files.append(os.path.join(root, name))
    return files


def final_output_path(source_path, source_dir, output_dir):
    """Output path keeping the original file name and folder structure."""
    rel = os.path.relpath(source_path, source_dir)
    rel_noext = os.path.splitext(rel)[0]
    return os.path.join(output_dir, rel_noext + ".wav")


def inference_output_name(source_path, target_path, args):
    """Name that inference.py gives the file (used to rename afterwards)."""
    source_name = os.path.splitext(os.path.basename(source_path))[0]
    target_name = os.path.splitext(os.path.basename(target_path))[0]
    return (f"vc_{source_name}_{target_name}_{args.length_adjust}_"
            f"{args.diffusion_steps}_{args.inference_cfg_rate}.wav")


def main():
    args = parse_args()

    if not os.path.isdir(args.source_dir):
        print(f"ERROR: source directory not found: {args.source_dir}")
        sys.exit(1)
    if not os.path.isfile(args.target):
        print(f"ERROR: target reference audio not found: {args.target}")
        sys.exit(1)
    if args.checkpoint and not os.path.isfile(os.path.join(REPO_ROOT, args.checkpoint)) \
            and not os.path.isfile(args.checkpoint):
        print(f"ERROR: checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    sources = collect_sources(args.source_dir)
    if args.limit:
        sources = sources[: args.limit]
    if not sources:
        print(f"ERROR: no audio files found in {args.source_dir}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    sys.path.insert(0, REPO_ROOT)
    import inference

    # Namespace mimicking inference.py's CLI arguments.
    ns = argparse.Namespace(
        source=None,
        target=args.target,
        output=args.output_dir,
        diffusion_steps=args.diffusion_steps,
        length_adjust=args.length_adjust,
        inference_cfg_rate=args.inference_cfg_rate,
        f0_condition=args.f0_condition,
        auto_f0_adjust=args.auto_f0_adjust,
        semi_tone_shift=args.semi_tone_shift,
        checkpoint=args.checkpoint,
        config=args.config,
        fp16=args.fp16,
    )

    print("Loading models (once)...")
    models = inference.load_models(ns)
    # Reuse the loaded models for every file instead of reloading per call.
    inference.load_models = lambda _ns: models

    failures = []
    done = skipped = 0
    for i, source in enumerate(sources, 1):
        out_file = final_output_path(source, args.source_dir, args.output_dir)
        if os.path.exists(out_file) and not args.overwrite:
            skipped += 1
            continue
        ns.source = source
        try:
            inference.main(ns)
            produced = os.path.join(args.output_dir,
                                    inference_output_name(source, args.target, ns))
            if produced != out_file:
                os.makedirs(os.path.dirname(out_file), exist_ok=True)
                os.replace(produced, out_file)
            done += 1
            print(f"[{i}/{len(sources)}] OK: {out_file}")
        except Exception:
            failures.append(source)
            print(f"[{i}/{len(sources)}] FAILED: {source}")
            traceback.print_exc()

    print(f"\nConverted: {done} | skipped (already existed): {skipped} | failed: {len(failures)}")
    if failures:
        print("Failed files:")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
