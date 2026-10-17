#!/usr/bin/env python3
"""
Seed VC fine-tuning pipeline.

Standalone script that reproduces the workflow from
`seed_vc_finetuning_example.ipynb`:

  1. (Optional) Download the example dataset used in the notebook.
  2. (Optional) Zero-shot evaluation before training (baseline).
  3. Fine-tune the model with train.py.
  4. Evaluation after training with the fine-tuned checkpoint.
  5. (Optional) Launch the Gradio web UI with the fine-tuned model.

The dataset directory must contain one subfolder per speaker, with audio
files (.wav/.mp3/.flac/.ogg/.m4a/.opus) inside. Clips should be between
1 and 30 seconds long.

Usage examples:
  # Full pipeline with default settings
  python finetune.py --dataset-dir ./dataset

  # Train only, no evaluations
  python finetune.py --dataset-dir ./dataset --skip-pre-eval --skip-post-eval

  # Custom training hyperparameters and multiple reference audios
  python finetune.py --dataset-dir ./dataset --run-name my_run \
      --batch-size 4 --max-steps 500 --save-every 100 \
      --target dataset/speaker1/ref1.wav dataset/speaker2/ref2.wav

  # Launch the web UI after training finishes
  python finetune.py --dataset-dir ./dataset --launch-ui
"""

import argparse
import os
import shutil
import subprocess
import sys
import zipfile

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

DEFAULT_CONFIG = os.path.join("configs", "presets",
                              "config_dit_mel_seed_uvit_whisper_small_wavenet.yml")
DEFAULT_SOURCE = os.path.join("examples", "source", "source_s1.wav")

EXAMPLE_DATASET_REPO = "XzJosh/audiodataset"
EXAMPLE_DATASET_ARCHIVES = ["永雏塔菲.zip", "科比.zip"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune Seed VC on a custom voice dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    data = parser.add_argument_group("data")
    data.add_argument("--dataset-dir", type=str, default=None,
                      help="Directory containing speaker subfolders with audio files.")
    data.add_argument("--download-example-dataset", action="store_true",
                      help="Download and extract the example dataset from the notebook "
                           "into --dataset-dir before training.")

    # Training
    train = parser.add_argument_group("training")
    train.add_argument("--config", type=str, default=DEFAULT_CONFIG,
                       help="Training config (YAML).")
    train.add_argument("--pretrained-ckpt", type=str, default=None,
                       help="Optional pretrained checkpoint to start from.")
    train.add_argument("--run-name", type=str, default="my_finetune_run",
                       help="Name of the run. Checkpoints are saved under ./runs/<run-name>/.")
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--max-steps", type=int, default=200,
                       help="Total number of training steps.")
    train.add_argument("--max-epochs", type=int, default=1000,
                       help="Upper bound on epochs (training stops at --max-steps first).")
    train.add_argument("--save-every", type=int, default=100,
                       help="Save a checkpoint every N steps.")
    train.add_argument("--num-workers", type=int, default=2)
    train.add_argument("--gpu", type=int, default=0, help="GPU id to use.")

    # Evaluation
    eval_ = parser.add_argument_group("evaluation")
    eval_.add_argument("--source", type=str, default=DEFAULT_SOURCE,
                       help="Source audio used for evaluation inference.")
    eval_.add_argument("--target", type=str, nargs="+", default=None,
                       help="One or more reference audio files for evaluation inference. "
                            "Defaults to the first audio file found in each speaker folder "
                            "of the dataset.")
    eval_.add_argument("--skip-pre-eval", action="store_true",
                       help="Skip zero-shot evaluation before training.")
    eval_.add_argument("--skip-post-eval", action="store_true",
                       help="Skip evaluation with the fine-tuned checkpoint.")
    eval_.add_argument("--diffusion-steps", type=int, default=25)
    eval_.add_argument("--length-adjust", type=float, default=1.0)
    eval_.add_argument("--inference-cfg-rate", type=float, default=0.7)

    # Deployment
    deploy = parser.add_argument_group("deployment")
    deploy.add_argument("--launch-ui", action="store_true",
                        help="Launch the Gradio web UI with the fine-tuned model after training.")
    deploy.add_argument("--share", action="store_true",
                        help="Create a public Gradio link (used with --launch-ui).")

    return parser.parse_args()


def run(cmd, desc):
    """Run a command, streaming its output, and abort the pipeline on failure."""
    print(f"\n{'=' * 70}\n>> {desc}\n{'=' * 70}")
    print(" ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    if result.returncode != 0:
        print(f"\nERROR: '{desc}' failed with exit code {result.returncode}. Aborting.")
        sys.exit(result.returncode)


def check_repo_layout():
    for rel in ["train.py", "inference.py", DEFAULT_CONFIG]:
        if not os.path.exists(os.path.join(REPO_ROOT, rel)):
            print(f"ERROR: '{rel}' not found. Run this script from the seed-vc repository.")
            sys.exit(1)


AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus")


def find_default_targets(dataset_dir):
    """Pick one audio file per speaker to use as evaluation reference.

    Supports both dataset layouts:
      <dataset>/<speaker>/*.wav    (one subfolder per speaker)
      <dataset>/*.wav              (single speaker, files in the root)
    """
    groups = {}
    for root, _, files in os.walk(dataset_dir):
        for file in sorted(files):
            if file.lower().endswith(AUDIO_EXTENSIONS):
                rel = os.path.relpath(os.path.join(root, file), dataset_dir)
                speaker = rel.split(os.sep)[0] if os.sep in rel else ""
                groups.setdefault(speaker, os.path.join(root, file))
    return [groups[key] for key in sorted(groups)]


def download_example_dataset(dataset_dir):
    """Download and extract the example dataset used in the notebook."""
    print(f"\n>> Downloading example dataset '{EXAMPLE_DATASET_REPO}'...")
    os.makedirs(dataset_dir, exist_ok=True)
    for archive in EXAMPLE_DATASET_ARCHIVES:
        run(["huggingface-cli", "download", EXAMPLE_DATASET_REPO, archive,
             "--repo-type", "dataset", "--local-dir", dataset_dir],
            f"Downloading {archive}")
        zip_path = os.path.join(dataset_dir, archive)
        if os.path.exists(zip_path):
            print(f"Extracting {zip_path} ...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(dataset_dir)
            os.remove(zip_path)


def run_inference(source, targets, output_dir, config=None, checkpoint=None):
    for target in targets:
        cmd = [
            sys.executable, "inference.py",
            "--source", source,
            "--target", target,
            "--output", output_dir,
            "--diffusion-steps", str(args.diffusion_steps),
            "--length-adjust", str(args.length_adjust),
            "--inference-cfg-rate", str(args.inference_cfg_rate),
        ]
        if config:
            cmd += ["--config", config]
        if checkpoint:
            cmd += ["--checkpoint", checkpoint]
        run(cmd, f"Inference: {os.path.basename(source)} -> {os.path.basename(target)}")

        source_name = os.path.splitext(os.path.basename(source))[0]
        target_name = os.path.splitext(os.path.basename(target))[0]
        out_file = os.path.join(
            output_dir,
            f"vc_{source_name}_{target_name}_{args.length_adjust}_"
            f"{args.diffusion_steps}_{args.inference_cfg_rate}.wav",
        )
        if os.path.exists(out_file):
            print(f"Output: {out_file}")


def main():
    global args
    args = parse_args()

    check_repo_layout()

    dataset_dir = args.dataset_dir
    if args.download_example_dataset:
        if not dataset_dir:
            dataset_dir = os.path.join(REPO_ROOT, "dataset")
        download_example_dataset(dataset_dir)
    if not dataset_dir:
        print("ERROR: --dataset-dir is required (or use --download-example-dataset).")
        sys.exit(1)
    dataset_dir = os.path.abspath(dataset_dir)
    if not os.path.isdir(dataset_dir):
        print(f"ERROR: dataset directory not found: {dataset_dir}")
        sys.exit(1)

    targets = args.target if args.target else find_default_targets(dataset_dir)
    if not targets:
        print(f"ERROR: no reference audio found in {dataset_dir}. "
              "Pass --target explicitly.")
        sys.exit(1)
    print(f"Reference audio(s) for evaluation: {targets}")

    source = args.source if os.path.isabs(args.source) else os.path.join(REPO_ROOT, args.source)
    if not os.path.exists(source):
        print(f"ERROR: source audio not found: {source}")
        sys.exit(1)

    output_dir = os.path.join(REPO_ROOT, "reconstructed")
    os.makedirs(output_dir, exist_ok=True)

    # Step 1 (optional): zero-shot baseline
    if not args.skip_pre_eval:
        run_inference(source, targets, output_dir)

    # Step 2: training
    train_cmd = [
        sys.executable, "train.py",
        "--config", args.config,
        "--dataset-dir", dataset_dir,
        "--run-name", args.run_name,
        "--batch-size", str(args.batch_size),
        "--max-steps", str(args.max_steps),
        "--max-epochs", str(args.max_epochs),
        "--save-every", str(args.save_every),
        "--num-workers", str(args.num_workers),
        "--gpu", str(args.gpu),
    ]
    if args.pretrained_ckpt:
        train_cmd += ["--pretrained-ckpt", args.pretrained_ckpt]
    run(train_cmd, f"Fine-tuning (run: {args.run_name})")

    run_dir = os.path.join(REPO_ROOT, "runs", args.run_name)
    checkpoint = os.path.join(run_dir, "ft_model.pth")
    run_config = os.path.join(run_dir, os.path.basename(args.config))
    if not os.path.exists(checkpoint):
        print(f"ERROR: expected checkpoint not found: {checkpoint}")
        sys.exit(1)

    # Step 3 (optional): evaluation with the fine-tuned checkpoint
    if not args.skip_post_eval:
        run_inference(source, targets, output_dir,
                      config=run_config, checkpoint=checkpoint)

    # Step 4 (optional): deploy with the Gradio UI
    if args.launch_ui:
        run([
            sys.executable, "app_vc.py",
            "--checkpoint", checkpoint,
            "--config", run_config,
            "--share", str(args.share),
        ], "Launching web UI")

    print(f"\nDone. Fine-tuned checkpoint: {checkpoint}")
    if not args.launch_ui:
        print(f"To deploy it later, run:\n"
              f"  python app_vc.py --checkpoint {checkpoint} --config {run_config}")


if __name__ == "__main__":
    main()
