import argparse
import os
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from diffusers import SanaPipeline

from evaluation_code import Evaluator


def find_lora_path(subject: str, lora_root: str, checkpoint_type: str = "full"):
    """
    Finds the LoRA directory for a given subject under lora_root.

    Expected layout:

        lora_root/
            cat/
                pytorch_lora_weights.safetensors      strong
                weak/
                    pytorch_lora_weights.safetensors  weak

    checkpoint_type:
        "full" → lora_root/subject
        "weak" → lora_root/subject/weak
    """
    base = Path(lora_root)
    if not base.exists():
        raise ValueError(f"LoRA root does not exist: {lora_root}")

    # normalize subject name variants
    subject_variants = [
        subject,
        subject.replace(" ", "_"),
        subject.replace(" ", "-"),
        subject.lower(),
        subject.upper(),
    ]

    matched = None
    for folder in base.iterdir():
        if not folder.is_dir():
            continue
        name = folder.name
        lower_name = name.lower()
        for var in subject_variants:
            if lower_name == var.lower():
                matched = folder
                break
        if matched is not None:
            break

    if matched is None:
        return None

    if checkpoint_type == "full":
        return str(matched)

    if checkpoint_type == "weak":
        weak_dir = matched / "weak"
        if weak_dir.exists():
            return str(weak_dir)
        return None

    return None


def load_sana_lora(
    base_model: str,
    lora_dir: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> SanaPipeline:
    """
    Load a SanaPipeline with LoRA weights from a directory.

    lora_dir should contain at least one *.safetensors or *.bin file.
    """
    print(f"\nLoading Sana base model: {base_model}")
    pipe = SanaPipeline.from_pretrained(
        base_model,
        dtype=dtype,
    )

    lora_dir_path = Path(lora_dir)
    if not lora_dir_path.exists():
        raise ValueError(f"LoRA directory not found: {lora_dir}")

    has_lora = False
    for f in lora_dir_path.iterdir():
        if f.suffix in [".safetensors", ".bin"]:
            print("Loading LoRA:", f)
            pipe.load_lora_weights(str(f), adapter_name="default")
            has_lora = True

    if not has_lora:
        print(f"Warning: no LoRA weights found in {lora_dir}")

    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def evaluate_ag(
    base_model: str,
    lora_root: str,
    evaluation_json: str,
    output_dir: str,
    device: str = "cuda",
    cfg_scale: float = 7.5,
    num_images_per_prompt: int = 1,
    instance_data_dir: str = None,
    steps: int = 20,
    resolution: int = 1024,
) -> dict:
    """
    Sana AG style evaluation.

    For Sana we cannot keep both weak and strong pipelines on GPU at once easily,
    so this script uses the strong LoRA pipeline per subject and standard CFG,
    then computes CLIP T, CLIP I, and DINO as in the DreamBooth scripts.
    """

    os.makedirs(output_dir, exist_ok=True)

    evaluator = Evaluator(
        model_path="",  # we do not use StableDiffusion inside Evaluator
        json_path=evaluation_json,
        output_dir=output_dir,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        instance_data_dir=instance_data_dir,
    )

    print("Loaded subjects:", sorted(list(set(evaluator.subject_per_prompt))))

    # metrics over all prompts
    clip_i_scores = []
    clip_t_scores = []
    dino_scores = []

    # metrics per subject
    subject_metrics = {}

    text_features = evaluator._encode_text(evaluator.prompts)

    current_subject = None
    current_pipe = None

    for idx, prompt in enumerate(tqdm(evaluator.prompts, desc="Evaluating Sana AG")):
        subject = evaluator.subject_per_prompt[idx]

        # per subject pipeline loading
        if subject != current_subject:
            print(f"\nSwitching to subject: {subject}")

            subject_dir = find_lora_path(
                subject=subject,
                lora_root=lora_root,
                checkpoint_type="full",
            )
            if subject_dir is None:
                print(
                    f"Warning: no LoRA folder found for subject '{subject}' under {lora_root}, skipping prompts for this subject."
                )
                current_pipe = None
                current_subject = subject
                continue

            # free previous pipeline if any
            if current_pipe is not None:
                del current_pipe
                torch.cuda.empty_cache()

            current_pipe = load_sana_lora(
                base_model=base_model,
                lora_dir=subject_dir,
                device=device,
                dtype=torch.float16 if device == "cuda" else torch.float32,
            )
            current_subject = subject

            if subject not in subject_metrics:
                subject_metrics[subject] = {
                    "clip_i": [],
                    "clip_t": [],
                    "dino": [],
                    "prompt_count": 0,
                }

        if current_pipe is None:
            # failed to load for this subject
            continue

        # generate images with Sana (no blending, just LoRA enhanced CFG)
        out = current_pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=cfg_scale,
            num_images_per_prompt=num_images_per_prompt,
            height=resolution,
            width=resolution,
        )
        images = out.images  # list of PIL Images

        # encode images
        img_clip = evaluator._encode_images_clip(images)
        img_dino = evaluator._encode_images_dino(images)
        t_feat = text_features[idx].unsqueeze(0).repeat(num_images_per_prompt, 1)

        # CLIP T
        clip_t = evaluator._cosine(img_clip, t_feat).mean().item()
        clip_t_scores.append(clip_t)
        subject_metrics[subject]["clip_t"].append(clip_t)
        subject_metrics[subject]["prompt_count"] += 1

        # save first image for this prompt
        subject_dir_out = os.path.join(
            output_dir, subject.replace(" ", "_")
        )
        os.makedirs(subject_dir_out, exist_ok=True)
        prompt_safe = (
            prompt.replace("/", "_")
            .replace("\\", "_")
            .replace(" ", "_")
        )[:120]
        image_path = os.path.join(
            subject_dir_out, f"{idx:04d}_{prompt_safe}.png"
        )
        images[0].save(image_path)

        # CLIP I and DINO only if we have reference images
        if (
            evaluator.reference_images
            and subject in evaluator.reference_images
            and len(evaluator.reference_images[subject]) > 0
        ):
            ref_images = evaluator.reference_images[subject]
            ref_clip = evaluator._encode_images_clip(ref_images)
            ref_dino = evaluator._encode_images_dino(ref_images)

            # CLIP I
            clip_i_sims = []
            for gen_feat in img_clip:
                sims = F.cosine_similarity(
                    gen_feat.unsqueeze(0),
                    ref_clip,
                    dim=1,
                )
                clip_i_sims.append(sims.max().item())
            clip_i = float(np.mean(clip_i_sims))
            clip_i_scores.append(clip_i)
            subject_metrics[subject]["clip_i"].append(clip_i)

            # DINO image similarity
            dino_sims = []
            for gen_feat in img_dino:
                sims = F.cosine_similarity(
                    gen_feat.unsqueeze(0),
                    ref_dino,
                    dim=1,
                )
                dino_sims.append(sims.max().item())
            dino_i = float(np.mean(dino_sims))
            dino_scores.append(dino_i)
            subject_metrics[subject]["dino"].append(dino_i)

        print(f"[{idx}] {prompt}")
        print(f"  CLIP T: {clip_t:.4f}")
        if subject in subject_metrics and subject_metrics[subject]["clip_i"]:
            print(
                f"  last CLIP I: {subject_metrics[subject]['clip_i'][-1]:.4f},"
                f" last DINO: {subject_metrics[subject]['dino'][-1]:.4f}"
            )

    # final summary per subject
    summary_table = []
    for subject, metrics in subject_metrics.items():
        summary_table.append(
            {
                "subject": subject,
                "num_prompts": metrics["prompt_count"],
                "CLIP I": float(np.mean(metrics["clip_i"])) if metrics["clip_i"] else None,
                "CLIP T": float(np.mean(metrics["clip_t"])) if metrics["clip_t"] else None,
                "DINO": float(np.mean(metrics["dino"])) if metrics["dino"] else None,
            }
        )

    # save csv
    try:
        import pandas as pd

        csv_path = os.path.join(output_dir, "sana_ag_evaluation_summary.csv")
        df = pd.DataFrame(summary_table)
        df.to_csv(csv_path, index=False)
        print(f"\nEvaluation summary saved to {csv_path}")
    except Exception as e:
        print("Could not write CSV summary:", e)

    # global metrics
    overall = {
        "CLIP I": float(np.mean(clip_i_scores)) if clip_i_scores else None,
        "CLIP T": float(np.mean(clip_t_scores)) if clip_t_scores else None,
        "DINO": float(np.mean(dino_scores)) if dino_scores else None,
        "per_subject": summary_table,
    }

    results_path = os.path.join(output_dir, "sana_ag_results.json")
    with open(results_path, "w") as f:
        json.dump(overall, f, indent=2)
    print(f"\nFull metrics saved to {results_path}")

    print("\n=== FINAL SANA AG METRICS ===")
    for k, v in overall.items():
        if k == "per_subject":
            continue
        print(f"{k}: {v:.4f}" if v is not None else f"{k}: N/A")

    return overall


def main():
    parser = argparse.ArgumentParser(description="Sana AutoGuidance style evaluation")

    parser.add_argument(
        "--base_model",
        type=str,
        default="Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers",
        help="Sana base model id",
    )
    parser.add_argument(
        "--lora_root",
        type=str,
        required=True,
        help="Root directory containing subject specific LoRA folders",
    )
    parser.add_argument(
        "--evaluation_json",
        type=str,
        default="evaluation_prompts.json",
        help="Path to evaluation prompts json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sana_ag_outputs",
        help="Directory to save generated images and metrics",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on",
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=7.5,
        help="Classifier free guidance scale for Sana",
    )
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=1,
        help="Number of images to generate per prompt",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default="./datasets",
        help="Directory with reference images, used for CLIP I and DINO",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="Number of diffusion steps",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="Image resolution used for Sana (height and width)",
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    evaluate_ag(
        base_model=args.base_model,
        lora_root=args.lora_root,
        evaluation_json=args.evaluation_json,
        output_dir=args.output_dir,
        device=args.device,
        cfg_scale=args.cfg_scale,
        num_images_per_prompt=args.num_images_per_prompt,
        instance_data_dir=args.instance_data_dir,
        steps=args.steps,
        resolution=args.resolution,
    )


if __name__ == "__main__":
    main()
