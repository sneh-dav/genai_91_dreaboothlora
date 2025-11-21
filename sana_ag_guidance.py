import argparse
import os
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image

from diffusers import SanaPipeline

from evaluation_code import Evaluator


def load_sana_lora(base_model: str, lora_dir: str, device: str, dtype: torch.dtype):
    """
    Load a SanaPipeline with LoRA weights from a directory.
    This matches the way you loaded LoRA in the Sana CFG script.
    """
    print(f"Loading Sana base model: {base_model}")
    pipe = SanaPipeline.from_pretrained(
        base_model,
        torch_dtype=dtype,
    )

    lora_dir_path = Path(lora_dir)
    if not lora_dir_path.exists():
        raise ValueError(f"LoRA directory not found: {lora_dir}")

    # find first safetensors or bin file
    lora_files = list(lora_dir_path.glob("*.safetensors")) + list(
        lora_dir_path.glob("*.bin")
    )
    if len(lora_files) == 0:
        raise ValueError(f"No LoRA weights found in {lora_dir}")

    for f in lora_files:
        print(f"Loading LoRA: {f}")
        # adapter_name is optional, we can reuse "subject" for all
        pipe.load_lora_weights(str(f), adapter_name="subject")

    pipe.to(device)
    return pipe


def ag_blend_images(weak_img: Image.Image, strong_img: Image.Image, guidance_scale: float) -> Image.Image:
    """
    Simple AutoGuidance style blending:
    guidance_scale controls how much we trust the strong model.

    alpha in [0.5, 1)
    guidance_scale = 0   => alpha ~ 0.5 (equal mix)
    guidance_scale large => alpha -> 1 (mostly strong)
    """
    alpha = 0.5 + 0.5 * np.tanh(guidance_scale / 5.0)
    return Image.blend(weak_img, strong_img, float(alpha))


def evaluate_ag(
    base_model: str,
    lora_root: str,
    evaluation_json: str,
    output_dir: str,
    device: str = "cuda",
    guidance_scale: float = 2.0,
    cfg_scale: float = 7.5,
    num_images_per_prompt: int = 1,
    instance_data_dir: str | None = None,
    resolution: int = 1024,
    steps: int = 20,
    dtype: torch.dtype = torch.float32,
):
    os.makedirs(output_dir, exist_ok=True)

    evaluator = Evaluator(
        model_path=None,
        json_path=evaluation_json,
        output_dir=output_dir,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        instance_data_dir=instance_data_dir,
    )

    print("Loaded subjects:", evaluator.subject_per_prompt[: len(set(evaluator.subject_per_prompt))])

    clip_i_scores, clip_t_scores, dino_scores = [], [], []
    text_features = evaluator._encode_text(evaluator.prompts)

    subject_metrics: dict[str, dict] = {}
    current_subject = None
    strong_pipe: SanaPipeline | None = None
    weak_pipe: SanaPipeline | None = None

    for idx, prompt in enumerate(tqdm(evaluator.prompts, desc="Evaluating Sana AG")):
        subject = evaluator.subject_per_prompt[idx]

        # load strong and weak pipelines when subject changes
        if subject != current_subject:
            print(f"\nSwitching to subject: {subject}")

            subject_dir = Path(lora_root) / subject
            if not subject_dir.exists():
                print(f"Warning: no LoRA folder found for subject {subject} in {lora_root}, skipping prompts")
                continue

            strong_dir = subject_dir
            weak_dir = subject_dir / "weak"

            strong_pipe = load_sana_lora(base_model, str(strong_dir), device, dtype)

            if weak_dir.exists():
                weak_pipe = load_sana_lora(base_model, str(weak_dir), device, dtype)
            else:
                print(f"Warning: weak LoRA not found for subject {subject}, using base strong pipe as weak")
                weak_pipe = strong_pipe

            current_subject = subject

            if subject not in subject_metrics:
                subject_metrics[subject] = {
                    "clip_i": [],
                    "clip_t": [],
                    "dino": [],
                    "prompt_count": 0,
                }

        # safety check
        if strong_pipe is None or weak_pipe is None:
            print(f"Subject {subject} has no loaded pipelines, skipping")
            continue

        # generate AG images
        images = []
        for img_idx in range(num_images_per_prompt):
            # use deterministic seed per prompt image to keep noise aligned
            seed = idx * 1000 + img_idx
            generator = torch.Generator(device=device).manual_seed(seed)

            common_kwargs = dict(
                prompt=prompt,
                num_inference_steps=steps,
                guidance_scale=cfg_scale,
                height=resolution,
                width=resolution,
                num_images_per_prompt=1,
                generator=generator,
            )

            weak_out = weak_pipe(**common_kwargs)
            strong_out = strong_pipe(**common_kwargs)

            weak_img = weak_out.images[0]
            strong_img = strong_out.images[0]

            ag_img = ag_blend_images(weak_img, strong_img, guidance_scale)
            images.append(ag_img)

        # compute metrics
        img_clip = evaluator._encode_images_clip(images)
        img_dino = evaluator._encode_images_dino(images)
        t_feat = text_features[idx].unsqueeze(0).repeat(num_images_per_prompt, 1)

        clip_t = evaluator._cosine(img_clip, t_feat).mean().item()
        clip_t_scores.append(clip_t)
        subject_metrics[subject]["clip_t"].append(clip_t)
        subject_metrics[subject]["prompt_count"] += 1

        print(f"[{idx}] {prompt}")
        print(f"  CLIP T: {clip_t:.4f}")

        # save first AG image for inspection
        subject_out_dir = os.path.join(output_dir, subject.replace(" ", "_"))
        os.makedirs(subject_out_dir, exist_ok=True)
        prompt_safe = prompt.replace("/", "_").replace("\\", "_")[:100]
        image_path = os.path.join(subject_out_dir, f"{idx:04d}_{prompt_safe}.png")
        images[0].save(image_path)

        # CLIP I and DINO if reference images exist
        if evaluator.reference_images and subject in evaluator.reference_images and len(evaluator.reference_images[subject]) > 0:
            ref_images = evaluator.reference_images[subject]
            ref_clip = evaluator._encode_images_clip(ref_images)
            ref_dino = evaluator._encode_images_dino(ref_images)

            clip_i_sims = []
            for gen_feat in img_clip:
                sims = F.cosine_similarity(gen_feat.unsqueeze(0), ref_clip, dim=1)
                clip_i_sims.append(sims.max().item())
            clip_i = float(np.mean(clip_i_sims))
            clip_i_scores.append(clip_i)
            subject_metrics[subject]["clip_i"].append(clip_i)

            dino_sims = []
            for gen_feat in img_dino:
                sims = F.cosine_similarity(gen_feat.unsqueeze(0), ref_dino, dim=1)
                dino_sims.append(sims.max().item())
            dino_i = float(np.mean(dino_sims))
            dino_scores.append(dino_i)
            subject_metrics[subject]["dino"].append(dino_i)

    # per subject summary
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

    # save CSV
    try:
        import pandas as pd

        csv_path = os.path.join(output_dir, "sana_ag_evaluation_summary.csv")
        pd.DataFrame(summary_table).to_csv(csv_path, index=False)
        print(f"\nEvaluation summary saved to {csv_path}")
    except Exception as e:
        print(f"Could not save CSV summary: {e}")

    # global metrics
    results = {
        "CLIP I": float(np.mean(clip_i_scores)) if clip_i_scores else None,
        "CLIP T": float(np.mean(clip_t_scores)) if clip_t_scores else None,
        "DINO": float(np.mean(dino_scores)) if dino_scores else None,
        "per_subject": summary_table,
    }

    results_path = os.path.join(output_dir, "sana_ag_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    print("\n=== FINAL SANA AG METRICS ===")
    for k, v in results.items():
        if k == "per_subject":
            continue
        if v is None:
            print(f"  {k}: N A")
        else:
            print(f"  {k}: {v:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Sana AutoGuidance evaluation (weak strong blending)")
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
        help="Root directory containing subject LoRA folders (e g lora_models)",
    )
    parser.add_argument(
        "--evaluation_json",
        type=str,
        default="evaluation_prompts.json",
        help="Path to evaluation prompts JSON",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sana_ag_outputs",
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=2.0,
        help="AG guidance scale (controls blending between weak and strong)",
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=7.5,
        help="Classifier free guidance scale passed into SanaPipeline",
    )
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=1,
        help="Number of AG images per prompt",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        help="Reference images directory (for CLIP I and DINO)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="Image resolution for Sana",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="Number of inference steps",
    )

    args = parser.parse_args()

    evaluate_ag(
        base_model=args.base_model,
        lora_root=args.lora_root,
        evaluation_json=args.evaluation_json,
        output_dir=args.output_dir,
        device=args.device,
        guidance_scale=args.guidance_scale,
        cfg_scale=args.cfg_scale,
        num_images_per_prompt=args.num_images_per_prompt,
        instance_data_dir=args.instance_data_dir,
        resolution=args.resolution,
        steps=args.steps,
        dtype=torch.float32,
    )


if __name__ == "__main__":
    main()
