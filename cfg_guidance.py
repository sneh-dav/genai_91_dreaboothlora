import argparse
import os
import json
import torch
import numpy as np
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline, DDIMScheduler
from guidance_methods import GuidancePipeline
from evaluation_code import Evaluator
from golden_search import golden_section_search


def load_pretrained_pipeline(model_path: str, device: str = "cuda"):
    pipeline = StableDiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(device)
    return pipeline


def load_finetuned_pipeline(
    pretrained_model_path: str,
    lora_path: str,
    device: str = "cuda",
):
    pipeline = GuidancePipeline.from_pretrained(
        pretrained_model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.unet.load_attn_procs(lora_path)
    pipeline = pipeline.to(device)
    return pipeline


def evaluate_cfg(
    pretrained_model_path: str,
    lora_base_dir: str,
    evaluation_prompts_path: str,
    output_dir: str,
    device: str = "cuda",
    guidance_scale: float = 7.5,
    num_images_per_prompt: int = 1,
    instance_data_dir: str = None,
):
    import json
    from pathlib import Path
    
    with open(evaluation_prompts_path, "r") as f:
        data = json.load(f)
    subjects = data["subjects"]
    
    evaluator = Evaluator(
        model_path=None,
        json_path=evaluation_prompts_path,
        output_dir=output_dir,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        instance_data_dir=instance_data_dir,
    )
    
    clip_i_scores, clip_t_scores, dino_scores = [], [], []
    text_features = evaluator._encode_text(evaluator.prompts)
    
    subject_metrics = {}
    current_pipeline = None
    current_subject = None
    
    from tqdm import tqdm
    for idx, prompt in enumerate(tqdm(evaluator.prompts, desc="Evaluating")):
        subject = evaluator.subject_per_prompt[idx]
        
        if subject != current_subject:
            lora_path = evaluator._find_lora_path(subject, lora_base_dir, checkpoint_type="full")
            if not lora_path:
                print(f"Warning: No fully trained LoRA model found for subject '{subject}' in {lora_base_dir}, skipping...")
                continue
            
            current_pipeline = load_finetuned_pipeline(
                pretrained_model_path, lora_path, device
            )
            current_subject = subject
            
            if subject not in subject_metrics:
                subject_metrics[subject] = {
                    "clip_i": [],
                    "clip_t": [],
                    "dino": [],
                    "prompt_count": 0
                }
        
        def generate_images(p):
            result = current_pipeline.generate_with_guidance(
                prompt=p,
                guidance_method="cfg",
                guidance_scale=guidance_scale,
                num_inference_steps=50,
                num_images_per_prompt=num_images_per_prompt,
            )
            return result.images
        
        images = generate_images(prompt)
        img_clip = evaluator._encode_images_clip(images)
        img_dino = evaluator._encode_images_dino(images)
        t_feat = text_features[idx].unsqueeze(0).repeat(num_images_per_prompt, 1)
        
        clip_t = evaluator._cosine(img_clip, t_feat).mean().item()
        clip_t_scores.append(clip_t)
        subject_metrics[subject]["clip_t"].append(clip_t)
        subject_metrics[subject]["prompt_count"] += 1
        
        subject_dir = os.path.join(output_dir, subject.replace(" ", "_"))
        os.makedirs(subject_dir, exist_ok=True)
        prompt_safe = prompt.replace("/", "_").replace("\\", "_")[:100]
        image_path = os.path.join(subject_dir, f"{idx:04d}_{prompt_safe}.png")
        images[0].save(image_path)
        
        if evaluator.reference_images and subject in evaluator.reference_images and len(evaluator.reference_images[subject]) > 0:
            ref_images = evaluator.reference_images[subject]
            ref_clip = evaluator._encode_images_clip(ref_images)
            ref_dino = evaluator._encode_images_dino(ref_images)
            
            clip_i_sims = []
            for gen_feat in img_clip:
                sims = F.cosine_similarity(
                    gen_feat.unsqueeze(0),
                    ref_clip,
                    dim=1
                )
                clip_i_sims.append(sims.max().item())
            clip_i = np.mean(clip_i_sims)
            clip_i_scores.append(clip_i)
            subject_metrics[subject]["clip_i"].append(clip_i)
            
            dino_sims = []
            for gen_feat in img_dino:
                sims = F.cosine_similarity(
                    gen_feat.unsqueeze(0),
                    ref_dino,
                    dim=1
                )
                dino_sims.append(sims.max().item())
            dino_i = np.mean(dino_sims)
            dino_scores.append(dino_i)
            subject_metrics[subject]["dino"].append(dino_i)
    
    summary_table = []
    for subject, metrics in subject_metrics.items():
        summary_table.append({
            "subject": subject,
            "num_prompts": metrics["prompt_count"],
            "CLIP-I": float(np.mean(metrics["clip_i"])) if metrics["clip_i"] else None,
            "CLIP-T": float(np.mean(metrics["clip_t"])) if metrics["clip_t"] else None,
            "DINO": float(np.mean(metrics["dino"])) if metrics["dino"] else None,
        })
    
    import pandas as pd
    df = pd.DataFrame(summary_table)
    csv_path = os.path.join(output_dir, "evaluation_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nEvaluation summary saved to {csv_path}")
    
    return {
        "CLIP-I": float(np.mean(clip_i_scores)) if clip_i_scores else None,
        "CLIP-T": float(np.mean(clip_t_scores)),
        "DINO": float(np.mean(dino_scores)) if dino_scores else None,
        "per_subject": summary_table,
    }


def optimize_cfg(
    pretrained_model_path: str,
    lora_base_dir: str,
    evaluation_prompts_path: str,
    output_dir: str,
    device: str = "cuda",
    lambda_range: tuple = (-10.0, 10.0),
    num_images_per_prompt: int = 1,
    instance_data_dir: str = None,
):
    def objective(lambda_val):
        scores = evaluate_cfg(
            pretrained_model_path=pretrained_model_path,
            lora_base_dir=lora_base_dir,
            evaluation_prompts_path=evaluation_prompts_path,
            output_dir=os.path.join(output_dir, f"temp_cfg_{lambda_val}"),
            device=device,
            guidance_scale=lambda_val,
            num_images_per_prompt=num_images_per_prompt,
            instance_data_dir=instance_data_dir,
        )
        dino_score = scores.get("DINO", 0)
        return -dino_score

    best_lambda, best_score = golden_section_search(
        objective, lambda_range[0], lambda_range[1]
    )

    best_scores = evaluate_cfg(
        pretrained_model_path=pretrained_model_path,
        lora_base_dir=lora_base_dir,
        evaluation_prompts_path=evaluation_prompts_path,
        output_dir=os.path.join(output_dir, "cfg_optimized"),
        device=device,
        guidance_scale=best_lambda,
        num_images_per_prompt=num_images_per_prompt,
        instance_data_dir=instance_data_dir,
    )

    return best_lambda, best_scores


def main():
    parser = argparse.ArgumentParser(description="CFG Guidance Evaluation")
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Path to pretrained model",
    )
    parser.add_argument(
        "--lora_base_dir",
        type=str,
        required=True,
        help="Base directory containing subject-specific LoRA model folders",
    )
    parser.add_argument(
        "--evaluation_prompts_path",
        type=str,
        default="evaluation_prompts.json",
        help="Path to evaluation prompts JSON",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="cfg_outputs",
        help="Output directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Guidance scale",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Optimize guidance scale using golden search",
    )
    parser.add_argument(
        "--lambda_range",
        type=float,
        nargs=2,
        default=[-10.0, 10.0],
        help="Range for lambda optimization",
    )
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=1,
        help="Number of images per prompt",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        help="Directory containing reference images for DINO score calculation",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.optimize:
        print("Optimizing CFG guidance scale...")
        best_lambda, scores = optimize_cfg(
            pretrained_model_path=args.pretrained_model_path,
            lora_base_dir=args.lora_base_dir,
            evaluation_prompts_path=args.evaluation_prompts_path,
            output_dir=args.output_dir,
            device=args.device,
            lambda_range=tuple(args.lambda_range),
            num_images_per_prompt=args.num_images_per_prompt,
            instance_data_dir=args.instance_data_dir,
        )
        print(f"\n=== CFG Optimization Results ===")
        print(f"Best Lambda: {best_lambda:.4f}")
        print(f"Scores:")
        for metric, score in scores.items():
            if score is not None:
                print(f"  {metric}: {score:.4f}")

        results = {
            "method": "cfg",
            "lambda": best_lambda,
            "scores": scores,
        }
        results_path = os.path.join(args.output_dir, "cfg_optimization_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_path}")
    else:
        print(f"Evaluating CFG with guidance_scale={args.guidance_scale}...")
        scores = evaluate_cfg(
            pretrained_model_path=args.pretrained_model_path,
            lora_base_dir=args.lora_base_dir,
            evaluation_prompts_path=args.evaluation_prompts_path,
            output_dir=args.output_dir,
            device=args.device,
            guidance_scale=args.guidance_scale,
            num_images_per_prompt=args.num_images_per_prompt,
            instance_data_dir=args.instance_data_dir,
        )
        print("\n=== CFG Evaluation Results ===")
        for metric, score in scores.items():
            if score is not None:
                print(f"{metric}: {score:.4f}")


if __name__ == "__main__":
    main()

