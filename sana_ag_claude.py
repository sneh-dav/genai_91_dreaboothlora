import json
import torch
import argparse
from pathlib import Path
from diffusers import SanaPipeline
from evaluation_code import Evaluator
from PIL import Image
import os
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from golden_search import golden_section_search
import gc


def generate_sana_images_ag(
    fine_pipe,
    weak_pipe,
    prompt,
    steps,
    guidance_scale,
    cfg_scale,
    resolution,
):
    """
    Generate images using AutoGuidance for SANA.
    
    Instead of manually calling transformer, we use the pipeline's __call__ method
    to get the noise predictions properly, then combine them with AG formula.
    """
    # Set inference steps
    fine_pipe.scheduler.set_timesteps(steps)
    weak_pipe.scheduler.set_timesteps(steps)
    
    # Generate with weak model (CFG)
    weak_output = weak_pipe(
        prompt=prompt,
        num_inference_steps=steps,
        guidance_scale=cfg_scale,
        height=resolution,
        width=resolution,
        output_type="latent",  # Get latents, not decoded image
        return_dict=True,
    )
    weak_latents = weak_output.images
    
    # Generate with fine model (CFG)
    fine_output = fine_pipe(
        prompt=prompt,
        num_inference_steps=steps,
        guidance_scale=cfg_scale,
        height=resolution,
        width=resolution,
        output_type="latent",
        return_dict=True,
    )
    fine_latents = fine_output.images
    
    # Apply AutoGuidance in latent space
    # AG formula: final = weak + lambda * (fine - weak)
    ag_latents = weak_latents + guidance_scale * (fine_latents - weak_latents)
    
    # Decode to image
    ag_latents = ag_latents / fine_pipe.vae.config.scaling_factor
    ag_latents = ag_latents.to(dtype=fine_pipe.vae.dtype)  # Match VAE dtype
    with torch.no_grad():
        image = fine_pipe.vae.decode(ag_latents, return_dict=False)[0]
    
    image = fine_pipe.image_processor.postprocess(image, output_type="pil")
    return image[0]


def evaluate_ag_sana(
    base_model,
    lora_root,
    evaluation_json,
    output_dir,
    guidance_scale,
    cfg_scale,
    resolution,
    steps,
    instance_data_dir=None,
):
    with open(evaluation_json, "r") as f:
        data = json.load(f)

    subjects = data["subjects"]
    
    evaluator = Evaluator(
        model_path="",
        json_path=evaluation_json,
        output_dir=output_dir,
        num_images_per_prompt=1,
        instance_data_dir=instance_data_dir,
    )

    text_feats = evaluator._encode_text(evaluator.prompts)

    clip_t_scores = []
    clip_i_scores = []
    dino_scores = []

    subject_metrics = {}
    current_fine_pipe = None
    current_weak_pipe = None
    current_subject = None

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for idx, prompt in enumerate(tqdm(evaluator.prompts, desc="Evaluating Sana AG")):
        subject = evaluator.subject_per_prompt[idx]
        
        # Load new pipelines if subject changed
        if subject != current_subject:
            print(f"\nSwitching to subject: {subject}")
            
            # Clear previous pipelines
            if current_fine_pipe is not None:
                del current_fine_pipe
                del current_weak_pipe
                torch.cuda.empty_cache()
                gc.collect()
            
            subject_path = Path(lora_root) / subject
            if not subject_path.exists():
                print(f"Warning: LoRA folder not found for subject '{subject}'")
                continue
            
            # Load FINE pipeline
            print(f"Loading Sana base model: {base_model}")
            current_fine_pipe = SanaPipeline.from_pretrained(
                base_model,
                torch_dtype=torch.float16,
            ).to("cuda")
            
            fine_lora_file = None
            for f in subject_path.iterdir():
                if f.suffix in [".safetensors", ".bin"]:
                    fine_lora_file = str(f)
                    break
            
            if fine_lora_file is None:
                print(f"Warning: No LoRA weights found for {subject}")
                continue
            
            print(f"Loading fine LoRA: {fine_lora_file}")
            current_fine_pipe.load_lora_weights(fine_lora_file, adapter_name="subject")
            
            # Load WEAK pipeline
            weak_path = subject_path / "weak"
            if weak_path.exists() and weak_path.is_dir():
                print(f"Loading Sana base model for weak: {base_model}")
                current_weak_pipe = SanaPipeline.from_pretrained(
                    base_model,
                    torch_dtype=torch.float16,
                ).to("cuda")
                
                weak_lora_file = None
                for f in weak_path.iterdir():
                    if f.suffix in [".safetensors", ".bin"]:
                        weak_lora_file = str(f)
                        break
                
                print(f"Loading weak LoRA: {weak_lora_file}")
                current_weak_pipe.load_lora_weights(weak_lora_file, adapter_name="subject")
            else:
                print("No weak checkpoint found, using pretrained as weak model")
                current_weak_pipe = SanaPipeline.from_pretrained(
                    base_model,
                    torch_dtype=torch.float16,
                ).to("cuda")
            
            current_subject = subject
            
            if subject not in subject_metrics:
                subject_metrics[subject] = {
                    "clip_i": [],
                    "clip_t": [],
                    "dino": [],
                    "prompt_count": 0
                }

        print(f"[{idx}] {prompt}")

        # Generate image with AG
        image = generate_sana_images_ag(
            current_fine_pipe,
            current_weak_pipe,
            prompt,
            steps=steps,
            guidance_scale=guidance_scale,
            cfg_scale=cfg_scale,
            resolution=resolution,
        )

        # Save image
        subject_dir = out_path / subject.replace(" ", "_")
        subject_dir.mkdir(parents=True, exist_ok=True)
        prompt_safe = prompt.replace("/", "_").replace("\\", "_")[:100]
        save_path = subject_dir / f"{idx:04d}_{prompt_safe}.png"
        image.save(save_path)

        # Encode generated image
        img_clip = evaluator._encode_images_clip([image])
        img_dino = evaluator._encode_images_dino([image])
        
        # CLIP-T score
        t_feat = text_feats[idx].unsqueeze(0)
        clip_t = evaluator._cosine(img_clip, t_feat).mean().item()

        print(f"  CLIP-T: {round(clip_t, 4)}")

        clip_t_scores.append(clip_t)
        subject_metrics[subject]["clip_t"].append(clip_t)
        subject_metrics[subject]["prompt_count"] += 1
        
        # CLIP-I and DINO-I scores
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

    # Cleanup
    if current_fine_pipe is not None:
        del current_fine_pipe
        del current_weak_pipe
        torch.cuda.empty_cache()
        gc.collect()

    # Print summary
    print("\n=== FINAL SANA AG METRICS ===\n")
    
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

    print("Overall Metrics:")
    print(f"  Mean CLIP-T: {sum(clip_t_scores) / len(clip_t_scores):.4f}")
    if clip_i_scores:
        print(f"  Mean CLIP-I: {sum(clip_i_scores) / len(clip_i_scores):.4f}")
    else:
        print("  Mean CLIP-I: N/A")
    if dino_scores:
        print(f"  Mean DINO-I: {sum(dino_scores) / len(dino_scores):.4f}")
    else:
        print("  Mean DINO-I: N/A")

    print("\nSANA AG evaluation complete.\n")
    
    return {
        "CLIP-I": float(np.mean(clip_i_scores)) if clip_i_scores else None,
        "CLIP-T": float(np.mean(clip_t_scores)),
        "DINO": float(np.mean(dino_scores)) if dino_scores else None,
        "per_subject": summary_table,
    }


def optimize_ag_sana(
    base_model,
    lora_root,
    evaluation_json,
    output_dir,
    cfg_scale,
    lambda_range,
    resolution,
    steps,
    instance_data_dir=None,
):
    """Optimize AG guidance scale using golden section search"""
    
    def objective(lambda_val):
        scores = evaluate_ag_sana(
            base_model=base_model,
            lora_root=lora_root,
            evaluation_json=evaluation_json,
            output_dir=os.path.join(output_dir, f"temp_ag_lambda_{lambda_val}"),
            guidance_scale=lambda_val,
            cfg_scale=cfg_scale,
            resolution=resolution,
            steps=steps,
            instance_data_dir=instance_data_dir,
        )
        dino_score = scores.get("DINO", 0) or 0
        return -dino_score

    best_lambda, best_score = golden_section_search(
        objective, lambda_range[0], lambda_range[1]
    )

    print(f"\n=== Optimization Complete ===")
    print(f"Best Lambda: {best_lambda:.4f}")
    print(f"Best Score: {-best_score:.4f}")

    best_scores = evaluate_ag_sana(
        base_model=base_model,
        lora_root=lora_root,
        evaluation_json=evaluation_json,
        output_dir=os.path.join(output_dir, "ag_optimized"),
        guidance_scale=best_lambda,
        cfg_scale=cfg_scale,
        resolution=resolution,
        steps=steps,
        instance_data_dir=instance_data_dir,
    )

    return best_lambda, best_scores


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--lora_root", type=str, required=True)
    parser.add_argument("--evaluation_json", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--guidance_scale", type=float, default=2.0)
    parser.add_argument("--cfg_scale", type=float, default=7.5)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=20)
    
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--lambda_range", type=float, nargs=2, default=[-10.0, 10.0])
    parser.add_argument("--instance_data_dir", type=str, default=None)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.optimize:
        print("Optimizing SANA AG guidance scale (lambda)...")
        best_lambda, scores = optimize_ag_sana(
            base_model=args.base_model,
            lora_root=args.lora_root,
            evaluation_json=args.evaluation_json,
            output_dir=args.output_dir,
            cfg_scale=args.cfg_scale,
            lambda_range=tuple(args.lambda_range),
            resolution=args.resolution,
            steps=args.steps,
            instance_data_dir=args.instance_data_dir,
        )
        
        print(f"\n=== SANA AutoGuidance Optimization Results ===")
        print(f"Best Lambda: {best_lambda:.4f}")
        print(f"CFG Scale: {args.cfg_scale:.4f}")
        print(f"Scores:")
        for metric, score in scores.items():
            if isinstance(score, (int, float)):
                print(f"  {metric}: {score:.4f}")

        results = {
            "method": "sana_ag",
            "lambda": best_lambda,
            "cfg_scale": args.cfg_scale,
            "scores": scores,
        }
        results_path = os.path.join(args.output_dir, "sana_ag_optimization_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_path}")
    else:
        print(f"Evaluating SANA AG with lambda={args.guidance_scale}, cfg={args.cfg_scale}...")
        scores = evaluate_ag_sana(
            base_model=args.base_model,
            lora_root=args.lora_root,
            evaluation_json=args.evaluation_json,
            output_dir=args.output_dir,
            guidance_scale=args.guidance_scale,
            cfg_scale=args.cfg_scale,
            resolution=args.resolution,
            steps=args.steps,
            instance_data_dir=args.instance_data_dir,
        )
        
        print("\n=== SANA AutoGuidance Evaluation Results ===")
        for metric, score in scores.items():
            if isinstance(score, (int, float)):
                print(f"{metric}: {score:.4f}")


if __name__ == "__main__":
    main()