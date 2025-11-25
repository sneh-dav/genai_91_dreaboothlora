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

def generate_sana_images_bg_correct(
    fine_pipe,
    pretrained_pipe,
    prompt,
    steps,
    guidance_scale,
    omega,
    resolution,
):
    """
    Generate images using Bhavik Guidance for SANA.

    BG at each step:
        noise_pred = pretrained_uncond + lambda * (fine_cond - pretrained_uncond)
    with optional omega interpolation.
    """

    device = "cuda"

    # Set schedulers
    fine_pipe.scheduler.set_timesteps(steps)
    pretrained_pipe.scheduler.set_timesteps(steps)
    timesteps = fine_pipe.scheduler.timesteps

    # Encode prompts

    # Fine model conditional embeds
    fine_prompt_embeds = fine_pipe.encode_prompt(
        prompt=prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
    )[0]   # take text embeddings only

    # Pretrained model unconditional embeds
    pretrained_prompt_embeds = pretrained_pipe.encode_prompt(
        prompt="",
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
    )[0]

    # Latents init
    latent_channels = fine_pipe.transformer.config.in_channels
    height_latent = resolution // fine_pipe.vae_scale_factor
    width_latent = resolution // fine_pipe.vae_scale_factor

    latents = torch.randn(
        (1, latent_channels, height_latent, width_latent),
        device=device,
        dtype=fine_pipe.transformer.dtype,
    )
    latents = latents * fine_pipe.scheduler.init_noise_sigma

    # Denoising loop with BG
    for i, t in enumerate(tqdm(timesteps, desc="BG Denoising", leave=False)):
        # Make sure t is 1D tensor [batch]
        if torch.is_tensor(t):
            t_tensor = t.to(device)
            if t_tensor.dim() == 0:
                t_tensor = t_tensor[None]   # shape [1]
        else:
            t_tensor = torch.tensor([t], device=device, dtype=torch.long)

        # === Pretrained unconditional prediction ===
        latent_model_input = pretrained_pipe.scheduler.scale_model_input(latents, t_tensor)

        with torch.no_grad():
            pretrained_uncond_pred = pretrained_pipe.transformer(
                latent_model_input,
                timestep=t_tensor,
                encoder_hidden_states=pretrained_prompt_embeds,
            ).sample

        # === Fine conditional prediction ===
        latent_model_input = fine_pipe.scheduler.scale_model_input(latents, t_tensor)

        with torch.no_grad():
            fine_cond_pred = fine_pipe.transformer(
                latent_model_input,
                timestep=t_tensor,
                encoder_hidden_states=fine_prompt_embeds,
            ).sample

        # === Bhavik Guidance ===
        if omega > 0:
            interpolated_pred = (1.0 - omega) * pretrained_uncond_pred + omega * fine_cond_pred
            noise_pred = interpolated_pred + guidance_scale * (fine_cond_pred - interpolated_pred)
        else:
            noise_pred = pretrained_uncond_pred + guidance_scale * (
                fine_cond_pred - pretrained_uncond_pred
            )

        # Step scheduler (use fine scheduler since we decode with fine VAE)
        latents = fine_pipe.scheduler.step(noise_pred, t_tensor, latents, return_dict=False)[0]

    # Decode to image
    latents = latents / fine_pipe.vae.config.scaling_factor
    latents = latents.to(dtype=fine_pipe.vae.dtype)

    with torch.no_grad():
        image = fine_pipe.vae.decode(latents, return_dict=False)[0]

    image = fine_pipe.image_processor.postprocess(image, output_type="pil")
    return image[0]


def evaluate_bg_sana(
    base_model,
    lora_root,
    evaluation_json,
    output_dir,
    guidance_scale,
    omega,
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
    current_pretrained_pipe = None
    current_subject = None

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for idx, prompt in enumerate(tqdm(evaluator.prompts, desc="Evaluating Sana BG")):
        subject = evaluator.subject_per_prompt[idx]
        
        # Load new pipelines if subject changed
        if subject != current_subject:
            print(f"\nSwitching to subject: {subject}")
            
            # Clear previous pipelines
            if current_fine_pipe is not None:
                del current_fine_pipe
                del current_pretrained_pipe
                torch.cuda.empty_cache()
                gc.collect()
            
            subject_path = Path(lora_root) / subject
            if not subject_path.exists():
                print(f"Warning: LoRA folder not found for subject '{subject}'")
                continue
            
            # Load FINE pipeline
            print(f"Loading Sana fine model: {base_model}")
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
            
            # Load PRETRAINED pipeline (no LoRA)
            print(f"Loading Sana pretrained model: {base_model}")
            current_pretrained_pipe = SanaPipeline.from_pretrained(
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

        # Generate image with CORRECT BG
        image = generate_sana_images_bg_correct(
            current_fine_pipe,
            current_pretrained_pipe,
            prompt,
            steps=steps,
            guidance_scale=guidance_scale,
            omega=omega,
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
        del current_pretrained_pipe
        torch.cuda.empty_cache()
        gc.collect()

    # Print summary
    print("\n=== FINAL SANA BG METRICS ===\n")
    
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
    if clip_t_scores:
        print(f"  Mean CLIP-T: {sum(clip_t_scores) / len(clip_t_scores):.4f}")
    if clip_i_scores:
        print(f"  Mean CLIP-I: {sum(clip_i_scores) / len(clip_i_scores):.4f}")
    else:
        print("  Mean CLIP-I: N/A")
    if dino_scores:
        print(f"  Mean DINO-I: {sum(dino_scores) / len(dino_scores):.4f}")
    else:
        print("  Mean DINO-I: N/A")

    print("\nSANA BG evaluation complete.\n")
    
    return {
        "CLIP-I": float(np.mean(clip_i_scores)) if clip_i_scores else None,
        "CLIP-T": float(np.mean(clip_t_scores)) if clip_t_scores else 0.0,
        "DINO": float(np.mean(dino_scores)) if dino_scores else None,
        "per_subject": summary_table,
    }


def optimize_bg_lambda(
    base_model,
    lora_root,
    evaluation_json,
    output_dir,
    lambda_range,
    omega,
    resolution,
    steps,
    instance_data_dir=None,
):
    """Optimize BG lambda using golden section search"""
    
    def objective(lambda_val):
        scores = evaluate_bg_sana(
            base_model=base_model,
            lora_root=lora_root,
            evaluation_json=evaluation_json,
            output_dir=os.path.join(output_dir, f"temp_bg_lambda_{lambda_val}"),
            guidance_scale=lambda_val,
            omega=omega,
            resolution=resolution,
            steps=steps,
            instance_data_dir=instance_data_dir,
        )
        dino_score = scores.get("DINO", 0) or 0
        return -dino_score

    best_lambda, best_score = golden_section_search(
        objective, lambda_range[0], lambda_range[1]
    )

    best_scores = evaluate_bg_sana(
        base_model=base_model,
        lora_root=lora_root,
        evaluation_json=evaluation_json,
        output_dir=os.path.join(output_dir, "bg_optimized_lambda"),
        guidance_scale=best_lambda,
        omega=omega,
        resolution=resolution,
        steps=steps,
        instance_data_dir=instance_data_dir,
    )

    return best_lambda, best_scores


def optimize_bg_omega(
    base_model,
    lora_root,
    evaluation_json,
    output_dir,
    guidance_scale,
    omega_range,
    resolution,
    steps,
    instance_data_dir=None,
):
    """Optimize BG omega using grid search"""
    
    omega_values = [round(i * 0.1, 1) for i in range(11)]
    best_omega = 0.0
    best_dino = -float('inf')

    for omega_val in omega_values:
        print(f"\nEvaluating omega={omega_val:.1f}...")
        scores = evaluate_bg_sana(
            base_model=base_model,
            lora_root=lora_root,
            evaluation_json=evaluation_json,
            output_dir=os.path.join(output_dir, f"temp_bg_omega_{omega_val}"),
            guidance_scale=guidance_scale,
            omega=omega_val,
            resolution=resolution,
            steps=steps,
            instance_data_dir=instance_data_dir,
        )
        dino_score = scores.get("DINO", 0) or 0
        
        if dino_score > best_dino:
            best_dino = dino_score
            best_omega = omega_val

    best_scores = evaluate_bg_sana(
        base_model=base_model,
        lora_root=lora_root,
        evaluation_json=evaluation_json,
        output_dir=os.path.join(output_dir, "bg_optimized_omega"),
        guidance_scale=guidance_scale,
        omega=best_omega,
        resolution=resolution,
        steps=steps,
        instance_data_dir=instance_data_dir,
    )

    return best_omega, best_scores


def optimize_bg_both(
    base_model,
    lora_root,
    evaluation_json,
    output_dir,
    lambda_range,
    omega_range,
    resolution,
    steps,
    instance_data_dir=None,
):
    """Optimize both lambda and omega"""
    
    print("=== Step 1: Optimizing Lambda ===")
    best_lambda, _ = optimize_bg_lambda(
        base_model=base_model,
        lora_root=lora_root,
        evaluation_json=evaluation_json,
        output_dir=output_dir,
        lambda_range=lambda_range,
        omega=0.0,
        resolution=resolution,
        steps=steps,
        instance_data_dir=instance_data_dir,
    )

    print(f"\n=== Step 2: Optimizing Omega (with lambda={best_lambda:.4f}) ===")
    best_omega, best_scores = optimize_bg_omega(
        base_model=base_model,
        lora_root=lora_root,
        evaluation_json=evaluation_json,
        output_dir=output_dir,
        guidance_scale=best_lambda,
        omega_range=omega_range,
        resolution=resolution,
        steps=steps,
        instance_data_dir=instance_data_dir,
    )

    return best_lambda, best_omega, best_scores


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--lora_root", type=str, required=True)
    parser.add_argument("--evaluation_json", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--omega", type=float, default=0.0)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=20)
    
    parser.add_argument("--optimize", type=str, choices=["lambda", "omega", "both"])
    parser.add_argument("--lambda_range", type=float, nargs=2, default=[-10.0, 10.0])
    parser.add_argument("--omega_range", type=float, nargs=2, default=[0.0, 1.0])
    parser.add_argument("--instance_data_dir", type=str, default=None)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.optimize == "lambda":
        print("Optimizing SANA BG lambda...")
        best_lambda, scores = optimize_bg_lambda(
            base_model=args.base_model,
            lora_root=args.lora_root,
            evaluation_json=args.evaluation_json,
            output_dir=args.output_dir,
            lambda_range=tuple(args.lambda_range),
            omega=args.omega,
            resolution=args.resolution,
            steps=args.steps,
            instance_data_dir=args.instance_data_dir,
        )
        
        print(f"\n=== SANA BG Lambda Optimization Results ===")
        print(f"Best Lambda: {best_lambda:.4f}")
        print(f"Omega: {args.omega:.4f}")

        results = {
            "method": "sana_bg",
            "lambda": best_lambda,
            "omega": args.omega,
            "scores": scores,
        }
        results_path = os.path.join(args.output_dir, "sana_bg_lambda_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        
    elif args.optimize == "omega":
        print("Optimizing SANA BG omega...")
        best_omega, scores = optimize_bg_omega(
            base_model=args.base_model,
            lora_root=args.lora_root,
            evaluation_json=args.evaluation_json,
            output_dir=args.output_dir,
            guidance_scale=args.guidance_scale,
            omega_range=tuple(args.omega_range),
            resolution=args.resolution,
            steps=args.steps,
            instance_data_dir=args.instance_data_dir,
        )
        
        print(f"\n=== SANA BG Omega Optimization Results ===")
        print(f"Lambda: {args.guidance_scale:.4f}")
        print(f"Best Omega: {best_omega:.4f}")

        results = {
            "method": "sana_bg",
            "lambda": args.guidance_scale,
            "omega": best_omega,
            "scores": scores,
        }
        results_path = os.path.join(args.output_dir, "sana_bg_omega_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        
    elif args.optimize == "both":
        print("Optimizing SANA BG lambda and omega...")
        best_lambda, best_omega, scores = optimize_bg_both(
            base_model=args.base_model,
            lora_root=args.lora_root,
            evaluation_json=args.evaluation_json,
            output_dir=args.output_dir,
            lambda_range=tuple(args.lambda_range),
            omega_range=tuple(args.omega_range),
            resolution=args.resolution,
            steps=args.steps,
            instance_data_dir=args.instance_data_dir,
        )
        
        print(f"\n=== SANA BG Full Optimization Results ===")
        print(f"Best Lambda: {best_lambda:.4f}")
        print(f"Best Omega: {best_omega:.4f}")

        results = {
            "method": "sana_bg",
            "lambda": best_lambda,
            "omega": best_omega,
            "scores": scores,
        }
        results_path = os.path.join(args.output_dir, "sana_bg_full_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        
    else:
        print(f"Evaluating SANA BG with lambda={args.guidance_scale}, omega={args.omega}...")
        scores = evaluate_bg_sana(
            base_model=args.base_model,
            lora_root=args.lora_root,
            evaluation_json=args.evaluation_json,
            output_dir=args.output_dir,
            guidance_scale=args.guidance_scale,
            omega=args.omega,
            resolution=args.resolution,
            steps=args.steps,
            instance_data_dir=args.instance_data_dir,
        )
        
        print("\n=== SANA BG Evaluation Results ===")
        for metric, score in scores.items():
            if isinstance(score, (int, float)):
                print(f"{metric}: {score:.4f}")


if __name__ == "__main__":
    main()