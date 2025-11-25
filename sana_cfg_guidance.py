import json
import torch
import argparse
from pathlib import Path
from diffusers import SanaPipeline
from evaluation_code import Evaluator
from PIL import Image


def load_sana(base_model, lora_root, subject, dtype=torch.float32):
    print("Loading LoRA for subject:", subject)

    pipe = SanaPipeline.from_pretrained(
        base_model,
        torch_dtype=dtype,
    )

    subject_path = Path(lora_root) / subject
    if not subject_path.exists():
        raise ValueError(f"LoRA folder not found for subject {subject}")

    # load all LoRA weights found in subject folder
    for f in subject_path.iterdir():
        if f.suffix in [".safetensors", ".bin"]:
            print("Loading:", f)
            pipe.load_lora_weights(str(f), adapter_name="subject")

    return pipe.to("cuda")


def generate_sana_images(pipe, prompt, steps, guidance_scale, resolution):
    out = pipe(
        prompt,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        height=resolution,
        width=resolution,
    )
    return out.images[0]


def evaluate_cfg_sana(
    base_model,
    lora_root,
    evaluation_json,
    output_dir,
    guidance_scale,
    resolution,
    steps,
):

    with open(evaluation_json, "r") as f:
        data = json.load(f)

    subjects = data["subjects"]
    prompts = data["prompts"]

    subject = subjects[0]  # assumed single subject evaluation

    pipe = load_sana(base_model, lora_root, subject)

    evaluator = Evaluator(
        model_path="",
        json_path=evaluation_json,
        output_dir=output_dir,
        num_images_per_prompt=1,
        instance_data_dir="./datasets",
    )

    text_feats = evaluator._encode_text(evaluator.prompts)

    clip_t_scores = []
    clip_i_scores = []
    dino_scores = []

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for idx, prompt in enumerate(evaluator.prompts):
        print(f"[{idx}] {prompt}")

        image = generate_sana_images(
            pipe,
            prompt,
            steps=steps,
            guidance_scale=guidance_scale,
            resolution=resolution,
        )

        save_path = out_path / f"{idx:04d}.png"
        image.save(save_path)

        subject = evaluator.subject_per_prompt[idx]
        t_feat = text_feats[idx].unsqueeze(0)

        clip_t, clip_i, dino_i = evaluator.compute_metrics(image, subject, t_feat)

        print("  CLIP-T:", round(clip_t, 4))

        clip_t_scores.append(clip_t)
        if clip_i is not None:
            clip_i_scores.append(clip_i)
        if dino_i is not None:
            dino_scores.append(dino_i)

    print("\n=== FINAL SANA CFG METRICS ===\n")
    print("Subject:", subjects[0])
    print("  Mean CLIP-T :", sum(clip_t_scores) / len(clip_t_scores))

    if clip_i_scores:
        print("  Mean CLIP-I :", sum(clip_i_scores) / len(clip_i_scores))
    else:
        print("  Mean CLIP-I : N/A")

    if dino_scores:
        print("  Mean DINO-I :", sum(dino_scores) / len(dino_scores))
    else:
        print("  Mean DINO-I : N/A")

    print("\nSANA CFG evaluation complete.\n")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--lora_root", type=str, required=True)
    parser.add_argument("--evaluation_json", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=20)

    args = parser.parse_args()

    evaluate_cfg_sana(
        base_model=args.base_model,
        lora_root=args.lora_root,
        evaluation_json=args.evaluation_json,
        output_dir=args.output_dir,
        guidance_scale=args.guidance_scale,
        resolution=args.resolution,
        steps=args.steps,
    )


if __name__ == "__main__":
    main()
