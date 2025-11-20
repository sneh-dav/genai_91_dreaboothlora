import os
import json
import torch
import numpy as np
from typing import List, Dict, Optional
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from diffusers import StableDiffusionPipeline
import torch.nn.functional as F
import clip
import open_clip


class Evaluator:

    def __init__(
        self,
        model_path: str,
        json_path: str,
        output_dir: str = "outputs",
        device: Optional[str] = None,
        num_images_per_prompt: int = 1,
        instance_data_dir: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = output_dir
        self.num_images = num_images_per_prompt
        self.instance_data_dir = instance_data_dir
        os.makedirs(output_dir, exist_ok=True)

        if model_path:
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_path, torch_dtype=torch.float16
            ).to(self.device)
        else:
            self.pipe = None

        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)

        try:
            dino_model, _, dino_preprocess = open_clip.create_model_and_transforms(
                "dinov2_vitg14", pretrained="laion2b_s39b_b160k"
            )
        except Exception as e:
            print(
                f"Warning: {e}. Falling back to OpenCLIP ViT-L-14 (laion2b_s32b_b82k) for the DINO metric."
            )
            dino_model, _, dino_preprocess = open_clip.create_model_and_transforms(
                "ViT-L-14", pretrained="laion2b_s32b_b82k"
            )
        self.dino_model = dino_model.to(self.device).eval()
        self.preprocess_dino = dino_preprocess

        self.prompts = self._load_prompts(json_path)
        self.reference_images = self._load_reference_images() if instance_data_dir else None

    def _load_prompts(self, json_path: str) -> List[str]:
        with open(json_path, "r") as f:
            data = json.load(f)
        subjects = data["subjects"]
        templates = data["prompts"]
        prompts = [p.replace("<subject>", s) for s in subjects for p in templates]
        self.subject_per_prompt = [s for s in subjects for _ in templates]
        return prompts

    def _normalize_folder_to_subject(self, folder_name: str, subjects: List[str]) -> Optional[str]:
        folder_lower = folder_name.lower()
        
        for subject in subjects:
            subject_lower = subject.lower()
            
            if folder_lower == subject_lower:
                return subject
            
            if folder_lower == subject_lower.replace(" ", "_"):
                return subject
            
            if folder_lower == subject_lower.replace(" ", "-"):
                return subject
            
            if folder_lower.startswith(subject_lower):
                remaining = folder_lower[len(subject_lower):]
                if remaining and (remaining[0].isdigit() or remaining[0].isalpha()):
                    return subject
        
        folder_with_spaces = folder_name.replace("_", " ").replace("-", " ")
        if folder_with_spaces.lower() in [s.lower() for s in subjects]:
            return next(s for s in subjects if s.lower() == folder_with_spaces.lower())
        
        return None

    def _load_reference_images(self) -> Dict[str, List[Image.Image]]:
        reference_images = {}
        if not self.instance_data_dir or not os.path.exists(self.instance_data_dir):
            return reference_images
        
        from pathlib import Path
        instance_dir = Path(self.instance_data_dir)
        
        subjects = list(set(self.subject_per_prompt))
        
        for folder_path in instance_dir.iterdir():
            if folder_path.is_dir():
                folder_name = folder_path.name
                normalized_subject = self._normalize_folder_to_subject(folder_name, subjects)
                if normalized_subject:
                    image_files = sorted([f for f in folder_path.iterdir() 
                                        if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']])
                    ref_images = []
                    for img_path in image_files:
                        try:
                            img = Image.open(img_path).convert('RGB')
                            ref_images.append(img)
                        except Exception as e:
                            print(f"Warning: Could not load {img_path}: {e}")
                    
                    if ref_images:
                        if normalized_subject not in reference_images:
                            reference_images[normalized_subject] = []
                        reference_images[normalized_subject].extend(ref_images)
        
        return reference_images

    def _encode_text(self, texts: List[str]) -> torch.Tensor:
        tokens = clip.tokenize(texts).to(self.device)
        with torch.no_grad():
            feats = self.clip_model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

    def _encode_images_clip(self, images: List[Image.Image]) -> torch.Tensor:
        imgs = torch.stack([self.clip_preprocess(img) for img in images]).to(self.device)
        with torch.no_grad():
            feats = self.clip_model.encode_image(imgs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

    def _encode_images_dino(self, images: List[Image.Image]) -> torch.Tensor:
        imgs = torch.stack([self.preprocess_dino(img) for img in images]).to(self.device)
        with torch.no_grad():
            feats = self.dino_model.encode_image(imgs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

    @staticmethod
    def _cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a * b).sum(dim=-1)

    def _generate_images(self, prompt: str) -> List[Image.Image]:
        images = []
        for _ in range(self.num_images):
            img = self.pipe(prompt, guidance_scale=7.5).images[0]
            images.append(img)
        return images

    def _find_lora_path(self, subject: str, lora_base_dir: str, checkpoint_type: str = "full") -> Optional[str]:
        from pathlib import Path
        base_dir = Path(lora_base_dir)
        
        subject_variants = [
            subject,
            subject.replace(" ", "_"),
            subject.replace(" ", "-"),
        ]
        
        full_folder_path = None
        
        for variant in subject_variants:
            full_path = base_dir / variant
            if full_path.exists() and full_path.is_dir():
                full_folder_path = full_path
                break
        
        if full_folder_path is None:
            for folder_path in base_dir.iterdir():
                if folder_path.is_dir():
                    normalized_subject = self._normalize_folder_to_subject(folder_path.name, [subject])
                    if normalized_subject == subject:
                        full_folder_path = folder_path
                        break
        
        if full_folder_path is None:
            return None
        
        if checkpoint_type == "full":
            return str(full_folder_path)
        else:
            weak_path = full_folder_path / checkpoint_type
            if weak_path.exists() and weak_path.is_dir():
                return str(weak_path)
            return None

    def evaluate(self) -> Dict[str, float]:
        clip_i_scores, clip_t_scores, dino_scores = [], [], []
        text_features = self._encode_text(self.prompts)

        for idx, prompt in enumerate(tqdm(self.prompts, desc="Evaluating")):
            images = self._generate_images(prompt)
            img_clip = self._encode_images_clip(images)
            img_dino = self._encode_images_dino(images)
            t_feat = text_features[idx].unsqueeze(0).repeat(self.num_images, 1)

            clip_t = self._cosine(img_clip, t_feat).mean().item()
            clip_t_scores.append(clip_t)

            subject = self.subject_per_prompt[idx]
            if self.reference_images and subject in self.reference_images and len(self.reference_images[subject]) > 0:
                ref_images = self.reference_images[subject]
                ref_clip = self._encode_images_clip(ref_images)
                ref_dino = self._encode_images_dino(ref_images)
                
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

            images[0].save(os.path.join(self.output_dir, f"{idx:04d}.png"))

        return {
            "CLIP-I": float(np.mean(clip_i_scores)) if clip_i_scores else None,
            "CLIP-T": float(np.mean(clip_t_scores)),
            "DINO": float(np.mean(dino_scores)) if dino_scores else None,
        }


if __name__ == "__main__":
    evaluator = Evaluator(
        model_path="",
        json_path="",
        num_images_per_prompt=1,
    )
    scores = evaluator.evaluate()
    print("\n=== Mean Evaluation Metrics ===")
    for k, v in scores.items():
        print(f"{k}: {v:.4f}" if v is not None else f"{k}: N/A")
