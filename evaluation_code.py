import os
import json
import torch
import numpy as np
from typing import List, Dict, Optional
from PIL import Image
from tqdm import tqdm
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

        # Only used for OLD SD DreamBooth evaluation (not SANA)
        self.pipe = None

        # Load CLIP
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)

        # Load DINO (fallback safe)
        try:
            dino_model, _, dino_preprocess = open_clip.create_model_and_transforms(
                "dinov2_vitg14", pretrained="laion2b_s39b_b160k"
            )
        except Exception as e:
            print(f"Warning: {e}. Falling back to OpenCLIP ViT-L-14")
            dino_model, _, dino_preprocess = open_clip.create_model_and_transforms(
                "ViT-L-14", pretrained="laion2b_s32b_b82k"
            )
        self.dino_model = dino_model.to(self.device).eval()
        self.preprocess_dino = dino_preprocess

        # Load prompts
        self.prompts = self._load_prompts(json_path)

        # Load reference images (patched flexible)
        if instance_data_dir is None:
            print("\n[Evaluator] No instance_data_dir provided. CLIP-I and DINO-I disabled.")
            self.reference_images = {}
        else:
            self.reference_images = self._load_reference_images_patched()

        print("\nLoaded subjects:", list(set(self.subject_per_prompt)))
        print("Reference subjects found:", list(self.reference_images.keys()))

    # ---------------------------------------------------------
    # PROMPTS
    # ---------------------------------------------------------
    def _load_prompts(self, json_path: str) -> List[str]:
        with open(json_path, "r") as f:
            data = json.load(f)

        subjects = data["subjects"]
        templates = data["prompts"]

        prompts = []
        subject_per_prompt = []
        for s in subjects:
            for t in templates:
                prompts.append(t.replace("<subject>", s))
                subject_per_prompt.append(s)

        self.subject_per_prompt = subject_per_prompt
        return prompts

    # ---------------------------------------------------------
    # ⭐ FLEXIBLE PATCHED REFERENCE IMAGE LOADER
    # Works with ANY of these:
    # datasets/cat/*.jpg
    # datasets/cat/cat/*.jpg
    # datasets/cat/images/*.jpg
    # ---------------------------------------------------------
    def _load_reference_images_patched(self) -> Dict[str, List[Image.Image]]:
        from pathlib import Path

        print("\n[Evaluator] Loading reference images in PATCHED FLEX MODE")

        reference_images = {}
        instance_dir = Path(self.instance_data_dir)

        if not instance_dir.exists():
            print("[Evaluator] instance_data_dir does not exist!")
            return {}

        subjects = set(self.subject_per_prompt)

        for subject in subjects:
            subject_lower = subject.lower()
            collected = []

            # Case A: datasets/cat/*.jpg
            folderA = instance_dir / subject_lower
            if folderA.exists():
                for f in folderA.iterdir():
                    if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                        try:
                            collected.append(Image.open(f).convert("RGB"))
                        except:
                            pass

                if collected:
                    reference_images[subject] = collected
                    continue

            # Case B: datasets/cat/cat/*
            folderB = instance_dir / subject_lower / subject_lower
            if folderB.exists():
                for f in folderB.iterdir():
                    if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                        try:
                            collected.append(Image.open(f).convert("RGB"))
                        except:
                            pass

                if collected:
                    reference_images[subject] = collected
                    continue

            print(f"[Evaluator] No reference images found for subject '{subject}'")

        return reference_images

    # ---------------------------------------------------------
    # ENCODERS
    # ---------------------------------------------------------
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

    def _cosine(self, a, b):
        return (a * b).sum(dim=-1)

    # ---------------------------------------------------------
    # METRIC COMPUTATION
    # ---------------------------------------------------------
    def compute_metrics(self, generated_image: Image.Image, subject: str, text_feature):
        clip_t = None
        clip_i = None
        dino_i = None

        img_clip = self._encode_images_clip([generated_image])
        img_dino = self._encode_images_dino([generated_image])

        # CLIP-T
        clip_t = self._cosine(img_clip, text_feature).mean().item()

        # CLIP-I / DINO-I if reference available
        if subject in self.reference_images and len(self.reference_images[subject]) > 0:
            ref_imgs = self.reference_images[subject]

            ref_clip = self._encode_images_clip(ref_imgs)
            ref_dino = self._encode_images_dino(ref_imgs)

            # CLIP-I
            clip_i = (
                torch.cosine_similarity(img_clip, ref_clip).max().item()
            )

            # DINO-I
            dino_i = (
                torch.cosine_similarity(img_dino, ref_dino).max().item()
            )

        return clip_t, clip_i, dino_i
