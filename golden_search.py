import numpy as np
import torch
import torch.nn.functional as F
from typing import Callable, Tuple
import json
import os
from PIL import Image
from guidance_methods import GuidancePipeline
from evaluation_code import Evaluator


def golden_section_search(
    func: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-5,
    max_iter: int = 50,
) -> Tuple[float, float]:
    phi = (1 + np.sqrt(5)) / 2
    resphi = 2 - phi

    c = b - resphi * (b - a)
    d = a + resphi * (b - a)
    fc = func(c)
    fd = func(d)

    for _ in range(max_iter):
        if abs(c - d) < tol:
            break
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - resphi * (b - a)
            fc = func(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + resphi * (b - a)
            fd = func(d)

    return (a + b) / 2, func((a + b) / 2)


class ParameterOptimizer:
    def __init__(
        self,
        pipeline: GuidancePipeline,
        pretrained_pipeline: GuidancePipeline,
        evaluation_prompts_path: str,
        reference_images_dir: str,
        device: str = "cuda",
    ):
        self.pipeline = pipeline
        self.pretrained_pipeline = pretrained_pipeline
        self.pipeline.set_pretrained_models(pretrained_pipeline)
        self.evaluation_prompts_path = evaluation_prompts_path
        self.reference_images_dir = reference_images_dir
        self.device = device

    def evaluate_guidance(
        self,
        guidance_method: str,
        guidance_scale: float,
        omega: float = 0.0,
        num_samples: int = 1,
    ) -> dict:
        temp_output_dir = f"temp_eval_{guidance_method}_{guidance_scale}_{omega}"
        os.makedirs(temp_output_dir, exist_ok=True)

        evaluator = Evaluator(
            model_path=None,
            json_path=self.evaluation_prompts_path,
            output_dir=temp_output_dir,
            device=self.device,
            num_images_per_prompt=num_samples,
        )

        def generate_images(prompt):
            result = self.pipeline.generate_with_guidance(
                prompt=prompt,
                guidance_method=guidance_method,
                guidance_scale=guidance_scale,
                omega=omega,
                num_inference_steps=50,
                num_images_per_prompt=num_samples,
            )
            return result.images

        evaluator.pipe = self.pipeline
        evaluator._generate_images = generate_images

        scores = evaluator.evaluate()

        import shutil
        shutil.rmtree(temp_output_dir, ignore_errors=True)

        return scores

    def optimize_lambda(
        self,
        guidance_method: str,
        lambda_range: Tuple[float, float] = (2.5, 10.0),
        omega: float = 0.0,
    ) -> Tuple[float, dict]:
        def objective(lambda_val):
            scores = self.evaluate_guidance(
                guidance_method=guidance_method,
                guidance_scale=lambda_val,
                omega=omega,
            )
            dino_score = scores.get("DINO", 0)
            return -dino_score

        best_lambda, best_score = golden_section_search(
            objective, lambda_range[0], lambda_range[1]
        )

        best_scores = self.evaluate_guidance(
            guidance_method=guidance_method,
            guidance_scale=best_lambda,
            omega=omega,
        )

        return best_lambda, best_scores

    def optimize_omega(
        self,
        lambda_val: float,
        omega_range: Tuple[float, float] = (0.0, 1.0),
    ) -> Tuple[float, dict]:
        def objective(omega_val):
            scores = self.evaluate_guidance(
                guidance_method="bg",
                guidance_scale=lambda_val,
                omega=omega_val,
            )
            dino_score = scores.get("DINO", 0)
            return -dino_score

        best_omega, best_score = golden_section_search(
            objective, omega_range[0], omega_range[1]
        )

        best_scores = self.evaluate_guidance(
            guidance_method="bg",
            guidance_scale=lambda_val,
            omega=best_omega,
        )

        return best_omega, best_scores

    def optimize_all(
        self,
        lambda_range: Tuple[float, float] = (2.5, 10.0),
        omega_range: Tuple[float, float] = (0.0, 1.0),
    ) -> dict:
        results = {}

        for method in ["cfg", "ag", "bg"]:
            print(f"Optimizing {method.upper()}...")
            if method == "bg":
                best_lambda, lambda_scores = self.optimize_lambda(
                    method, lambda_range, omega=0.0
                )
                best_omega, omega_scores = self.optimize_omega(best_lambda, omega_range)
                results[method] = {
                    "lambda": best_lambda,
                    "omega": best_omega,
                    "scores": omega_scores,
                }
            else:
                best_lambda, lambda_scores = self.optimize_lambda(method, lambda_range)
                results[method] = {
                    "lambda": best_lambda,
                    "scores": lambda_scores,
                }

        return results

