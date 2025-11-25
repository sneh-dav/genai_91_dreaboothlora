import torch
import torch.nn.functional as F
from typing import Optional
from PIL import Image
from diffusers import SanaPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipelineOutput,
)


class SanaGuidancePipeline(SanaPipeline):
    """
    This is the SANA version of your old GuidancePipeline.
    Works with:
        - CFG guidance
        - AutoGuidance (AG)
    Supports:
        - weak model (for AG)
        - unconditional embeddings
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weak_unet = None

    def set_weak_model(self, weak_pipeline):
        """Attach a weak SANA UNet for AG."""
        self.weak_unet = weak_pipeline.unet

    # ---------------------------------------------------------
    #   UNCONDITIONAL EMBEDDINGS
    # ---------------------------------------------------------
    def _get_unconditional_embeddings(self, batch_size: int, device):
        """Get unconditional text encoder embeddings."""
        uncond_tokens = [""] * batch_size
        max_length = self.tokenizer.model_max_length

        uncond_input = self.tokenizer(
            uncond_tokens,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

        uncond_embeds = self.text_encoder(
            uncond_input.input_ids.to(device)
        )[0]

        return uncond_embeds

    # ---------------------------------------------------------
    #   CFG GUIDANCE
    # ---------------------------------------------------------
    def cfg_guidance(
        self,
        latents,
        timestep,
        prompt_embeds,
        guidance_scale: float
    ):
        if prompt_embeds.shape[0] == latents.shape[0] * 2:
            uncond_embeds, cond_embeds = prompt_embeds.chunk(2)
        else:
            uncond_embeds = self._get_unconditional_embeddings(
                latents.shape[0], latents.device
            )
            cond_embeds = prompt_embeds

        uncond_pred = self.unet(
            latents,
            timestep,
            encoder_hidden_states=uncond_embeds,
        ).sample

        cond_pred = self.unet(
            latents,
            timestep,
            encoder_hidden_states=cond_embeds,
        ).sample

        return uncond_pred + guidance_scale * (cond_pred - uncond_pred)

    # ---------------------------------------------------------
    #   AUTO GUIDANCE (AG)
    # ---------------------------------------------------------
    def ag_guidance(
        self,
        latents,
        timestep,
        prompt_embeds,
        guidance_scale: float,
        cfg_scale: float,
    ):
        # Weak model UNet
        weak_unet = self.weak_unet
        if weak_unet is None:
            raise ValueError("Weak model not set — call set_weak_model().")

        batch = latents.shape[0]
        uncond_embeds = self._get_unconditional_embeddings(batch, latents.device)

        # Weak model CFG
        weak_uncond = weak_unet(latents, timestep, encoder_hidden_states=uncond_embeds).sample
        weak_cond   = weak_unet(latents, timestep, encoder_hidden_states=prompt_embeds).sample
        weak_cfg_pred = weak_uncond + cfg_scale * (weak_cond - weak_uncond)

        # Fine model CFG
        fine_uncond = self.unet(latents, timestep, encoder_hidden_states=uncond_embeds).sample
        fine_cond   = self.unet(latents, timestep, encoder_hidden_states=prompt_embeds).sample
        fine_cfg_pred = fine_uncond + cfg_scale * (fine_cond - fine_uncond)

        # AG final
        return weak_cfg_pred + guidance_scale * (fine_cfg_pred - weak_cfg_pred)

    # ---------------------------------------------------------
    #   MAIN GENERATION FUNCTION
    # ---------------------------------------------------------
    def generate_with_guidance(
        self,
        prompt: str,
        guidance_method: str = "ag",
        guidance_scale: float = 2.0,
        cfg_scale: float = 7.5,
        num_inference_steps: int = 50,
        height: int = 512,
        width: int = 512,
        num_images_per_prompt: int = 1,
    ) -> StableDiffusionPipelineOutput:

        device = self._execution_device
        self.scheduler.set_timesteps(num_inference_steps, device=device)

        # Encode text
        text_inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )

        text_embeds = self.text_encoder(
            text_inputs.input_ids.to(device)
        )[0]

        # Latents
        latents = torch.randn(
            (num_images_per_prompt, self.unet.config.in_channels, height // 8, width // 8),
            device=device,
            dtype=text_embeds.dtype,
        )
        latents = latents * self.scheduler.init_noise_sigma

        for t in self.scheduler.timesteps:

            if guidance_method == "cfg":
                latent_in = torch.cat([latents] * 2)
                latent_in = self.scheduler.scale_model_input(latent_in, t)

                noise_pred = self.cfg_guidance(
                    latent_in,
                    t,
                    text_embeds,
                    guidance_scale,
                )

            elif guidance_method == "ag":
                latent_in = self.scheduler.scale_model_input(latents, t)

                noise_pred = self.ag_guidance(
                    latent_in,
                    t,
                    text_embeds,
                    guidance_scale,
                    cfg_scale,
                )

            else:
                raise ValueError(f"Unknown guidance method: {guidance_method}")

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # Decode
        image = self.vae.decode(
            latents / self.vae.config.scaling_factor,
            return_dict=False,
        )[0]

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = (image * 255).astype("uint8")

        images = [Image.fromarray(img) for img in image]
        return StableDiffusionPipelineOutput(images=images, nsfw_content_detected=None)
