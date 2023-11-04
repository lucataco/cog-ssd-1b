# Prediction interface for Cog

from cog import BasePredictor, Input, Path
import os
import torch
from diffusers import (
    StableDiffusionXLPipeline,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    PNDMScheduler
)
from typing import List

MODEL_NAME = "segmind/SSD-1B"
MODEL_CACHE = "model-cache"

class KarrasDPM:
    def from_config(config):
        return DPMSolverMultistepScheduler.from_config(config, use_karras_sigmas=True)

SCHEDULERS = {
    "DDIM": DDIMScheduler,
    "DPMSolverMultistep": DPMSolverMultistepScheduler,
    "HeunDiscrete": HeunDiscreteScheduler,
    "KarrasDPM": KarrasDPM,
    "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler,
    "K_EULER": EulerDiscreteScheduler,
    "PNDM": PNDMScheduler,
}

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            MODEL_CACHE,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        ).to("cuda")

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="with smoke, half ice and half fire and ultra realistic in detail.wolf, typography, dark fantasy, wildlife photography, vibrant, cinematic and on a black background"
        ),
        negative_prompt: str = Input(
            description="Negative Input prompt",
            default="scary, cartoon, painting"
        ),
        width: int = Input(
            description="Width of output image",
            default=768
        ),
        height: int = Input(
            description="Height of output image",
            default=768
        ),
        num_outputs: int = Input(
            description="Number of images to output.",
            ge=1,
            le=4,
            default=1,
        ),
        scheduler: str = Input(
            description="scheduler",
            choices=SCHEDULERS.keys(),
            default="K_EULER",
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=25
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=50, default=7.5
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")
        generator = torch.Generator("cuda").manual_seed(seed)

        self.pipe.scheduler = SCHEDULERS[scheduler].from_config(self.pipe.scheduler.config)

        common_args = {
            "prompt": [prompt] * num_outputs,
            "negative_prompt": [negative_prompt] * num_outputs,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "num_inference_steps": num_inference_steps,
        }

        sdxl_kwargs = {}
        sdxl_kwargs["width"] = width
        sdxl_kwargs["height"] = height

        output = self.pipe(**common_args, **sdxl_kwargs)
        
        output_paths = []
        for i, _ in enumerate(output.images):
            output_path = f"/tmp/out-{i}.png"
            output.images[i].save(output_path)
            output_paths.append(Path(output_path))

        return output_paths