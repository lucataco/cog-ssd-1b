# Prediction interface for Cog

from cog import BasePredictor, Input, Path
import os
import math
import time
import torch
from PIL import Image
from diffusers import (
    StableDiffusionXLImg2ImgPipeline,
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
        t1 = time.time()
        self.pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
            cache_dir=MODEL_CACHE,
        ).to("cuda")
        t2 = time.time()
        print("Setup ssd-1b took: ", t2 - t1)

    def scale_down_image(self, image_path, max_size):
        image = Image.open(image_path)
        width, height = image.size
        scaling_factor = min(max_size/width, max_size/height)
        new_width = int(width * scaling_factor)
        new_height = int(height * scaling_factor)
        resized_image = image.resize((new_width, new_height))
        cropped_image = self.crop_center(resized_image)
        return cropped_image

    def crop_center(self, pil_img):
        img_width, img_height = pil_img.size
        crop_width = self.base(img_width)
        crop_height = self.base(img_height)
        return pil_img.crop(
                (
                    (img_width - crop_width) // 2,
                    (img_height - crop_height) // 2,
                    (img_width + crop_width) // 2,
                    (img_height + crop_height) // 2
                )
            )

    def base(self, x):
        return int(8 * math.floor(int(x)/8))
    
    @torch.inference_mode()
    def predict(
        self,
        image: Path = Input(description="Input image"),
        prompt: str = Input(
            description="Input prompt",
            default="a wolf with pink and blue fur"
        ),
        negative_prompt: str = Input(
            description="Negative Input prompt",
            default="scary, cartoon, painting"
        ),
        scheduler: str = Input(
            description="scheduler",
            choices=SCHEDULERS.keys(),
            default="K_EULER",
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=20
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=50, default=8
        ),
        strength: float = Input(
            description="strength/weight", ge=0, le=1, default=0.9
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")
        generator = torch.Generator("cuda").manual_seed(seed)

        self.pipe.scheduler = SCHEDULERS[scheduler].from_config(self.pipe.scheduler.config)

        resized_image = self.scale_down_image(image, 1024)

        common_args = {
            "image": resized_image,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "strength": strength,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "generator": generator,
        }
        output = self.pipe(**common_args)
        
        output_path = f"/tmp/output.png"
        output.images[0].save(output_path)

        return Path(output_path)