# segmind/SSD-1BCog model

This is an implementation of the [segmind/SSD-1B](https://huggingface.co/segmind/SSD-1B) as a Cog model. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)

First, download the pre-trained weights:

    cog run script/download-weights

Then, you can run predictions:

    cog predict -i prompt="with smoke, half ice and half fire and ultra realistic in detail.wolf, typography, dark fantasy, wildlife photography, vibrant, cinematic and on a black background" -i seed=36446545872

Or img2img:

    cog predict -i image=@output.0.png -i prompt="a wolf with pink and blue fur" -i seed=21272 -i disable_safety_checker=True


## Examples:

txt2img

![alt text](output.0.png)

img2img

![alt text](output.img2img.png)
