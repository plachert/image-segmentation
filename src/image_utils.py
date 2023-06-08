"""This module provides functions for image manipulations."""
from PIL import Image


def blend_with_mask(image: Image.Image, mask: Image.Image, alpha=0.4):
    """Blend an image with its segmentation mask."""
    mask_rgba = mask.convert("RGBA")
    r, g, b, a = mask_rgba.split()
    new_alpha = a.point(lambda x: int(x * alpha))
    mask_with_alpha = Image.merge("RGBA", (r, g, b, new_alpha))
    blended_image = Image.alpha_composite(image.convert("RGBA"), mask_with_alpha)
    return blended_image