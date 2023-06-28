"""This module provides functions for image manipulations."""
from __future__ import annotations

import io

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL import ImagePalette

PALETTE = ImagePalette.ImagePalette(
    'RGB',
    [
        0, 0, 0,
        128, 0, 0,
        0, 128, 0,
        128, 128, 0,
        0, 0, 128,
        128, 0, 128,
        0, 128, 128,
        128, 128, 128,
        64, 0, 0,
        192, 0, 0,
        64, 128, 0,
        192, 128, 0,
        64, 0, 128,
        192, 0, 128,
        64, 128, 128,
        192, 128, 128,
        0, 64, 0,
        128, 64, 0,
        0, 192, 0,
        128, 192, 0,
        0, 64, 128,
    ],
)


def blend_with_mask(image: Image.Image, mask: Image.Image, alpha=0.4):
    """Blend an image with its segmentation mask."""
    mask_rgba = mask.convert('RGBA')
    r, g, b, a = mask_rgba.split()
    new_alpha = a.point(lambda x: int(x * alpha))
    mask_with_alpha = Image.merge('RGBA', (r, g, b, new_alpha))
    blended_image = Image.alpha_composite(
        image.convert('RGBA'), mask_with_alpha,
    )
    return blended_image


def fig2png(fig):
    """Convert matplotlib fig to np.array."""
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    png = buf.getvalue()
    decoded_png = np.array(Image.open(io.BytesIO(png)))
    return decoded_png


def plot_segmentation(image: np.array, mask: np.array, ax: plt.Axes):
    """Plot segmentation mask on image."""
    image = Image.fromarray(image.astype(np.uint8))
    mask = Image.fromarray(mask.astype(np.uint8))
    mask.putpalette(PALETTE)
    blended = blend_with_mask(image, mask, alpha=0.7)
    ax.imshow(blended)
