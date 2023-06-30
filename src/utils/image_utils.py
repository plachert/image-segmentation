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


def plot_segmentation_mask(image: np.ndarray, mask: np.array, ax: plt.Axes):
    """Plot segmentation mask on image."""
    image = 255 * image  # TODO: check the scale
    image = Image.fromarray(image.astype(np.uint8))
    mask = Image.fromarray(mask.astype(np.uint8))
    mask.putpalette(PALETTE)
    blended = blend_with_mask(image, mask, alpha=0.7)
    ax.imshow(blended)
    ax.axis('off')


def plot_segmentation(images: np.ndarray, masks: np.ndarray, classes: list[str]):
    """Plot a bunch of segmentation masks."""
    n_classes = len(classes)
    assert len(images) == len(masks)
    size = len(images)
    n_rows = int(np.ceil(np.sqrt(size)))
    n_cols = n_rows + 1  # +1 for legend

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, 10))
    for ax in np.ravel(axs):
        ax.axis('off')  # remove axes from all plots
    for idx, (image, mask) in enumerate(zip(images, masks)):
        row = idx // (n_cols - 1)
        col = idx % (n_cols - 1)
        if col == n_cols - 1:
            axs[row][col].remove()
            continue
        plot_segmentation_mask(image, mask, axs[row][col])

    ax_legend = plt.subplot(1, n_cols, n_cols)
    ax_legend.axis('off')
    colors = [np.array(PALETTE.palette[i*3:i*3+3]) for i in range(n_classes)]
    legend = ax_legend.barh(
        np.zeros(n_classes),
        np.zeros(n_classes),
        tick_label=classes,
        align='center',
        color=[x/255 for x in colors],
    )
    plt.legend(legend, classes, loc='center')

    return fig
