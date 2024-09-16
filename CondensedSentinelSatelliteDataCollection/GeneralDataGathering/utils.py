"""
Utilities used by example notebooks
"""
from typing import Any, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import cv2

def plot_image(
    image: np.ndarray, factor: float = 1.0, clip_range: Optional[Tuple[float, float]] = None, **kwargs: Any
) -> None:
    """Utility function for plotting RGB images."""
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 9)) #normally 15 by 15
    if clip_range is not None:
        ax.imshow(np.clip(image * factor, *clip_range), **kwargs)
        plt.show()
    else:
        ax.imshow(image * factor, **kwargs)
        plt.show()
    ax.set_xticks([])
    ax.set_yticks([])

def save_image(
    path: str, image: np.ndarray, factor: float = 1.0, clip_range: Optional[Tuple[float, float]] = None, **kwargs: Any
) -> None:
    """Utility function for plotting RGB images."""
    plt.axis('off')
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))
    ax.set_axis_off()
    ax.axis('off')
    if clip_range is not None:
        ax.imshow(np.clip(image * factor, *clip_range), **kwargs)
    else:
        ax.imshow(image * factor, **kwargs)

    plt.savefig(path, dpi = 300, bbox_inches='tight', transparent=True)