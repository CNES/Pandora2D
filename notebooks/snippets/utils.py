"""
Utilities for notebooks.
"""

import os
from typing import Union

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from numpy.typing import NDArray


def plot_image(
    img: NDArray,
    title: Union[str, None] = None,
    output_dir: Union[str, os.PathLike[str], None] = None,
    cmap: str = "viridis",
) -> None:
    """
     Show images
    :param img: image to show
    :param title: Title of graph
    :param output_dir: path to output directory
    :param cmap: type of colors for the map

    :return: None
    """
    fig = plt.figure()
    plt.title(title)
    plt.imshow(img, cmap=cmap)
    plt.colorbar()
    if output_dir is not None:
        fig.savefig(os.path.join(output_dir, title + ".pdf"))


def plot_two_images(
    img1: NDArray,
    img2: NDArray,
    title1: str = "first_image",
    title2: str = "second_image",
    output_dir: Union[str, os.PathLike[str], None] = None,
    cmap: str = "viridis",
) -> None:
    """
    Show images side by side
    :param img1: image 1 to show
    :param img2: image 2 to show
    :param title1: Title of graph 1
    :param title2: Title of graph 2
    :param output_dir: path to output directory
    :param cmap: type of colors for the map

    :return: None
    """
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    im1 = axs[0].imshow(img1, cmap=cmap)
    axs[0].set_title(title1)
    axs[0].axis("off")

    axs[1].imshow(img2, cmap=cmap)
    axs[1].set_title(title2)
    axs[1].axis("off")

    fig.colorbar(im1, ax=axs.ravel().tolist(), shrink=0.6)

    if output_dir is not None:
        fig.savefig(os.path.join(output_dir, title1 + "_" + title2 + ".pdf"))


def pandora_cmap():
    """
    Instantiate colors for disparity maps
    """
    colors = ["crimson", "lightpink", "white", "yellowgreen"]
    nodes = [0.0, 0.4, 0.5, 1.0]
    cmap_shift = LinearSegmentedColormap.from_list("mycmap", list(zip(nodes, colors)))

    return cmap_shift


def pandora_cmap_deformation_grid():
    """
    Instantiate colors for deformation grids
    """
    colors = ["white", "lightpink", "crimson", "yellowgreen"]
    nodes = [0.0, 0.3, 0.6, 1.0]
    cmap = LinearSegmentedColormap.from_list("pandora_deformation_grid", list(zip(nodes, colors)))
    return cmap
