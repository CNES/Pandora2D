import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def plot_image(img, title=None, output_dir=None, cmap="viridis"):
    """
     Show images
    :param img: image to show
    :type img: numpy.ndarray
    :param title: Title of graph
    :type title: str
    :param output_dir: path to output directory
    :type output_dir: str
    :param cmap: type of colors for the map
    :type cmap: str

    :return: None
    """
    fig = plt.figure()
    plt.title(title)
    plt.imshow(img, cmap=cmap)
    plt.colorbar()
    if output_dir is not None:
        fig.savefig(os.path.join(output_dir, title + ".pdf"))


def plot_two_images(img1, img2, title1="first_image", title2="second_image", output_dir=None, cmap="viridis"):
    """
    Show images side by side
    :param img1: image 1 to show
    :type img1: numpy.ndarray
    :param img2: image 2 to show
    :type img2: numpy.ndarray
    :param title1: Title of graph 1
    :type title1: str
    :param title2: Title of graph 2
    :type title1: str
    :param output_dir: path to output directory
    :type output_dir: str
    :param cmap: type of colors for the map
    :type cmap: str

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
