import io
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

def plot_image(img, title=None, output_dir=None, cmap="viridis"):
    """
     Show images
    :param img: image to show
    :type param: numpy.array
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
        fig.savefig(os.path.join(output_dir,title + '.pdf'))


def pandora_cmap():
    """
    Instantiate colors for disparity maps
    """
    colors = ["crimson", "lightpink", "white", "yellowgreen"]
    nodes = [0.0, 0.4, 0.5, 1.0]
    cmap_shift = LinearSegmentedColormap.from_list("mycmap", list(zip(nodes, colors)))

    return cmap_shift
