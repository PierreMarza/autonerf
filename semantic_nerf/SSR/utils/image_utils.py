################################################################
# Code adapted from https://github.com/Harry-Zhi/semantic_nerf #
# Adapted by Pierre Marza (pierre.marza@insa-lyon.fr)          #
################################################################

import imgviz
from imgviz import label_colormap
from imgviz import draw as draw_module
import matplotlib.pyplot as plt
import numpy as np
import os

color_palette = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.6, 0.6, 0.6],
        [0.95, 0.95, 0.95],
        [0.96, 0.36, 0.26],
        [0.12156862745098039, 0.47058823529411764, 0.7058823529411765],
        [0.9400000000000001, 0.7818, 0.66],
        [0.9400000000000001, 0.8868, 0.66],
        [0.8882000000000001, 0.9400000000000001, 0.66],
        [0.7832000000000001, 0.9400000000000001, 0.66],
        [0.6782000000000001, 0.9400000000000001, 0.66],
        [0.66, 0.9400000000000001, 0.7468000000000001],
        [0.66, 0.9400000000000001, 0.8518000000000001],
        [0.66, 0.9232, 0.9400000000000001],
        [0.66, 0.8182, 0.9400000000000001],
        [0.66, 0.7132, 0.9400000000000001],
        [0.7117999999999999, 0.66, 0.9400000000000001],
        [0.8168, 0.66, 0.9400000000000001],
        [0.9218, 0.66, 0.9400000000000001],
        [0.9400000000000001, 0.66, 0.8531999999999998],
        [0.9400000000000001, 0.66, 0.748199999999999],
    ]
)


def plot_semantic_legend(
    label,
    label_name,
    colormap=None,
    font_size=30,
    font_path=None,
    save_path=None,
    img_name=None,
):

    """Plot Colour Legend for Semantic Classes

    Parameters
    ----------
    label: numpy.ndarray, (N,), int
        One-dimensional array containing the unique labels of exsiting semantic classes
    label_names: list of string
        Label id to label name.
    font_size: int
        Font size (default: 30).
    colormap: numpy.ndarray, (M, 3), numpy.uint8
        Label id to color.
        By default, :func:`~imgviz.label_colormap` is used.
    font_path: str
        Font path.

    Returns
    -------
    res: numpy.ndarray, (H, W, 3), numpy.uint8
    Legend image of visualising semantic labels.

    """

    label = np.unique(label)
    if colormap is None:
        colormap = label_colormap()

    text_sizes = np.array(
        [
            draw_module.text_size(label_name[l], font_size, font_path=font_path)
            for l in label
        ]
    )

    text_height, text_width = text_sizes.max(axis=0)
    legend_height = text_height * len(label) + 5
    legend_width = text_width + 20 + (text_height - 10)

    legend = np.zeros((legend_height + 50, legend_width + 50, 3), dtype=np.uint8)
    aabb1 = np.array([25, 25], dtype=float)
    aabb2 = aabb1 + (legend_height, legend_width)

    legend = draw_module.rectangle(
        legend, aabb1, aabb2, fill=(255, 255, 255)
    )  # fill the legend area by white colour

    y1, x1 = aabb1.round().astype(int)
    y2, x2 = aabb2.round().astype(int)

    for i, l in enumerate(label):
        box_aabb1 = aabb1 + (i * text_height + 5, 5)
        box_aabb2 = box_aabb1 + (text_height - 10, text_height - 10)
        legend = draw_module.rectangle(
            legend, aabb1=box_aabb1, aabb2=box_aabb2, fill=colormap[l]
        )
        legend = draw_module.text(
            legend,
            yx=aabb1 + (i * text_height, 10 + (text_height - 10)),
            text=label_name[l],
            size=font_size,
            font_path=font_path,
        )

    plt.figure(1)
    plt.title("Semantic Legend!")
    plt.imshow(legend)
    plt.axis("off")

    img_arr = imgviz.io.pyplot_to_numpy()
    plt.close()
    if save_path is not None:
        import cv2

        if img_name is not None:
            sav_dir = os.path.join(save_path, img_name)
        else:
            sav_dir = os.path.join(save_path, "semantic_class_Legend.png")
        cv2.imwrite(sav_dir, img_arr[:, :, ::-1])
    return img_arr
