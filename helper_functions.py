import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from skimage.filters import threshold_otsu

from plotly.offline import plot
from plotly.figure_factory import create_trisurf
import plotly.graph_objects as go

import numpy as np


def calculate_dice_similarity_coefficient(my_segmentation, truth):
    '''
    Dice Similarity Coefficient = \frac{2 * cardinality(A intersection B)}{cardinality(A) + cardinality(B)}

    Args:
      my_segmentation: numpy array of booleans
      truth: numpy array of booleans
    '''
    intersection = np.logical_and(my_segmentation, truth)
    union = np.logical_or(my_segmentation, truth)
    numerator = 2 * np.sum(intersection)
    denominator = np.sum(union) + np.sum(intersection)
    print(np.sum(union))
    print(np.sum(intersection))
    dsc = (2 * np.sum(intersection))/(np.sum(union) + np.sum(intersection))
    return dsc, numerator, denominator


def view_3D(volume):
    # NOTE: This is VERY resource-intensive! Use with caution.
    # If you want a responsive plot, be sure to resample the volume first!
    verts, faces = make_mesh(volume)
    plotly_3d(verts, faces, volume)


def make_mesh(image, threshold=0, step_size=1):

    print("Transposing surface")
    p = image.transpose(2, 1, 0)

    print("Calculating surface")
    verts, faces, norm, val = measure.marching_cubes(
        p, threshold, step_size=step_size, allow_degenerate=True)

    return verts, faces


def plotly_3d(verts, faces, volume):
    x, y, z = zip(*verts)

    # Make the colormap single color since the axes are positional not intensity.
    colormap = ['rgb(236, 236, 212)', 'rgb(236, 236, 212)']

    fig = create_trisurf(x=x,
                         y=y,
                         z=z,
                         plot_edges=False,
                         colormap=colormap,
                         simplices=faces,
                         backgroundcolor='rgb(64, 64, 64)',
                         title="Interactive Visualization")
    plot(fig)


def create_slices(volume):
    slices = []
    for i in range(volume.shape[2]):
        # separate slices
        slice = volume[:, :, i]
        slices.append(slice)
    return slices


def create_bronchial_slices(volume):
    slices = []
    for i in range(volume.shape[0]):
        # separate slices
        slice = volume[i, :, :]
        slices.append(slice)
    return slices


def separate_hounsfield_range(slice, thresh_min, thresh_max):
    slc = slice.copy()
    slc[slc <= thresh_min] = 0
    slc[slc >= thresh_max] = 0
    return slc


def create_binary_mask(volume):
    thresh = threshold_otsu(volume)
    binary = volume > thresh
    return binary

def create_grayscale_mask(self, volume):
    thresh = threshold_otsu(volume)
    slc = volume.copy()
    slc[slc >= thresh] = 0
    slc[slc <= thresh] = 255
    return slc
