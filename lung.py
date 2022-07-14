'''
This class segments the whole lung using a watershed algorithm similar to that in R Shojaii et al (2005, DOI: 10.1109/ICIP.2005.1530294). It is primarily based on
some code posted on Kaggle for the 2017 Data Science Bowl (https://www.kaggle.com/ankasor/improved-lung-segmentation-using-watershed/notebook, accessed March 2022)
which used code from various sources, including the authors of the original paper (R Shojaii et al). This version adds changes intended for the dataset referenced in Kassin, et al.
(2021, https://doi.org/10.1038/s41598-021-85694-5) and updates some of the older libraries used.

Watershed relies on using markers to conduct segmentation. Usually, this is done manually and is seen frequently in radiological segmentation software. Here, 
the process is automated, which is possible due to the narrow use case.
'''

import numpy as np 
import pandas as pd # numpy and pandas overlap, here we use numpy for most operations and pandas for file I/O
import os
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

from skimage import measure, segmentation
from skimage.filters import try_all_threshold
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import helper_functions

LUNG_TEST = False

class Lung:
    def __init__(self, volume, px_height, px_width, voxel_x, voxel_y, voxel_z) -> None:
        # Note that for this to work, the Hounsfield Units must be preserved in the passed in volume, do NOT convert to another format before running
        self.volume = volume
        self.px_height = px_height
        self.px_width = px_width
        self.voxel_x = voxel_x
        self.voxel_y = voxel_y
        self.voxel_z = voxel_z
        self.slices = helper_functions.create_slices(self.volume)
        self.segmentations, self.masks = self.process()
        self.lung_volume = self.calculate_lung_volume()

    def calculate_lung_volume(self):
        # for each slice, count non-zero pixels
        # add up volume
        total_mask = 0
        total_mask_vol = 0
        for i in range(len(self.masks)):
            mask_px = np.count_nonzero(self.masks[i])
            total_mask += mask_px
            mask_dim = mask_px * (self.voxel_x * self.voxel_y)
            total_mask_vol += (mask_dim * self.voxel_z)
        return total_mask_vol/1000
        

    def seperate_lungs(self, slice):
        #Creation of the markers as shown above:
        marker_internal, marker_external, marker_watershed = self.generate_watershed_markers(slice)
        
        #Creation of the Sobel-Gradient
        sobel_filtered_dx = ndimage.sobel(slice, 1)
        sobel_filtered_dy = ndimage.sobel(slice, 0)
        sobel_gradient = np.hypot(sobel_filtered_dx, sobel_filtered_dy)
        sobel_gradient *= 255.0 / np.max(sobel_gradient)
        
        #Watershed algorithm
        watershed = segmentation.watershed(sobel_gradient, marker_watershed)
        
        #Reducing the image created by the Watershed algorithm to its outline
        outline = ndimage.morphological_gradient(watershed, size=(3,3))
        outline = outline.astype(bool)
        
        #Performing Black-Tophat Morphology for reinclusion
        #Creation of the disk-kernel and increasing its size a bit
        blackhat_struct = [[0, 0, 1, 1, 1, 0, 0],
                        [0, 1, 1, 1, 1, 1, 0],
                        [1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1],
                        [0, 1, 1, 1, 1, 1, 0],
                        [0, 0, 1, 1, 1, 0, 0]]
        blackhat_struct = ndimage.iterate_structure(blackhat_struct, 8)
        #Perform the Black-Hat
        outline += ndimage.black_tophat(outline, structure=blackhat_struct)
        
        #Use the internal marker and the Outline that was just created to generate the lungfilter
        lungfilter = np.bitwise_or(marker_internal, outline)
        #Close holes in the lungfilter
        #fill_holes is not used here, since in some slices the heart would be reincluded by accident
        lungfilter = ndimage.morphology.binary_closing(lungfilter, structure=np.ones((5,5)), iterations=3)
        
        #Apply the lungfilter (note the filtered areas being assigned -2000 HU)
        segmented = np.where(lungfilter == 1, slice, -2000*np.ones((self.px_height, self.px_width)))
        
        return segmented, lungfilter, outline, watershed, sobel_gradient, marker_internal, marker_external, marker_watershed

    def generate_watershed_markers(self, slice):
        # This code is taken from the Kaggle notebook with some minor changes
        #Creation of the internal Marker
        marker_internal = slice < -400
        marker_internal = segmentation.clear_border(marker_internal)
        marker_internal_labels = measure.label(marker_internal)
        areas = [r.area for r in measure.regionprops(marker_internal_labels)]
        areas.sort()
        if len(areas) > 2:
            for region in measure.regionprops(marker_internal_labels):
                if region.area < areas[-2]:
                    for coordinates in region.coords:                
                        marker_internal_labels[coordinates[0], coordinates[1]] = 0
        marker_internal = marker_internal_labels > 0
        #Creation of the external Marker
        external_a = ndimage.binary_dilation(marker_internal, iterations=10)
        external_b = ndimage.binary_dilation(marker_internal, iterations=55)
        marker_external = external_b ^ external_a
        #Creation of the Watershed Marker matrix
        marker_watershed = np.zeros((self.px_width, self.px_height), dtype=np.int)
        marker_watershed += marker_internal * 255
        marker_watershed += marker_external * 128
        
        return marker_internal, marker_external, marker_watershed

    def process(self, healthy=False):
        segmentations = []
        masks = []
        slice_list = self.slices
        if healthy:
            slice_list = self.healthy_slices
        for i in range(len(slice_list)):
            segmented, lungfilter, outline, watershed, sobel_gradient, marker_internal, marker_external, marker_watershed = self.seperate_lungs(slice_list[i])
            segmentations.append(segmented)
            masks.append(lungfilter)

        if LUNG_TEST:
            segmented, lungfilter, outline, watershed, sobel_gradient, marker_internal, marker_external, marker_watershed = self.seperate_lungs(self.slices[len(slice_list)//2])
            plt.imshow(lungfilter, cmap="gray")
            plt.show()
            plt.imshow(segmented, cmap="gray")
            plt.show()
        return segmentations, masks