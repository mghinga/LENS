from skimage.transform import resize
from skimage.filters import frangi
from skimage import util
from skimage.morphology import erosion, ball, white_tophat

import matplotlib.pyplot as plt

import numpy as np

import helper_functions

class Lesion:
    def __init__(self, volume, lung_mask, bronchial_mask, mm_x, mm_y, mm_z, footprint_size, truth=None, no_processing=False) -> None:
        self.volume = volume
        self.lung_mask = lung_mask
        self.footprint_size = footprint_size

        self.mm_x = mm_x
        self.mm_y = mm_y
        self.mm_z = mm_z


        self.lung_mask = resize(self.lung_mask, self.bronchial_mask.shape)
        self.volume = resize(self.lung_mask, self.bronchial_mask.shape)

        if no_processing:
            self.bronchial_mask = self.volume * self.lung_mask
        else:
            # remove parts of the masks and volumes
            length = self.volume.shape[2]
            # original image
            self.volume = self.volume[:, :, 2:length-12]
            # output of bronchial segmentation, bronchial tree is white
            self.bronchial_mask = bronchial_mask[:, :, 2:length-12]
            # lung segmentation mask, lung region is white
            self.lung_mask = self.lung_mask[:, :, 2:length-12]


            self.bronchial_mask = helper_functions.create_grayscale_mask(self.bronchial_mask)
            self.bronchial_mask = self.create_combined_mask()



            self.eroded_lung_mask = erosion(self.lung_mask, ball(self.footprint_size))

            self.bronchial_mask = self.bronchial_mask * self.eroded_lung_mask

            # remove small structures with tophat
            footprint = ball(1)
            white_hat = white_tophat(self.bronchial_mask, footprint)

            self.bronchial_mask = self.bronchial_mask - white_hat
            self.bronchial_mask = self.generate_mask_for_calculating_volume()
       
        

        self.total_lesion_voxels, self.total_lesion_volume = self.calculate_total_lesion_volume(self.bronchial_mask)
        

        if truth is not None:
            truth = truth[:, :, 2:length-12]
            truth = resize(truth, self.bronchial_mask.shape)
            binary_lesion = self.create_binary_mask(self.bronchial_mask)
            binary_truth = self.create_binary_mask(truth)
            dice_coeff = helper_functions.calculate_dice_similarity_coefficient(binary_lesion, binary_truth)
            print("DSC with Bronchial Removal: ", dice_coeff)
        
    def generate_mask_for_calculating_volume(self):
        binary_bronchial_mask = helper_functions.create_binary_mask(self.bronchial_mask)
        mask = np.bitwise_and(self.eroded_lung_mask, binary_bronchial_mask)
        return mask


    def view_volume(self, volume):
        slices = helper_functions.create_slices(volume)
        for i in range(len(slices)):
            plt.imshow(slices[i], cmap='gray')
            plt.show()
            # plt.savefig('figures/710/02/'+str(i))

    def calculate_total_lesion_volume(self, lesions):
        total_voxels = np.count_nonzero(lesions)
        total_volume = total_voxels * self.mm_x * self.mm_y * self.mm_z
        total_volume = total_volume/1000
        return total_voxels, total_volume

    def resize_jelly_bean_mask(self, jelly_bean_mask):
        desired_x, desired_y, desired_z = self.lung_mask.shape
        x, y, z = jelly_bean_mask.shape
        difference_x = desired_x - x
        difference_y = desired_y - y
        difference_z = desired_z - z
        pad_x_left = difference_x//2
        pad_x_right = difference_x - pad_x_left
        pad_y_left = difference_y//2
        pad_y_right = difference_y - pad_y_left
        pad_z_left = difference_z//2
        pad_z_right = difference_z - pad_z_left
        padded_jelly_bean = np.pad(jelly_bean_mask, ((
            pad_x_left, pad_x_right), (pad_y_left, pad_y_right), (pad_z_left, pad_z_right)), 'constant')
        return padded_jelly_bean

    def calculate_change_in_dimensions(self, jelly_bean_mask):
        jelly_bean_size = np.count_nonzero(jelly_bean_mask)
        volume_size = np.count_nonzero(self.volume)
        return jelly_bean_size/volume_size

    def isolate_lesions(self):
        original_volume = self.volume * self.lung_mask
        lesion_volume = original_volume * self.combined
        lesion_volume = lesion_volume * self.volume
        return lesion_volume


    def create_combined_mask(self):
        original = self.bronchial_mask
        lung_mask = self.lung_mask
        volume = self.volume
        # remove bronchial tubes from lung mask
        bronchial_mask = util.invert(original)
        mask = lung_mask * bronchial_mask
        # keep masked area in volume
        frangi_mask = mask * volume
        frangi_mask = helper_functions.separate_hounsfield_range(
            frangi_mask, -1000, -300)
        frangi_mask = frangi(
            frangi_mask, sigmas=range(1, 3), black_ridges=False)
        frangi_mask = util.invert(frangi_mask)
        frangi_mask = self.create_mask(frangi_mask)
        # remove Frangi mask result from mask
        mask = lung_mask * frangi_mask
        final = volume * mask
        final = self.create_mask(final)
        final = util.invert(final)
        return final