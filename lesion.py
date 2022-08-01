from skimage.transform import resize
from skimage.filters import frangi
from skimage import util, exposure
from skimage.morphology import erosion, ball, disk, white_tophat

import matplotlib.pyplot as plt

import numpy as np

import helper_functions, individual_lesion_volume

class Lesion:
    def __init__(self, src_file, volume, lung_mask, bronchial_mask, mm_x, mm_y, mm_z, footprint_size, number, demo=False, two_d=False, truth=None, no_processing=False) -> None:
        print('Begin lesion segmentation.')
        self.src_file = src_file
        self.volume = volume
        self.lung_mask = lung_mask
        self.bronchial_mask = bronchial_mask
        self.footprint_size = footprint_size
        self.number = number
        self.two_d = two_d
        self.demo = demo
        self.mm_x = mm_x
        self.mm_y = mm_y
        self.mm_z = mm_z

        self.lung_mask = resize(self.lung_mask, self.bronchial_mask.shape)
        self.volume = resize(self.volume, self.bronchial_mask.shape)

        self.volume = self.lung_mask * self.volume
        bronchial_slices = helper_functions.create_slices(self.volume)

        if no_processing:
            self.bronchial_mask = self.volume * self.lung_mask
            self.bronchial_mask = helper_functions.create_grayscale_mask(self.bronchial_mask)
            self.bronchial_mask = self.create_combined_mask()
            self.total_lesion_voxels, self.total_lesion_volume = self.calculate_total_lesion_volume(self.bronchial_mask)
            return
        
        elif len(self.lung_mask) > 2:
            if self.lung_mask.shape[2] > 1:
                print(self.lung_mask.shape)
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
                self.bronchial_mask = exposure.equalize_hist(self.bronchial_mask)
                self.bronchial_mask = helper_functions.create_binary_mask(self.bronchial_mask)
                white_hat = white_tophat(self.bronchial_mask, ball(1))
                self.bronchial_mask = self.bronchial_mask ^ white_hat
                self.bronchial_mask = self.generate_mask_for_calculating_volume()
                self.bronchial_mask = self.bronchial_mask * self.volume
                self.bronchial_mask = erosion(self.bronchial_mask, ball(1))
                self.bronchial_mask = self.generate_mask_for_calculating_volume()

        else:
            self.bronchial_mask = helper_functions.create_grayscale_mask(self.bronchial_mask)
            self.bronchial_mask = self.create_combined_mask()
            self.eroded_lung_mask = erosion(self.lung_mask, disk(self.footprint_size))
            self.bronchial_mask = self.bronchial_mask * self.eroded_lung_mask
            white_hat = white_tophat(self.bronchial_mask, disk(self.footprint_size))
            self.bronchial_mask = self.bronchial_mask ^ white_hat
            self.bronchial_mask = self.generate_mask_for_calculating_volume()
            self.bronchial_mask = util.invert(self.bronchial_mask)
            self.bronchial_mask = self.bronchial_mask * self.volume
            


        self.total_lesion_voxels, self.total_lesion_volume = self.calculate_total_lesion_volume(self.bronchial_mask)
        if not self.two_d:
            self.individual_lesions = individual_lesion_volume.LesionVolume(self.bronchial_mask, self.mm_x, self.mm_y, self.mm_z)
            if truth is not None:
                binary_truth = helper_functions.create_binary_mask(truth)
                self.individual_truth_lesions = individual_lesion_volume.LesionVolume(binary_truth, self.mm_x, self.mm_y, self.mm_z, True)

        if self.demo:
            if self.two_d:
                plt.imshow(self.bronchial_mask, cmap='gray')
                plt.title('Lesion Mask')
                plt.show()
            else:
                bronchial_slices = helper_functions.create_slices(self.bronchial_mask)
                plt.imshow(util.invert(bronchial_slices[len(bronchial_slices)//2]), cmap='gray')
                plt.title('Lesion Mask')
                plt.show()
        name = self.generate_filename()

        if truth is not None:
            truth = resize(truth, self.bronchial_mask.shape)
            binary_lesion = helper_functions.create_binary_mask(self.bronchial_mask)
            binary_truth = helper_functions.create_binary_mask(truth)
            dice_coeff, numerator, denominator = helper_functions.calculate_dice_similarity_coefficient(binary_lesion, binary_truth)
            print("DSC with Bronchial Removal: ", dice_coeff)
            print(dice_coeff, numerator, denominator)
            if not self.two_d:
                helper_functions.view_3D(binary_lesion)
        
    def generate_mask_for_calculating_volume(self):
        binary_bronchial_mask = helper_functions.create_binary_mask(self.bronchial_mask)
        mask = np.bitwise_and(self.eroded_lung_mask, binary_bronchial_mask)
        return mask

    def generate_filename(self, dsc=-1):
        file = 'has_dsc'
        if dsc > 0:
            name = 'output/two_dimensional/' + file + '_' + str(self.number) + '_DSC_' + str(dsc) + '.png'
        else:
            file = 'no_truth'
            name = 'output/two_dimensional/' + file + '_' + str(self.number) + '.png'
        return name

    def view_volume(self, volume):
        slices = helper_functions.create_slices(volume)
        for i in range(len(slices)):
            plt.imshow(slices[i], cmap='gray')
            plt.show()

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
        frangi_mask = helper_functions.create_binary_mask(frangi_mask)
        # remove Frangi mask result from mask
        mask = lung_mask * frangi_mask
        final = volume * mask
        final = helper_functions.create_binary_mask(final)
        final = util.invert(final)
        return final