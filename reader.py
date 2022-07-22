import os
from turtle import clear
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

import volume
import helper_functions


class Reader:
    def __init__(self, src_file, number, footprint_size, truth, two_d=False, demo=False, bronchial_segmentation='') -> None:
        self.src_file = src_file
        self.file_dest = []
        self.voxel_x, self.voxel_y, self.voxel_z = -1, -1, -1
        self.volume = self.read_file(src_file)
        self.number = number
        self.footprint_size = footprint_size
        self.two_d = two_d
        self.demo = demo
        self.bronchial_segmentation = bronchial_segmentation
        self.truth = truth
        if two_d:
            self.truth = self.read_file(truth)
            self.volumes = helper_functions.create_slices(self.volume)
            self.truth = helper_functions.create_slices(self.truth)
            volumes = []
            for i in range(len(self.volumes)):
                print('Processing Slice Number: ', i)
                vol = volume.Volume(self.src_file, self.volumes[i], self.voxel_x, self.voxel_y, 0, i, self.footprint_size, self.truth[i], self.demo, self.bronchial_segmentation)
                self.lesion_volume = 0
                self.lesion_volume = 0
                self.no_processing_lesion_volume = 0
        else:
            if truth is not None:
                self.truth = self.read_file(truth)
                self.vol = volume.Volume(self.src_file, self.volume, self.voxel_x, self.voxel_y, self.voxel_z, self.number, self.footprint_size, self.two_d, self.truth, self.demo, self.bronchial_segmentation)
            else:
                self.vol = volume.Volume(self.src_file, self.volume, self.voxel_x, self.voxel_y, self.voxel_z, self.number, self.footprint_size, self.two_d, self.truth, self.demo, self.bronchial_segmentation)
            self.lung_volume = self.vol.lung_volume
            self.lesion_volume = self.vol.lesion_volume
            self.no_processing_lesion_volume = self.vol.no_processing_lesions

    def read_file(self, src):
        if os.path.isfile(src):
            ext = os.path.splitext(src)[1]
            if ext == '.nii':
                file = nib.load(src)
                vol = np.array(file.get_fdata())
                self.voxel_x, self.voxel_y, self.voxel_z = self.find_niftii_dimensions(file)
        else:
            print("Can only read valid files/directories. Exiting ...")
            exit()
        return vol

    def find_niftii_dimensions(self, file):
        # units are mm
        px_dim = file.header["pixdim"]
        x, y, z = px_dim[1], px_dim[2], px_dim[3]
        return x, y, z
