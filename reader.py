import os
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

import volume, helper_functions


class Reader:
    def __init__(self, src_file, number, footprint_size, truth=None) -> None:
        self.src_file = src_file       
        self.file_dest = []
        self.voxel_x, self.voxel_y, self.voxel_z = -1, -1, -1
        self.volume = self.read_file()
        self.number = number
        self.footprint_size = footprint_size
        if truth is not None:
            truth_file = nib.load(truth)
            truth = np.array(truth_file.get_fdata())
            self.vol = volume.Volume(self.volume, self.voxel_x, self.voxel_y, self.voxel_z, self.number, self.footprint_size, truth)
        else:
            self.vol = volume.Volume(self.volume, self.voxel_x, self.voxel_y, self.voxel_z, self.number, self.footprint_size)
        self.lung_volume = self.vol.lung_volume
        self.lesion_volume = self.vol.lesion_volume
        self.size_lesion_volume = self.vol.size_lesion_volume
        self.no_processing_lesion_volume = self.vol.no_processing_lesions

    def read_file(self):
        if os.path.isfile(self.src_file):
            ext = os.path.splitext(self.src_file)[1]
            if ext == '.nii':
                file = nib.load(self.src_file)
                vol = np.array(file.get_fdata())
                self.voxel_x, self.voxel_y, self.voxel_z = self.find_niftii_dimensions(file)
        elif os.path.isdir(self.src_file):
            vol, px_spacing, slice_thickness = helper_functions.read_dicom_directory(self.src_file)
            self.voxel_x, self.voxel_y = px_spacing
            self.voxel_z = slice_thickness
        else:
            print("Can only read valid files/directories. Exiting ...")
            exit()
        return vol

    def find_niftii_dimensions(self, file):
        # units are mm
        px_dim = file.header["pixdim"]
        x, y, z = px_dim[1], px_dim[2], px_dim[3]
        return x, y, z
