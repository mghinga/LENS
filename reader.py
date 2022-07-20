import os
import nibabel as nib
import numpy as np

import volume, helper_functions


class Reader:
    def __init__(self, src_file, number, footprint_size, truth, two_d=False,) -> None:
        self.src_file = src_file       
        self.file_dest = []
        self.voxel_x, self.voxel_y, self.voxel_z = -1, -1, -1
        self.volume = self.read_file(src_file)
        self.number = number
        self.footprint_size = footprint_size
        self.two_d = two_d
        if two_d:
            self.truth = self.read_file(truth)
            self.volumes = helper_functions.create_slices(self.volume)
            self.truth = helper_functions.create_slices(self.truth)
            volumes = []
            for i in range(78,len(self.volumes)):
                vol = volume.Volume(self.src_file, self.volumes[i], self.voxel_x, self.voxel_y, 0, i, self.footprint_size, self.truth[i])
                self.lesion_volume = 0
                self.lesion_volume = 0
                self.no_processing_lesion_volume = 0
        else:
            if truth is not None:
                truth_file = self.read_file(truth)
                truth = np.array(truth_file.get_fdata())
                self.vol = volume.Volume(self.src_file, self.volume, self.voxel_x, self.voxel_y, self.voxel_z, self.number, self.footprint_size, truth)
            else:
                self.vol = volume.Volume(self.src_file, self.volume, self.voxel_x, self.voxel_y, self.voxel_z, self.number, self.footprint_size)
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
        elif os.path.isdir(src):
            vol, px_spacing, slice_thickness = helper_functions.read_dicom_directory(src)
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
