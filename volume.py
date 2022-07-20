import numpy as np

import lesion, lung, bronchial_tree

VOL_TEST = True

class Volume:
    def __init__(self, src_file:str, volume:np.numarray, voxel_x, voxel_y, voxel_z, number, footprint_size, truth=None, file_type=".nii", target_blobs=None) -> None:
        self.volume = volume
        self.src_file = src_file
        self.px_height = self.volume.shape[0]
        self.px_width = self.volume.shape[1]

        self.voxel_x = voxel_x
        self.voxel_y = voxel_y
        self.voxel_z = voxel_z
        self.number = number
        self.footprint_size = footprint_size
        self.truth = truth
        self.file_dest = []
        self.file_type = file_type
        self.target_blobs = target_blobs
        self.lung_volume, self.lesion_volume, self.no_processing_lesions = self.process_volume()

    def process_volume(self):
        print('Processing slices and writing to disk...')
        lung_vol = lung.Lung(self.volume, self.px_height, self.px_width, self.voxel_x, self.voxel_y, self.voxel_z)
        meng_airway = bronchial_tree.BronchialTree(self.volume, lung_vol.segmentations, self.voxel_x, self.voxel_y, self.number)

        if self.truth is None:
            lesion_vol = lesion.Lesion(self.src_file, self.volume, meng_airway.lung_mask, meng_airway.cef_volume, self.voxel_x, self.voxel_y, self.voxel_z, self.footprint_size, self.number)
            no_processing = lesion.Lesion(self.src_file, self.volume, meng_airway.lung_mask, meng_airway.cef_volume, self.voxel_x, self.voxel_y, self.voxel_z, self.footprint_size, self.number)
        else:
            lesion_vol = lesion.Lesion(self.src_file, self.volume, meng_airway.lung_mask, meng_airway.cef_volume, self.voxel_x, self.voxel_y, self.voxel_z, self.footprint_size, self.number, self.truth)
            no_processing = lesion.Lesion(self.src_file, self.volume, meng_airway.lung_mask, meng_airway.cef_volume, self.voxel_x, self.voxel_y, self.voxel_z, self.footprint_size, self.number, self.truth)
        return lung_vol.lung_volume, lesion_vol.total_lesion_volume, no_processing.total_lesion_volume
        