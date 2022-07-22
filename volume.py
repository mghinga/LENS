import numpy as np

import lesion, lung, bronchial_tree

VOL_TEST = True

class Volume:
    def __init__(self, src_file:str, volume:np.numarray, voxel_x, voxel_y, voxel_z, number, footprint_size, two_d=False, truth=None, demo=False, bronchial_segmentation='') -> None:
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
        self.two_d = two_d
        self.file_dest = []
        self.demo = demo
        self.bronchial_segmentation = bronchial_segmentation
        self.lung_volume, self.lesion_volume, self.no_processing_lesions = self.process_volume()

    def process_volume(self):
        print('Begin segmentation...')
        lung_vol = lung.Lung(self.volume, self.px_height, self.px_width, self.voxel_x, self.voxel_y, self.voxel_z, self.demo)
        meng_airway = bronchial_tree.BronchialTree(self.volume, lung_vol.segmentations, self.voxel_x, self.voxel_y, self.number, self.demo, self.bronchial_segmentation)

        if self.truth is None:
            lesion_vol = lesion.Lesion(self.src_file, self.volume, meng_airway.lung_mask, meng_airway.cef_volume, self.voxel_x, self.voxel_y, self.voxel_z, self.footprint_size, self.number, self.demo)
            no_processing = lesion.Lesion(self.src_file, self.volume, meng_airway.lung_mask, meng_airway.cef_volume, self.voxel_x, self.voxel_y, self.voxel_z, self.footprint_size, self.number, self.demo, no_processing=True)
        else:
            lesion_vol = lesion.Lesion(self.src_file, self.volume, meng_airway.lung_mask, meng_airway.cef_volume, self.voxel_x, self.voxel_y, self.voxel_z, self.footprint_size, self.number, self.demo, self.two_d, self.truth)
            no_processing = lesion.Lesion(self.src_file, self.volume, meng_airway.lung_mask, meng_airway.cef_volume, self.voxel_x, self.voxel_y, self.voxel_z, self.footprint_size, self.number, self.demo, self.two_d, self.truth, no_processing=True)
        return lung_vol.lung_volume, lesion_vol.total_lesion_volume, no_processing.total_lesion_volume
        