'''
Implementation (with minor modifications) of Meng, Kitasaka, Numura, Oda, Ueno, and Mori "Automatic Segmentation
of airway tree based on local intensity filter and machine learning technique in 3D chest CT volume" (2016).
https://doi.org/10.1007/s11548-016-1492-2
At a high level, the algorithm works as follows:
1. Load CT volume
2. Preprocessing (unsharp mask filter)
3. Hessian analysis (specifically Sato)
4. Multiscale cavity enhancement filter (CFE)
5. Reduce False Positives with an SVM -- NOTE: This is not being implemented here because it is (possibly) not necessary.
6. Graph Cut Algorithm -- NOTE: Also not implemented here.

The result should be a segmentation of the bronchus region.
'''

import statistics, math, sys, pickle
from skimage.filters import unsharp_mask, sato
from skimage import util
from skimage.transform import rescale

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import helper_functions

class BronchialTree():
    def __init__(self, volume, lung_segmentations, x_mm, y_mm, number, demo=False, bronchial_segmentation='') -> None:
        self.number = number
        self.volume = volume
        self.lung_vol = np.dstack(lung_segmentations)

        self.lung_mask = self.create_lung_mask()
        self.x_mm = x_mm
        self.y_mm = y_mm
        if len(self.volume.shape) > 2:
            self.slices = helper_functions.create_slices(self.volume)
        else:
            self.slices = [self.volume]
        self.morphometry_path = 'resources/morphometry.txt'
        self.morphometry_info = self.read_in_morphometry_info()
        self.avg_radii = self.calculate_radii_per_generation()
        print('Begin bronchial tree preprocessing step ...')
        self.preprocessed_slices, self.preprocessed_volume = self.preprocessing()
        print('Begin bronchial tree Hessian analysis ...')
        self.hessian_slices, self.hessian_volume = self.hessian_analysis()
        if demo and bronchial_segmentation != '':
            self.cef_volume = pd.read_pickle(bronchial_segmentation)
            cef_slices = helper_functions.create_slices(self.cef_volume)
            plt.imshow(cef_slices[len(cef_slices)//2], cmap='gray')
            plt.title('Bronchial Segmentation')
            plt.show()
        else:
            if len(self.volume.shape) > 2:
                print('Begin bronchial tree cavity enhancement filter ...')
                self.cef_volume = self.cavity_enhancement_filter()
            else:
                self.cef_volume = self.hessian_volume
                self.cef_volume = self.cavity_enhancement_filter()


    def create_lung_mask(self):
        # binary threshold
        binary = helper_functions.create_binary_mask(self.lung_vol)
        return binary       


    def calculate_radii_per_generation(self):
        morphometry_radii = self.morphometry_info.loc[:, ['generation', 'radius_mm']]
        grouped_radii = morphometry_radii.groupby('generation')['radius_mm'].apply(list)
        grouped_radii = grouped_radii.values.tolist()

        average_radii = []
        for i in range(len(grouped_radii)):
            avg_radius = statistics.fmean(grouped_radii[i])
            if avg_radius >= self.x_mm or avg_radius >= self.y_mm:
                average_radii.append(math.ceil(avg_radius))

        return list(set(average_radii))

    def calculate_per_tube(self, row):
        row_list = row.tolist()
        num_tubes = row_list[1]
        total_area_mm = row_list[6]*100
        return total_area_mm / num_tubes

    def read_in_morphometry_info(self):
        data = pd.read_csv(self.morphometry_path, sep=" ", header=None)
        data.columns = ["generation", "num_tubes", "length_cm", "diameter_cm", "branching_angle", "gravity_angle", "total_area_cm^2", 'volume_cm^3', 'cumulative_volume_cm^3']
        # convert to mm from cm
        data["total_area_cm^2"] = data["total_area_cm^2"].apply(lambda x: x*100)
        data['length_cm'] = data['length_cm'].apply(lambda x: x*10)
        data['diameter_cm'] = data['diameter_cm'].apply(lambda x: x*10)
        # convert diameter to radius
        data['diameter_cm'] = data['diameter_cm'].apply(lambda x: x/2)
        data.rename(columns={'generation':'generation', "num_tubes":'num_tubes', "length_cm":'length_mm', "diameter_cm":"radius_mm", "branching_angle":'branching_angle',
         "gravity_angle":'gravity_angle', "total_area_cm^2":"average_area_per_tube_mm^2", 'volume_cm^3':'volume_mL', 'cumulative_volume_cm^3':'cumulative_volume_mL'}, inplace=True)
        # stats for total area and volume measurements
        # calculate area per tube
        data['average_area_per_tube_mm^2'] = data.apply(self.calculate_per_tube, axis=1)
        return data

    def preprocessing(self):
        '''
        The preprocessing step defined in the paper is to use an unsharp mask. No radius or amount were
        defined in the paper. The values herein are based on manual tuning.
        '''
        if len(self.slices) > 1:
            preprocessed_slices = []
            for i in reversed(range(len(self.slices))):
                unsharp = unsharp_mask(self.slices[i], radius=5, amount=2)
                preprocessed_slices.append(unsharp)
            preprocessed_volume = np.dstack(preprocessed_slices)
        else:
            unsharp = unsharp_mask(self.volume, radius=5, amount=2)
            return None, unsharp
        return preprocessed_slices, preprocessed_volume

    def hessian_analysis(self):
        '''
        Returns slices (for viewing) and volume (for viewing and further processing) that have had a Hessian
        matrix applied to it in which the function is a Gaussian filter.
        '''

        if self.preprocessed_volume is not None:
            hessian_vol = sato(self.preprocessed_volume, sigmas=range(1, 3), black_ridges=False)
            return None, hessian_vol
        else:
            hessian_vol = sato(self.preprocessed_volume, sigmas=range(1, 3), black_ridges=False)
            hessian_slices = []
            for i in range(len(self.preprocessed_slices)):
                hessian_img = sato(self.preprocessed_slices[i], sigmas=range(1, 3), black_ridges=False)
                hessian_slices.append(hessian_img)
        return hessian_slices, hessian_vol

    def is_in_bounds(self, x:int, y:int, z:int)->bool:
        '''
        Returns true if voxel is within the Hessian volume.
        
        Args:
          x:int - x-coordinate
          y:int - y-coordinate
          z:int - z-coordinate
        '''
        shape = self.hessian_volume.shape
        if len(shape) == 2:
            cols, rows = self.hessian_volume.shape
            if x < cols and y < rows:
                return True
        else:
            cols, rows, plane = self.hessian_volume.shape
            if x < cols and y < rows and z < plane:
                return True
        return False

    def calculate_P(self, x:int, y:int, z:int, i:int, j:int, k:int, r_1:int, r_2:int):
        '''
        Returns P = abs(IntensityOfVoxel_(x-ir_1, y-jr_1, z-kr_1) - IntensityOfVoxel_(x+ir_2, y+jr_2, z+kr_2)) or a
        negative value if out of bounds.
        
        Args:
          i:int, x-offset
          j:int, y-offset
          k:int, z-offset
          r_1:int, radius 1
          r_2:int, radius 2
        '''
        p_term = -1

        first_term_x = x-i*r_1
        first_term_y = y-j*r_1
        first_term_z = z-k*r_1

        second_term_x = x+i*r_2
        second_term_y = y+j*r_2
        second_term_z = z+k*r_2

        if z > -1:
            if self.is_in_bounds(first_term_x, first_term_y, first_term_z) and self.is_in_bounds(second_term_x, second_term_y, second_term_z):
                p_term = abs(self.hessian_volume[first_term_x, first_term_y, first_term_z] - self.hessian_volume[second_term_x, second_term_y, second_term_z])
        else:
            if self.is_in_bounds(first_term_x, first_term_y, -1) and self.is_in_bounds(second_term_x, second_term_y, -1):
                p_term = abs(self.hessian_volume[first_term_x, first_term_y] - self.hessian_volume[second_term_x, second_term_y])

        return p_term

    def calculate_L(self, x:int, y:int, z:int, i:int, j:int, k:int, r_1:int, r_2:int):
        '''
        Returns L = IntensityOfVoxel_(x-ir_1, y-jr_1, z-kr_1) - 2(IntensityOfVoxel_(x, y, z) + IntensityOfVoxel_(x+ir_2, y+jr_2, z+kr_2))
        or the minimum possible value if out of bounds.

        Args:
          i:int, x-offset
          j:int, y-offset
          k:int, z-offset
          r_1:int, radius 1
          r_2:int, radius 2
        '''

        l_term = sys.float_info.min

        first_term_x = x-i*r_1
        first_term_y = y-j*r_1
        first_term_z = z-k*r_1

        third_term_x = x+i*r_2
        third_term_y = y+j*r_2
        third_term_z = z+k*r_2

        if z == -1:
            first_term_z = -1
            third_term_z = -1

        if self.is_in_bounds(first_term_x, first_term_y, first_term_z) and self.is_in_bounds(x, y, z) and self.is_in_bounds(third_term_x, third_term_y, third_term_z):
            if z > -1:
                first_term = self.hessian_volume[first_term_x, first_term_y, first_term_z]
                second_term = 2*self.hessian_volume[x, y, z]
                third_term = self.hessian_volume[(x+i*r_2), (y+j*r_2), (z+k*r_2)]
            else:
                first_term = self.hessian_volume[first_term_x, first_term_y]
                second_term = 2*self.hessian_volume[x, y]
                third_term = self.hessian_volume[(x+i*r_2), (y+j*r_2)]
            l_term = first_term - second_term + third_term
        return l_term

    def cavity_enhancement_filter(self):
        '''
        Cavity Enhancement Filter (Hirano, Tachibana, and Kido 2011) finds the bronchial
        wall, which helps mitigate the Partial Volume Effect (PVE). 

        TODO: Limit this to only be called within the lung.
        '''
        if self.preprocessed_slices is not None:
            cols, rows, plane = self.hessian_volume.shape

            for c in range(cols-1):
                for r in range(rows-1):
                    for p in range(plane-1):
                        max_differences = []
                        for i in [-1, 0, 1]:
                            for j in [-1, 0, 1]:
                                for k in [-1, 0, 1]:
                                    max_difference_p_l = sys.float_info.min
                                    for r_1 in self.avg_radii:
                                        for r_2 in self.avg_radii:
                                            l_term = self.calculate_L(x=c, y=r, z=p, i=i, j=j, k=k, r_1=r_1, r_2=r_2)
                                            p_term = self.calculate_P(x=c, y=r, z=p, i=i, j=j, k=k, r_1=r_1, r_2=r_2)
                                            if p_term >= 0.0 and l_term > sys.float_info.min:
                                                difference = l_term - p_term
                                                if difference > max_difference_p_l:
                                                    max_difference_p_l = difference
                                    max_differences.append(max_difference_p_l)
                        self.hessian_volume[c, r, p] = sum(max_differences)
        else:
            cols, rows = self.hessian_volume.shape
            for c in range(cols-1):
                for r in range(rows-1):
                    max_differences = []
                    for i in [-1, 0, 1]:
                        for j in [-1, 0, 1]:
                            max_difference_p_l = sys.float_info.min
                            for r_1 in self.avg_radii:
                                for r_2 in self.avg_radii:
                                    l_term = self.calculate_L(x=c, y=r, z=-1, i=i, j=j, k=-1, r_1=r_1, r_2=r_2)
                                    p_term = self.calculate_P(x=c, y=r, z=-1, i=i, j=j, k=-1, r_1=r_1, r_2=r_2)
                                    if p_term >= 0.0 and l_term > sys.float_info.min:
                                        difference = l_term - p_term
                                        if difference > max_difference_p_l:
                                            max_difference_p_l = difference
                            max_differences.append(max_difference_p_l)
                    self.hessian_volume[c, r] = sum(max_differences)
                    
        return self.hessian_volume



