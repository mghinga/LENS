import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

import helper_functions
# take in lesion segmention and divide into slices
# see if slices before and after have overlapping contours
# count voxels
# calculate volume
class LesionVolume():
    def __init__(self, lesion_segmentation:np.numarray, mm_x, mm_y, mm_z, truth:bool=False) -> None:
        self.lesion_segmentation = lesion_segmentation
        self.mm_x = mm_x
        self.mm_y = mm_y
        self.mm_z = mm_z
        self.truth = truth
        self.lesion_slices = helper_functions.create_slices(self.lesion_segmentation)
        self.find_lesions()





    def contours_intersect(self, cnt_ref, cnt_query):
        # better, more elegant solution 
        # from https://stackoverflow.com/questions/55641425/check-if-two-contours-intersect
        def ccw(A,B,C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
        ## Contour is a list of points
        ## Connect each point to the following point to get a line
        ## If any of the lines intersect, then break

        for ref_idx in range(len(cnt_ref)-1):
        ## Create reference line_ref with point AB
            A = cnt_ref[ref_idx][0]
            B = cnt_ref[ref_idx+1][0] 
        
            for query_idx in range(len(cnt_query)-1):
                ## Create query line_query with point CD
                C = cnt_query[query_idx][0]
                D = cnt_query[query_idx+1][0]
            
                ## Check if line intersect
                if ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D):
                    ## If true, break loop earlier
                    return True

    def find_lesions(self):
        overlapping_contours = []
        for i in range(1, len(self.lesion_slices)-1):
            current_np = np.array(self.lesion_slices[i], dtype=np.uint8)
            current = cv.normalize(current_np, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
            next_np = np.array(self.lesion_slices[i+1], dtype=np.uint8)
            next = cv.normalize(next_np, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
            # these are already binary, so no thresholding needed before finding contours
            current_contours, *_ = cv.findContours(current, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            next_contours, *_ = cv.findContours(next, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            # set up canvas for drawing contours
            current_rgb = np.zeros((current.shape[0], current.shape[1], 3), dtype=np.uint8)
            # sort through contours by size and whether they overlap

            slices = []
            if len(current_contours) > 0:
                for curr in current_contours:
                    curr_num_px = cv.contourArea(curr)
                    moments = cv.moments(curr)
                    cx = int(moments['m10']/(moments['m00']+ 1e-5))
                    cy = int(moments['m01']/(moments['m00']+ 1e-5))
                    curr_center = (cx, cy, i)
                    for n in next_contours:
                        if self.contours_intersect(curr, n):
                            moments = cv.moments(n)
                            nx = int(moments['m10']/(moments['m00']+ 1e-5))
                            ny = int(moments['m01']/(moments['m00']+ 1e-5))
                            n_center = (nx, ny, i+1)
                            cv.drawContours(current_rgb, [curr], 0, (0,0,255), -1)
                            overlaps = dict(overlapping_contours)
                            if overlaps.get(curr_center):
                                old_num_px = overlaps.get(curr_center)
                                overlapping_contours.remove((curr_center, old_num_px))
                                new_num_px = old_num_px + curr_num_px
                                overlapping_contours.append((n_center, new_num_px))
                            else:
                                overlapping_contours.append((n_center, curr_num_px))
            slices.append(current_rgb)
        lesion_volumes = []
        for cnt in overlapping_contours:
            centroid, area = cnt
            x, y, z = centroid
            volume = area * self.mm_x * self.mm_y
            volume = volume + (z*self.mm_z)
            lesion_volumes.append(volume/1000)
        lesion_volumes.sort(reverse=True)
        lesion_volume = set(lesion_volumes)
       
        print("Lesion volumes: ", lesion_volumes)
