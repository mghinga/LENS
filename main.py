import argparse
import os
from operator import gt
import numpy as np
from matplotlib.image import imsave

import reader

HAS_GROUND_TRUTH = False
TWO_DIM = False


def separate_ground_truth(filenames: list) -> list:
    ground_truth = []
    images = []
    for f in filenames:
        if 'mask' in f:
            ground_truth.append(f)
        else:
            images.append(f)
    return images, ground_truth


def check_directory(directory: str) -> bool:
    for file in os.listdir(directory):
        if file.endswith('.nii'):
            return True
    return False


def walk_files(directory: str) -> list:
    paths = []
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            filename = os.path.join(dirpath, filename)
            root, ext = os.path.splitext(filename)
            if ext == '.nii':
                paths.append(filename)
    return paths


def main():
    dataset_name = ''
    # take in and process arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--type", type=str,
                       help="Selection from k for Kassin et al, s for 3D segmented volumes, t for 2D segmentations, or x for exit.")
    options = parser.parse_args()

    footprint_size = 6

    if options.type == 'k':
        dataset_name = 'Kassin et al'
        directory = input("Enter file path: ")
        while not os.path.isdir(directory) or not check_directory(directory):
            directory = input(
                "Directory invalid. Enter a valid directory or press 'h' for help.: ")
            if directory == 'h':
                print("The data from Kassin et al must be requested from the authors. This software expects that all files have .nii file extension.")
        paths = walk_files(directory)
    elif options.type == 's':
        dataset_name = 'three dimensional'
        HAS_GROUND_TRUTH = True
        paths = walk_files('resources/three_dimensional')
    elif options.type == 't':
        dataset_name = 'two dimensional'
        HAS_GROUND_TRUTH = True
        TWO_DIM = True
        paths = walk_files('resources/two_dimensional')
    elif options.type == 'x':
        print('Exiting.')
        exit()
    else:
        print('Select only one of these options:')
        print(' --kassin for data from Kassin et al.')
        print(' --segmented_volumes from 3D volumes from https://medicalsegmentation.com/covid19/')
        print(' --two_dim for 2D segmentations from https://medicalsegmentation.com/covid19/')
        print('Exiting. Try again.')

    print('{} images found.'.format(len(paths)))
    print('Begin processing {} dataset...'.format(dataset_name))
    # set up file reader and analysis on volumes
    lung_volumes = []
    lesion_volumes = []
    no_processing_volumes = []
    if HAS_GROUND_TRUTH:
        paths, ground_truth = separate_ground_truth(paths)
        paths.sort()
        ground_truth.sort()
        for i in range(len(paths)):

            read = reader.Reader(
                paths[i], i, footprint_size, ground_truth[i], two_d=TWO_DIM)
            gt_name = 'output/two_dimensional/gt_' + str(i) + '_.png'
            imsave(gt_name, ground_truth[i], cmap='gray')
            lung_volumes.append(read.lung_volume)
            lesion_volumes.append(read.lesion_volume)
            no_processing_volumes.append(read.lesion_volume)
    else:
        for i in range(len(paths)):
            read = reader.Reader(paths[i], i, footprint_size)
            lung_volumes.append(read.lung_volume)
            lesion_volumes.append(read.lesion_volume)
            no_processing_volumes.append(read.lesion_volume)

    std_dev = np.std(lung_volumes)
    lung_avg = np.average(lung_volumes)
    median = np.median(lung_volumes)
    print(lung_volumes)
    print("Lung Volume Stats: ")
    print(' Mean Lung Volume: ', lung_avg)
    print(' Median Lung Volume: ', median)
    print(' Lung Volume Standard Deviation: ', std_dev)

    print('Lesion Volumes with Bronchial Tree Removal:')
    for i in range(len(lesion_volumes)):
        print(' --', lesion_volumes[i])

    print('Lesion Volumes without Bronchial Tree Removal: ')
    for i in range(len(no_processing_volumes)):
        print(' --', no_processing_volumes[i])


if __name__ == "__main__":
    main()
