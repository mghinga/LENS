import reader, helper_functions

class Demo:
    def __init__(self, demo_type:str) -> None:
        self.demo_type = demo_type
        src_file = 'resources/three_dimensional/three_dimensional_images/1.nii'
        ground_truth_src_file = 'resources/three_dimensional/three_dimensional_masks/1.nii'
        bronchial_tree_segmentation_src_file = 'bronchial_segmentations/segmented_volumes/1.p'
        if demo_type == 's':
            print("Begin 3D segmentation demo.")
            read = reader.Reader(src_file, 0, 4, ground_truth_src_file, False, True, bronchial_tree_segmentation_src_file)
            helper_functions.view_3D(read.truth)
            
        elif demo_type == 't':
            print("Begin 2D segmentation demo.")
            src_file = 'resources/two_dimensional/two_dimensional_images.nii/tr_im.nii'
            ground_truth_src_file = 'resources/two_dimensional/two_dimensional_masks.nii/tr_mask.nii'
            read = reader.Reader(src_file, 0, 6, ground_truth_src_file, True, True, '')
            
        else: 
            "This should not be possible. Impressive. Exiting."
            exit()

        lung_volumes = read.lung_volume
        lesion_volumes = read.lesion_volume
        no_processing = read.no_processing_lesion_volume
        print("Lung Volume: ", lung_volumes)
        print("Lesion Volume: ", lesion_volumes)
        print("Lesion Volume without Processing: ", no_processing)
        print("Lesion Percentage: ", str(lesion_volumes/lung_volumes))
        print("No Processing Percentage: ", str(no_processing/lung_volumes))

        