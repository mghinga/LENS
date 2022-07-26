# LENS
## Introduction
LENS (**L**ung l**E**sio**N** **S**egmentation) is for fully-automated segmentation of lesions in COVID-19-positive lungs. LENS uses traditional image processing methods instead of deep learning in order to avoid having to acquire labeled data. It can handle both 3D volumes and 2D slices in NIFTI format. It works in three phases:
1. LENS segments the whole lung from the CT scan. Depending on the size of the file, this takes a few minutes.
2. LENS segments the bronchial tree. This process takes upwards of 25 hours on a desktop computer. It is advised to save the output file using pickle so that this step does not need to be rerun should you wish to segment the same lung again.
3. LENS removes the bronchial tree from the lung. This is the step at which manual tuning may take place if necessary. Depending on the size of the file, it runs in a few minutes.

## Modes
LENS has four modes:
- Demo: Used for running a sample of either the 2D or 3D segmentations found [here](https://medicalsegmentation.com/covid19/). The demo feature takes advantage of the provided bronchial segmentation. It is recommended to try this step first because phase 2 can take a long time to run.
- Two Dimensional: Used for running the 2D segmentations available [here](https://medicalsegmentation.com/covid19/). It is too large to put on this repo, so be aware that it at the time of writing some segmentations are empty. Those are skipped.
- Three Dimensional: Used for running the 3D segmentations available [here](https://medicalsegmentation.com/covid19/). When using this, keep in mind that the execution time is > 24 hours.
- Kassin et al: One of the datasets on which LENS was tested, from Kassin, et al. *Generalized Chest CT and Lab Curves Throughout the Course of COVID-19*, is not publicly available. Should you request their data, this mode is used to run LENS on those images.

## Command Line Arguments
To run in DEMO mode, use the following command: `python main.py -t d` and follow the prompts.
To run in TWO DIMENSIONAL mode, use the following command: `python main.py -t t`
To run in THREE DIMENSIONAL mode, use the following command: `python main.py -t s`
To run in KASSIN ET AL. mode, use the following command: `python main.py -t k`
