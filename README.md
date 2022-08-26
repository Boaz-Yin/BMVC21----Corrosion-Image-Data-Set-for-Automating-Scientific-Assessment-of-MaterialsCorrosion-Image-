# An Experimental Pipeline for Corrosion Image Classification using Pytorch

Biao Yin, Worcester Polytechnic Institute


## What is it for

This pipeline enables highly-efficient implementation to corrosion classification experiments. Multiple CNN-based models with multiple augmentation methods are applied to assess corrosion images and the corresponding segmentations.

Accuracy, loss and confusion matrix, validation groundtruth, and validation predictions are printed for the sake of model evalution.
You can also open tensorboard to view train images (augumented and not-augmented), validation images, model parameters, train/validation accuracy and train/validation loss.

If you have any questions, please feel free to contact: byin@wpi.edu

Enjoy : )

01/02/2020

## Data folders
```bash
#Download from https://arl.wpi.edu/corrosion_dataset/#
DATA_SET_FOR_RELEASE
```

## Requirements

* Ubuntu (It's only tested on Ubuntu, so it may not work on Windows.)
* Python >= 3.7
* PyTorch >= 1.4.0
* torchvision
* # pytorch install suggestion: conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
* [NVIDIA Apex](https://github.com/NVIDIA/apex)

```bash
pip install -r requirements.txt
```

## Example Sbatch Submission on Turing Machine 

```bash
#!/bin/bash
#
#SBATCH -N 1                      # number of nodes
#SBATCH -n 16                      # number of cores
#SBATCH  --mem=120G                  # memory pool for all cores
#SBATCH -t 1-00:00                 # time (D-HH:MM)
#SBATCH -C T4
#SBATCH --gres=gpu:1             # number of GPU
#SBATCH --job-name=test
#SBATCH -o slurm-test-output      # STDOUT
#SBATCH -e slurm-test-error       # STDERR
#SBATCH --mail-type=END,FAIL      # notifications for job done & fail
#SBATCH --mail-user=XXX@XXX.XXX

cd ~/my_pytorch_classification_corrosion

source env/bin/activate

# An example from submission_files_examples/training/ResNet_50.txt

python3 train_o.py --config configs/imagenet/resnext50_32x4d.yaml dataset.dataset_dir DATA_SET_FOR_RELEASE/DATA_SET_FOR_RELEASE/renamed/cross_val_1  dataset.n_classes 5 train.base_lr 1e-3 train.weight_decay 5e-2 train.batch_size 32 validation.batch_size 1 scheduler.epochs 2000 scheduler.warmup.type 'exponential' scheduler.type 'cosine' augmentation.use_colorjitter True augmentation.colorjitter.bright_1 1.5 augmentation.colorjitter.bright_2 2.0 augmentation.colorjitter.contrast_1 0.5 augmentation.colorjitter.contrast_2 1.5 augmentation.colorjitter.sat_1 0.5 augmentation.colorjitter.sat_2 1.5 augmentation.colorjitter.hue 0.5 augmentation.colorjitter.prob 0.25 augmentation.use_random_erasing True augmentation.random_erasing.prob 0.25 augmentation.random_erasing.area_ratio_range_1 0.05 augmentation.random_erasing.area_ratio_range_2 0.15 augmentation.random_erasing.max_attempt 5 augmentation.use_random_perspective True augmentation.random_perspective.distortion_scale 0.25 augmentation.random_perspective.prob 0.75 augmentation.use_randomresizecrop True augmentation.random_resize_crop.scale_1 0.3 augmentation.random_resize_crop.scale_2 0.7 augmentation.random_resize_crop.prob 0.25 augmentation.use_random_crop True augmentation.random_crop.padding 4 augmentation.random_crop.padding_mode 'constant' augmentation.random_crop.prob 0.50 train.output_dir scheduler_single_10cv_bestparam/ori/R50/crossval/combo/cv1

```

## Models

```bash
# check all the model submissionn files under folder: submission_files_examples
  Densenet
  Pyramidnet
  resnet18
  resnet34
  resnet50
  HR-NET
  vgg
```
## Augmentations can be realized:

```bash
  use_random_crop: True / False
  use_gaussianblur: True / False
  use_colorjitter: True / False
  use_center_crop: True / False
  use_random_horizontal_flip: True / False
  use_cutout: True / False
  use_random_erasing: True / False
  use_dual_cutout: True / False
  use_mixup: True / False
  use_ricap: True / False
  use_cutmix: True / False
  use_label_smoothing: True / False
``` 
## How to change detailed augumentation parameters:
```bash
cd pytorch_image_classification/config/defaults.py

Change the default settings here:

config.augmentation.random_crop = ConfigNode()
config.augmentation.random_crop.padding = 4
config.augmentation.random_crop.fill = 0
config.augmentation.random_crop.padding_mode = 'constant'

config.augmentation.gaussianblur = ConfigNode()
config.augmentation.gaussianblur.kernel_size = 27
config.augmentation.gaussianblur.sigma = 5

config.augmentation.colorjitter = ConfigNode()

config.augmentation.random_horizontal_flip = ConfigNode()
config.augmentation.random_horizontal_flip.prob = 0.5

config.augmentation.cutout = ConfigNode()
config.augmentation.cutout.prob = 1.0
config.augmentation.cutout.mask_size = 16
config.augmentation.cutout.cut_inside = False
config.augmentation.cutout.mask_color = 0
config.augmentation.cutout.dual_cutout_alpha = 0.1

config.augmentation.random_erasing = ConfigNode()
config.augmentation.random_erasing.prob = 0.5
config.augmentation.random_erasing.area_ratio_range = [0.02, 0.4]
config.augmentation.random_erasing.min_aspect_ratio = 0.3
config.augmentation.random_erasing.max_attempt = 20

config.augmentation.mixup = ConfigNode()
config.augmentation.mixup.alpha = 1.0

config.augmentation.ricap = ConfigNode()
config.augmentation.ricap.beta = 0.3

config.augmentation.cutmix = ConfigNode()
config.augmentation.cutmix.alpha = 1.0

config.augmentation.label_smoothing = ConfigNode()
config.augmentation.label_smoothing.epsilon = 0.1

config.tta = ConfigNode()
config.tta.use_resize = False
config.tta.use_center_crop = False
config.tta.resize = 256
```

## Model snapshot files
```bash
https://drive.google.com/drive/folders/1xpS24yrUclQ_8B_fAYKJ03nGq7xawnTy?usp=sharing
```

## LICENSE and citation requirement
MIT License © [2021] [Worcester Polytechnic Institute]
This corrosion image data set has been introduced by the paper, “Corrosion Image Data Set for Automating Scientific Assessment of Materials” published at the British Machine Vision Conference BMVC 2021. Please follow the instructions below for downloading, use and proper citation of the data set. Additional details on the data included on the download is found in the downloaded README file.

Cite this dataset:
@InProceedings{yin2021BMVC, author = {Yin, Biao and Josselyn, Nicholas and Considine, Thomas and Kelley, John and Rinderspacher, Berend and Jensen, Robert and Snyder, James and Zhang, Ziming and Rundensteiner, Elke},title = {Corrosion Image Data Set for Automating Scientific Assessment of Materials},booktitle = {British Machine Vision Conference (BMVC)},year = {2021}}

