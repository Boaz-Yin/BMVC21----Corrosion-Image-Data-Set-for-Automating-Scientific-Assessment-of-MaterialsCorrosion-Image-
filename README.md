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

1. indoor_binary
2. indoor_ori

## Requirements

* Ubuntu (It's only tested on Ubuntu, so it may not work on Windows.)
* Python >= 3.7
* PyTorch >= 1.4.0
* # suggestion: conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
* torchvision
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

python3 train_o.py --config configs/imagenet/resnet34.yaml dataset.dataset_dir indoor_ori dataset.n_classes 6 train.base_lr 1e-3 train.weight_decay 5e-2 train.batch_size 64 validation.batch_size 1 scheduler.epochs 1 augmentation.use_gaussianblur True train.output_dir experiments/corrosion/ori/resnet34/aug_3
python3 train_o.py --config configs/imagenet/resnet34.yaml dataset.dataset_dir indoor_binary dataset.n_classes 6 train.base_lr 1e-3 train.weight_decay 5e-2 train.batch_size 64 validation.batch_size 1 scheduler.epochs 1 augmentation.use_gaussianblur True train.output_dir experiments/corrosion/binary/resnet34/aug_3

```
## Models

```bash
  Densenet
  Pyramidnet
  resnet18
  resnet34
  resnet50
  vgg
```
## Augmentations can be realized:

```bash
  use_random_crop: True / Flase
  use_gaussianblur: True / Flase
  use_colorjitter: True / Flase
  use_center_crop: True / Flase
  use_random_horizontal_flip: True / Flase
  use_cutout: True / Flase
  use_random_erasing: True / Flase
  use_dual_cutout: True / Flase
  use_mixup: True / Flase
  use_ricap: True / Flase
  use_cutmix: True / Flase
  use_label_smoothing: True / Flase
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
