ReadMe file for BMVC 2021 Dataset paper submission

Paper title: Corrosion Image Data Set for Automating Scientific Assessment of Materials

Code originally based on: link to original github 

License file found in:

Code based on: https://github.com/hysts/pytorch_image_classification  

Key folder/file descriptions:

BMVC_final_codebase

------requirements.txt
----------install all requirements in this file in a virtual environment
----------Experiments were run using torch 1.4.0, torchvision 0.5.0, using Python 3.8.5
----------Apex installation help: https://github.com/NVIDIA/apex 

------configs
----------yaml configuration files for all models

------pytorch_image_classification
----------config/defaults.py --> contains the default training configurations used, all augmentation parameters can be set in this file
----------transforms/transforms.py --> all augmentations used (and extra) are found here, additional ones can be added

------submission_files_examples
----------training
--------------these are example submission scripts to TRAIN the models, provided are the scripts which run the best performing augmentations provided in the paper
----------testing
--------------run these inside testing/ folder
--------------these are example submission scripts to TEST the models, provided are the scripts which run the best performing augmentations provided in the paper

------testing
------RUN these .py files while in the testing/ directory
----------configs
--------------yaml configuration files for all models, similar to other configs folder
----------pytorch_image_classification
--------------similar to the above version
----------evaluate.py
--------------this file is run when executing the testing scripts in submission_files_examples/testing/ for all models EXCEPT HRNet
----------evaluate_HRNet.py
--------------this file is run when executing the testing scripts in submission_files_examples/testing/ for HRNet model ONLY
----------mlp_r18_eva.py
--------------run this file for evaluating pretrained ResNet18 models with MLP classifier (results shown in supplementary material)
----------mlp_r50_eva.py
--------------run this file for evaluating pretrained ResNet50 models with MLP classifier (results shown in supplementary material)
----------r18_eva.py
--------------run this file for evaluating pretrained ResNet18 models with linear classifier (results shown in supplementary material)
----------r50_eva.py
--------------run this file for evaluating pretrained ResNet50 models with linear classifier (results shown in supplementary material)

------mlp_pretrained_baseline_r18.py
----------run this to train pretrained ResNet18 model with an MLP classifier (results shown in supplementary material)

------mlp_pretrained_baseline_r50.py
----------run this to train pretrained ResNet50 model with MLP classifier (results shown in supplementary material)

------pretrained_baseline_r18.py
----------run this to train pretrained ResNet18 model with linear classifier (results shown in supplementary material)

------pretrained_baseline_r50.py
----------run this to train pretrained ResNet50 model with linear classifier (results shown in supplementary material)

------train_o.py
----------main training script to train all models EXCEPT HRNet
----------this script runs when running the submission scripts found in submission_files_examples/training/

------train_o_HR.py
----------main training script to train HRNet ONLY
----------this script runs when running the submission script for HRNet found in submission_files_examples/training/


------pytorch-grad-cam-master
----------codes for visualizing final convolutional layer for each model (ResNet-18, ResNet-50, DenseNet, HRNet)