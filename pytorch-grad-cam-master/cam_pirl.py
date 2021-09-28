import argparse
import cv2
import numpy as np
import torch
#from torchvision import models

from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM

from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='/Users/tiger_yin/Documents/Corrosion_Classification_/Corrosion_Classification_Pipeline_using_Pytorch-master-e66f71eb332685d9bf9300ec6619c27961105f32/test/',
                        help='Input image path')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument('--eigen_smooth', action='store_true',
                        help='Reduce noise by taking the first principle componenet'
                             'of cam_weights*activations')
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=['gradcam', 'gradcam++',
                                 'scorecam', 'xgradcam',
                                 'ablationcam', 'eigencam',
                                 'eigengradcam', 'layercam'],
                        help='Can be gradcam/gradcam++/scorecam/xgradcam'
                             '/ablationcam/eigencam/eigengradcam/layercam')
    parser.add_argument('--model_path', type=str, default='/Users/tiger_yin/Documents/Corrosion_Classification_/Corrosion_Classification_Pipeline_using_Pytorch-master-e66f71eb332685d9bf9300ec6619c27961105f32/pytorch-grad-cam-master/PIRL.pth',
                        help='Input model path')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


if __name__ == '__main__':
    """ python cam.py -image-path <path_to_image>
    Example usage of loading an image, and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """

    args = get_args()
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM}

    #model = models.resnet50(pretrained=True)
    checkpoint = torch.load(args.model_path, map_location='cpu')

    from models.pirl import RGBSingleHead
    model = RGBSingleHead()

    # import timm
    # model = timm.create_model('hrnet_w18', pretrained=False, num_classes=5)

    checkpoint = torch.load(args.model_path, map_location='cpu')
    state_dict = checkpoint['model']
    from collections import OrderedDict

    encoder_state_dict = OrderedDict()
    for k, v in state_dict.items():
        k = k.replace('module.', '')
        if 'encoder' in k:
            k = k.replace('encoder.', '')
            encoder_state_dict[k] = v
    model.encoder.load_state_dict(encoder_state_dict)

    model.load_state_dict(checkpoint['model'])

    model.eval()

    # target_layer = model.final_layer

    target_layer = model.stage4[-1]

    # Choose the target layer you want to compute the visualization for.
    # Usually this will be the last convolutional layer in the model.
    # Some common choices can be:
    # Resnet18 and 50: model.layer4[-1]
    # VGG, densenet161: model.features[-1]
    # mnasnet1_0: model.layers[-1]
    # You can print the model to help chose the layer
    #
    #
    # print(dir(model.final_layer))
    #
    # print(target_layer)

    cam = methods[args.method](model=model,
                               target_layer=target_layer,
                               use_cuda=args.use_cuda)

    import os
    # import xlsxwriter


    outlist = []

    for folder in os.listdir(args.image_path):

        print(folder)

        imglist = os.listdir(os.path.join(args.image_path,folder))

        print(imglist)

        for img in imglist:

            src_image_name = os.path.join(args.image_path,folder, img)

            print(src_image_name)

            rgb_img = cv2.imread(src_image_name, 1)[:, :, ::-1]
            rgb_img = cv2.resize(rgb_img,(512,512))

            rgb_img = np.float32(rgb_img) / 255
            input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])

            out = model(input_tensor)
            # print(out)

            _, index = torch.max(out, 1)

            print(index)

            # If None, returns the map for the highest scoring category.
            # Otherwise, targets the requested category.
            target_category = None

            # AblationCAM and ScoreCAM have batched implementations.
            # You can override the internal batch size for faster computation.
            cam.batch_size = 32

            grayscale_cam = cam(input_tensor=input_tensor,
                                target_category=target_category,
                                aug_smooth=args.aug_smooth,
                                eigen_smooth=args.eigen_smooth)

            # Here grayscale_cam has only one image in the batch
            grayscale_cam = grayscale_cam[0, :]

            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

            # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
            cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
            #
            # gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
            # gb = gb_model(input_tensor, target_category=target_category)
            #
            # cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
            # cam_gb = deprocess_image(cam_mask * gb)
            # gb = deprocess_image(gb)

            # cv2.imwrite(f'{args.method}_cam.jpg', cam_image)

            outlist.append([folder, img, int(index.numpy() + 5)])

            cv2.imwrite(os.path.join('/Users/tiger_yin/Documents/Corrosion_Classification_/Corrosion_Classification_Pipeline_using_Pytorch-master-e66f71eb332685d9bf9300ec6619c27961105f32/pytorch-grad-cam-master/save/jgden',folder,img), cam_image)


    print(outlist)

    import json
    with open("/Users/tiger_yin/Documents/Corrosion_Classification_/Corrosion_Classification_Pipeline_using_Pytorch-master-e66f71eb332685d9bf9300ec6619c27961105f32/pytorch-grad-cam-master/save/densenet.txt", "w") as fp:
        json.dump(outlist, fp)


    #
    # with open("/Users/tiger_yin/Documents/Corrosion_Classification_/Corrosion_Classification_Pipeline_using_Pytorch-master-e66f71eb332685d9bf9300ec6619c27961105f32/pytorch-grad-cam-master/save/hrnet.txt", "r") as fp:
    #    b = json.load(fp)

    # cv2.imwrite(f'{args.method}_gb.jpg', gb)
    # cv2.imwrite(f'{args.method}_cam_gb.jpg', cam_gb)

    # cv2.imwrite(cam_image)
    # cv2.imwrite(gb)
    # cv2.imwrite(cam_gb)