
import torch
import torch.nn.functional as F

import skimage.io
import argparse
import numpy as np
import time
import os
import cv2


from aanet.nets import AANet
from utils import loadPyTorchModel, aanetPreprocess


# AANet Configuration for AANet+ trained on KITTI2015 dataset
max_disp = 192
feature_type = "ganet"
no_feature_mdconv = False
feature_pyramid = True
feature_pyramid_network = False
feature_similarity = "correlation"
num_downsample = 2
aggregation_type = "adaptive"
num_scales = 3
num_fusions = 6
num_stage_blocks = 1
num_deform_blocks = 3
no_intermediate_supervision = False
deformable_groups = 2
mdconv_dilation = 2
refinement_type = "hourglass"
# only_cost_volume = True
# only_cost_volume = True
pretrained_aanet = "./aanet/pretrained/aanet+_kitti15-2075aea1.pth"

# This pads the KITTI dataset to be divisible by 48
img_width = 1248
img_height = 384


def loadAANetModel(model, only_cost_volume):

    pretrained_aanet = model

    torch.backends.cudnn.benchmark = True

    # Use GPU as default
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    aanet = AANet(max_disp,
                       num_downsample=num_downsample,
                       feature_type=feature_type,
                       no_feature_mdconv=no_feature_mdconv,
                       feature_pyramid=feature_pyramid,
                       feature_pyramid_network=feature_pyramid_network,
                       feature_similarity=feature_similarity,
                       aggregation_type=aggregation_type,
                       num_scales=num_scales,
                       num_fusions=num_fusions,
                       num_stage_blocks=num_stage_blocks,
                       num_deform_blocks=num_deform_blocks,
                       no_intermediate_supervision=no_intermediate_supervision,
                       refinement_type=refinement_type,
                       mdconv_dilation=mdconv_dilation,
                       deformable_groups=deformable_groups,
                       only_cost_volume=only_cost_volume).to(device)


    if os.path.exists(pretrained_aanet):
        print('Loading pretrained AANet:', pretrained_aanet)
        loadPyTorchModel(aanet, pretrained_aanet, no_strict = False)

    else:
        print("No model found... using random initialization")

    
    if torch.cuda.device_count() > 1:
        print('=> Use %d GPUs' % torch.cuda.device_count())
        aanet = torch.nn.DataParallel(aanet)

    return aanet, device


def getCostVolume(model, left, right):
    aanet, device = loadAANetModel(model, True)

    # PyTorch uses channel_first format for images, so we must adjust our images to that format
    pair = aanetPreprocess(left, right)

    # Unsquueze expands dimension of tensor to be a batch of one [1, 3, H, W]
    left = pair['left'].unsqueeze(0).to(device)
    right = pair['right'].unsqueeze(0).to(device)


    # Pad
    ori_height, ori_width = left.size()[2:]
    if ori_height < img_height or ori_width < img_width:
        top_pad = img_height - ori_height
        right_pad = img_width - ori_width

        # Pad size: (left_pad, right_pad, top_pad, bottom_pad)
        left = F.pad(left, (0, right_pad, top_pad, 0))
        right = F.pad(right, (0, right_pad, top_pad, 0))


    # Do prediction
    with torch.no_grad():
        # cost_volume = aanet(left, right)[-1]
        cost_volume = aanet(left, right)[-1]

    # if cost_volume.size(-1) < left.size(-1):
    #     cost_volume = cost_volume.unsqueeze(1)  # [B, 1, H, W]
    #     cost_volume = F.interpolate(cost_volume, (left.size(-2), left.size(-1)),
    #                                 mode='bilinear') * (left.size(-1) / cost_volume.size(-1))
    #     cost_volume = cost_volume.squeeze(1)  # [B, H, W]

    # # Crop
    # if ori_height < img_height or ori_width < img_width:
    #     if right_pad != 0:
    #         cost_volume = cost_volume[:, top_pad:, :-right_pad]
    #     else:
    #         cost_volume = cost_volume[:, top_pad:]

    # Get shape from D x H x W to H x W x D
    cost_volume = cost_volume.squeeze(0).detach().cpu().numpy()
    cost_volume = np.moveaxis(cost_volume, 0, 2)

    return cost_volume


def getDisparityMap(model, left, right):
    aanet, device = loadAANetModel(model, False)

    # PyTorch uses channel_first format for images, so we must adjust our images to that format
    pair = aanetPreprocess(left, right)

    # Unsquueze expands dimension of tensor to be a batch of one [1, 3, H, W]
    left = pair['left'].unsqueeze(0).to(device)
    right = pair['right'].unsqueeze(0).to(device)


    # Pad
    ori_height, ori_width = left.size()[2:]
    if ori_height < img_height or ori_width < img_width:
        top_pad = img_height - ori_height
        right_pad = img_width - ori_width

        # Pad size: (left_pad, right_pad, top_pad, bottom_pad)
        left = F.pad(left, (0, right_pad, top_pad, 0))
        right = F.pad(right, (0, right_pad, top_pad, 0))


    # Do prediction
    with torch.no_grad():
        disparity_map = aanet(left, right)[-1]

    if disparity_map.size(-1) < left.size(-1):
        disparity_map = disparity_map.unsqueeze(1)  # [B, 1, H, W]
        disparity_map = F.interpolate(disparity_map, (left.size(-2), left.size(-1)),
                                    mode='bilinear') * (left.size(-1) / disparity_map.size(-1))
        disparity_map = disparity_map.squeeze(1)  # [B, H, W]

    # Crop
    if ori_height < img_height or ori_width < img_width:
        if right_pad != 0:
            disparity_map = disparity_map[:, top_pad:, :-right_pad]
        else:
            disparity_map = disparity_map[:, top_pad:]

    disparity_map = disparity_map.detach().cpu().numpy()

    disparity_map = np.moveaxis(disparity_map, 0, 2)

    return disparity_map


def upSample(model, disparity_map, left, right):
    aanet, device = loadAANetModel(model)

    pair = aanetPreprocess(left, right)

    left = pair['left'].unsqueeze(0).to(device)
    right = pair['right'].unsqueeze(0).to(device)


    disparity_map = np.moveaxis(disparity_map, -1, 0)

    # Convert numpy to torch tensor
    disparity_pyramid = torch.from_numpy(disparity_map)


    # Upsample the disparity map
    disparity_pyramid += aanet.disparity_refinement(left, right, disparity_pyramid[-1])

    print(disparity_pyramid.size())

    disparity_pyramid = disparity_pyramid.detach().cpu().numpy()
    # disparity_pyramid = np.moveaxis(disparity_pyramid, 0, 2)


    return disparity_pyramid


# print("Initializing AANet")
# left = cv2.imread("test/left/0001.png")
# left = cv2.GaussianBlur(left, (3,3), 0, 0)
# right = cv2.imread("test/right/0001.png")
# right = cv2.GaussianBlur(right, (3,3), 0, 0)

# cost_volume = getCostVolume(pretrained_aanet, left, right)

# # print("\n\nCost Volume Shape:", cost_volume.size())
# print(cost_volume.shape)
    
