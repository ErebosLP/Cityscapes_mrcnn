# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 14:31:45 2022

@author: Jean-

"""

import os
import sys
import torch
import numpy as np
# import torchvision
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# tensorboard
from torch.utils.tensorboard import SummaryWriter

from torch import nn

# main()
sys.path.append('../')
# from vision.references.detection.engine import train_one_epoch, evaluate
from engine import train_one_epoch, evaluate
import utils

# get_transform()
import transforms as T

from PIL import Image
import matplotlib.pyplot as plt

# plot endresults
import cv2
import matplotlib.patches as patches
import random

from City_dataloader import CityscapeDataset

def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def get_instance_segmentation_model(num_classes, checkpoint_path):

    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels

    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                        hidden_layer,
                                                        num_classes)
                                                       
    if checkpoint_path is not None:
        # model.load_state_dict(torch.load(checkpoint_path))#, map_location=lambda storage, loc: storage))
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])

        print('\tinitial_checkpoint = %s\n' % checkpoint_path) 
                          
    return model

def main():
    import time
    start_time = time.time()
    device = torch.device('cuda') 
    
    model = get_instance_segmentation_model(11,None)
    checkpoint = torch.load('C:/Users/Jean-/sciebo/Documents/Masterprojekt/Code/model/Cityscapes_model/model_Cityscapes_version2_numEpochs100.pth')
    model.load_state_dict(checkpoint)
    model.to(device)
    root = 'E:/Dataset_test/'
    
    
    dataset = CityscapeDataset(root,"train", get_transform(train=False))
    data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=1, shuffle=False, num_workers=1,collate_fn=utils.collate_fn)
    
    
    
    _, stat = evaluate(model, data_loader, device=device)
    print("--- %s seconds ---" % (time.time() - start_time))



if __name__ == '__main__':
    main()