# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 13:22:56 2022

@author: Jean-
"""
import os
import torch
import torchvision
from collections import OrderedDict
import segmentation_models_pytorch as smp
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

num_classes = 11
model = smp.Unet(encoder_name='resnet50', encoder_weights=None, classes=16, activation='sigmoid')
source_path = 'C:/Users/Jean-/Desktop/Anno/V1/'
checkpoint = torch.load(source_path + 'max_valid_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

mask_model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
# get number of input features for the classifier
in_features = mask_model.roi_heads.box_predictor.cls_score.in_features

# replace the pre-trained head with a new one
mask_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
# now get the number of input features for the mask classifier
in_features_mask = mask_model.roi_heads.mask_predictor.conv5_mask.in_channels

hidden_layer = 256
# and replace the mask predictor with a new one
mask_model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                    hidden_layer,
                                                    num_classes)
checkpoint2 = mask_model.state_dict()

keys_unet = checkpoint['model_state_dict'].keys()
resnet_checkpoint = OrderedDict()
import ipdb
for key in keys_unet:
    
    temp_key = key.split('encoder.')
    if temp_key[0] == '':
        if not ('num_batches_tracked' in key):
            new_key = 'backbone.body.' + temp_key[1]
            resnet_checkpoint[new_key] = checkpoint['model_state_dict'][key]
        
for key in  checkpoint2.keys():
    if not 'backbone.body' in key:
        resnet_checkpoint[key] = checkpoint2[key]
        
mask_model.load_state_dict(resnet_checkpoint)
torch.save({'model_state_dict': mask_model.state_dict()},source_path +'/contrastive_anno_v1.pth')         