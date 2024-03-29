# -*- coding: utf-8 -*-
"""
Created on Sat May 21 10:17:03 2022

@author: Jean-
"""
# Packages
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import Image
from scipy.io import loadmat
import torch
import torchvision
import glob
import cv2

class CityscapeDataset(object):

    def __init__(self,root_img, root_mask ,  subset,factor, transforms_in=None,as_tensor = True):
        
        self.subset = subset
        self.root_img = root_img
        self.root_mask = root_mask
        #self.data = torchvision.datasets.Cityscapes(root,split=subset, mode='fine', target_type=['instance'], transform=None)
        #self.img_paths = glob.glob(root_img + subset + '/*/*_leftImg8bit.png')
        self.img_paths = glob.glob(root_img + 'leftImg8bit/' + subset + '/*/*_leftImg8bit.png')
        self.img_paths.sort()
        self.mask_paths = glob.glob(root_mask + 'gtfine/' + subset + '/*/*_gtFine_instanceIds.png')
        self.mask_paths.sort()
        self.transforms_in = transforms_in
        self.as_tensor = as_tensor
        self.factor =  factor
        print('num_images: ',len(self.img_paths))
        print('num_masks: ',len(self.mask_paths))
   
    def __len__(self):
       return int(len(self.img_paths )/ self.factor)
   
    def __getitem__(self, index):

        #img =plt.imread(self.img_paths[index])
        img = Image.open(self.img_paths[index])
        img = T.Resize((244,244))(img) 
        image_mask = Image.open(self.mask_paths[index])
        image_mask = T.Resize((244,244),interpolation = T.InterpolationMode.NEAREST)(image_mask) 
        # Compute IMage id
        image_id = torch.tensor([index])
        
        # Compute masks
        obj_ids = np.unique(image_mask)
        obj_ids = obj_ids[1:]  #cut Background
        
        #instance_classes = range(24,34)
        
        masks = []
        labels = np.zeros(np.sum(obj_ids>=1000),dtype=int)
        boxes = np.zeros((np.sum(obj_ids>=1000),4))
        j = 0
        for i in obj_ids:
            if i > 1000:
                maske = np.array(image_mask) == i
                #masks.append(maske)
                pos = np.where(maske)
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                
                if (xmax - xmin) * (ymax -ymin) > 0:
                    
                    boxes[j,:] = [xmin, ymin, xmax, ymax]
                    class_id = int(str(i)[:2])-23
                    labels[j] = class_id
                    masks.append(maske) #* class_id)
                    j = j + 1
                else:
                    labels = labels[:-1]
                    boxes = boxes[:-1]

        # Area der Bbox berchenen
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        
        
        # target values to Tensor
        if self.as_tensor:
            if len(labels) == 0:
                
                labels = torch.zeros(1, dtype=torch.int64)
                masks = torch.zeros(np.shape(image_mask), dtype=torch.uint8).unsqueeze(0)
                boxes = torch.as_tensor([0,0,1024,2048],dtype=torch.float32).unsqueeze(0)
                area = torch.as_tensor([1024*2048], dtype=torch.float32)               
            else:
                labels = torch.as_tensor(labels, dtype=torch.int64)
                masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)  #load if used for training
                boxes = torch.as_tensor(boxes, dtype=torch.float32)   
                area = torch.as_tensor(area, dtype=torch.float32)
        
        # is Crowd füllen
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

        # Abspeichern in Dictionarry
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] =  masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        
        if self.transforms_in is not None:
            img, target = self.transforms_in(img, target) 
            
           
        return img, target
                

if __name__ == '__main__':
    main()        
