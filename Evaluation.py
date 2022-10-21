import os
import torch
import numpy as np
from City_dataloader import CityscapeDataset
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from PIL import Image
from torchvision import transforms
import transforms as T

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

import cv2
import math

from engine import evaluate 
import utils

from sklearn.metrics import confusion_matrix

from scipy.io import savemat

def evaluate_sgmentation(masks, ground_truths):
    if len(masks.shape) > 2:
        mask = np.zeros(masks[0].shape)
        for i in range(0,masks.shape[0]):
            mask += masks[i]
    else:
        mask = masks        
    if len(ground_truths.shape) > 2:
        ground_truth = np.zeros(ground_truths[0].shape)
        for i in range(0,ground_truths.shape[0]):
            ground_truth += ground_truths[i]
    else:
        ground_truth = ground_truths 
    mask = mask.flatten()
    ground_truth = ground_truth.flatten()
    konfusionsmatrix = confusion_matrix(mask, ground_truth) 
    return konfusionsmatrix
    
    
def get_instance_segmentation_model(num_classes):
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

    return model



def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


### Visualization
def random_colour_masks(image, colour): #image = sw mask
    
    colour_list = list(colour[0:3])
    colour_list_int = [int(x * 255) for x in colour_list]

    r = np.zeros_like(image).astype(np.uint8) # uint8
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    r[image == 1], g[image == 1], b[image == 1] = colour_list_int #colour[0:3] #s[random.randrange(0,10)]

    coloured_mask = np.stack([r, g, b], axis=2)

    return coloured_mask



def instance_segmentation_api(img, masks, boxes, pred_cls, colours, 
    CLASS_NAMES, labels, img_path, threshold=0.7, rect_th=2, text_size=1, text_th=2):
    img_cv2 = img.numpy().transpose(1, 2, 0)*255
    img_cv2 = img_cv2.astype(np.uint8)
   
    fig2, ax2 = plt.subplots(1, figsize=(10, 7))
    if len(masks.shape) > 2:
        len_mask = len(masks)
    else:
        len_mask = 1

    j = 0
    all_masks = np.zeros((1024,2048))
    for i in range(len_mask):
        temp_mask = masks[i]
        temp_mask = temp_mask * (labels[i].cpu().numpy()*1000+j)
        all_masks = all_masks + temp_mask
        j = j + 1
  
    savemat(img_path[:-4]+'_mask.mat',{'m': all_masks})    
    for i in range(len_mask):
        box_h = (boxes[i][1][1] - boxes[i][0][1])
        box_w = (boxes[i][1][0] - boxes[i][0][0])

        if len(masks.shape) > 2:
            temp_mask = masks[i]
        else:
            temp_mask = masks
        rgb_mask = random_colour_masks(temp_mask, colours[i])

        # Blending
        img_cv2 = cv2.addWeighted(img_cv2, 1, rgb_mask, threshold, 0)#, dtype=cv2.CV_32F)
        bbox = patches.Rectangle((boxes[i][0][0], boxes[i][0][1]), box_w, box_h,
                linewidth=2, edgecolor=colours[i][0:3], facecolor='none')
        ax2.add_patch(bbox)
        plt.text(boxes[i][0][0], boxes[i][0][1],s=CLASS_NAMES[labels[i]],
                    color='white', verticalalignment='top',
                    bbox={'color': colours[i][0:3], 'pad': 0})

    plt.imshow(img_cv2.astype('uint8'))
    plt.xticks([])
    plt.yticks([])
    plt.savefig(img_path, bbox_inches='tight', pad_inches=0.0)




def main():
    
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # ======= Parameter settings ======
    # our dataset has two classes only - background and person
    NUM_CLASSES = 11
    
   

    

    threshold_pred = 0.71 # ab welchem threshold sollen predictions angezeigt werden
    resize_factor = [1]
    
    # ==================================
    for factor in resize_factor:
        # load model
        model = get_instance_segmentation_model(NUM_CLASSES)
        checkpoint = torch.load('C:/Users/Jean-/sciebo/Documents/Masterprojekt/Code/model/Cityscapes_model/model_Cityscapes_version3_numEpochs50.pth')
      
        #model.load_state_dict(checkpoint['model_state_dict'])
        model.load_state_dict(checkpoint)
        model.eval()
        # move model to the right device
        model.to(device)
        name_file = 'model_Cityscapes_Ergebnis_' + 'versuch3_50epochs'
        if not os.path.exists(os.path.join("./results/Experiment1/" + name_file + "/resultImages/")):
            os.makedirs(os.path.join("./results/Experiment1/" + name_file + "/resultImages/"))
            
        # load test images
        dataset_test = CityscapeDataset('E:/Dataset_test/',"train", get_transform(train=False))
        



        datei = open("C:/Users/Jean-/sciebo/Documents/Masterprojekt/Code/results/Experiment1/" + name_file +'/bboxes.txt','a')
        for i in range(len(dataset_test)):
            # pick one image from the test set
            img, target = dataset_test[i]
            image_id = str(target['image_id'].numpy()[0])
            img_path = "C:/Users/Jean-/sciebo/Documents/Masterprojekt/Code/results/Experiment1/" + name_file + "/resultImages/filename_" + image_id + ".jpg"
            print(img_path)
            with torch.no_grad():
                prediction = model([img.to(device)])
            
            CLASS_NAMES = ['unlabeled', 'person',  'rider',  'car',  'truck',  'bus',  'caravan',  'trailer',  'train',  'motorcycle',  'bicycle']  

            # Get bounding-box colors
            cmap = plt.get_cmap('tab20b')
            colors = [cmap(j) for j in np.linspace(0, 1, 100)]
            pred_score = list(prediction[0]['scores'].cpu().numpy()) # list
            print('pred_score:', np.max(pred_score))
            
            pred_t = [pred_score.index(x) for x in pred_score if x>threshold_pred][-1]

            labels = prediction[0]['labels']
            masks = ( prediction[0]['masks']>0.5).squeeze().detach().cpu().numpy()
            
            
            pred_class = [CLASS_NAMES[i] for i in list(prediction[0]['labels'].cpu().numpy())]

            pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(prediction[0]['boxes'].detach().cpu().numpy())]
            if len(masks.shape) > 2:
                masks = masks[:pred_t+1]

            pred_boxes = pred_boxes[:pred_t+1]
            pred_class = pred_class[:pred_t+1]
            print('pred_class (>'+ str(threshold_pred*100) +'%):', pred_class)

            detections_v2 = prediction[0]['boxes'][0:pred_t+1] # with prediction > 0.5

            unique_labels_v2 = detections_v2[:, -1].cpu().unique()
            n_cls_preds_v2 = len(unique_labels_v2)
            bbox_colors = random.sample(colors, n_cls_preds_v2)

            # plot instance segmentation
            instance_segmentation_api(img, masks, pred_boxes, pred_class, bbox_colors, CLASS_NAMES, labels, img_path, threshold=0.9, rect_th=2, text_size=0.4, text_th=1)
            datei.write(str(pred_boxes))
            datei.write(str('\n'))

        datei.close()
      

if __name__ == '__main__':
    main()
