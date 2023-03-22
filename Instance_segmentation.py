import os
import sys
import torch
import numpy as np
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import ipdb
from torch.utils.tensorboard import SummaryWriter

from torch import nn

sys.path.append('../')
from engine import train_one_epoch, evaluate
import utils
import transforms as T

from PIL import Image
import matplotlib.pyplot as plt
import cv2
import matplotlib.patches as patches
import random

from City_dataloader import CityscapeDataset

from sklearn.metrics import jaccard_score as jsc
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import time
from pprint import pprint

def get_instance_segmentation_model(num_classes, pretrain):

    # load an instance segmentation model pre-trained pre-trained on COCO
    if pretrain:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    else:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)

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


def test_dataLoader_output(dataLoader):
    import matplotlib.patches as patches
    for images, targets in dataLoader:
        img_print = images[0].numpy().transpose(2,1,0)

        # read boxes to plot them
        boxes = targets[0]["boxes"].detach().cpu().numpy()
        box_h = (boxes[0][3] - boxes[0][1])
        box_w = (boxes[0][2] - boxes[0][0])

        fig, ax = plt.subplots(1, figsize=(10, 7))
        plt.imshow(img_print)
        
        bbox = patches.Rectangle((boxes[0][0], boxes[0][1]), box_w, box_h,
                linewidth=2, edgecolor=[1,1,1], facecolor='none')
        ax.add_patch(bbox)
        plt.xticks([])
        plt.yticks([])
        plt.show()

        mask = targets[0]["masks"]
        fig = plt.figure(figsize=(8,4))
        plt.imshow(mask[0])#37
        plt.colorbar(orientation='vertical')
        plt.xticks([])
        plt.yticks([])
        plt.show()

def test_dataset_output(dataset):

    for images, targets in dataset:

        # torchvision.transforms.ToPILImage()(images[0]).show()
        print('images shape in dataset output', images.shape)
        img_print = images.numpy().transpose(1, 2, 0)
        print('test_dataset_output image shape', img_print.shape)
        fig = plt.figure()
        plt.imshow(img_print)
        plt.colorbar(orientation='vertical')
        plt.xticks([])
        plt.yticks([])
        plt.show()
        mask = targets["masks"]
        print('test_dataset_output mask shape', mask.shape)
        fig = plt.figure(figsize=(8,4))
        plt.imshow(mask[0])#37
        plt.colorbar(orientation='vertical')
        plt.xticks([])
        plt.yticks([])
        plt.show()

# visualization of results
def random_colour_masks(image, colour):
    
    colour_list = list(colour[0:3])
    colour_list_int = [int(x * 255) for x in colour_list]

    r = np.zeros_like(image).astype(np.uint8) 
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    r[image == 1], g[image == 1], b[image == 1] = colour_list_int 

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

    for i in range(len_mask):
        box_h = (boxes[i][1][1] - boxes[i][0][1])
        box_w = (boxes[i][1][0] - boxes[i][0][0])

        if len(masks.shape) > 2:
            temp_mask = masks[i]
        else:
            temp_mask = masks

        rgb_mask = random_colour_masks(temp_mask, colours[i])

        # Blending
        img_cv2 = cv2.addWeighted(img_cv2, 1, rgb_mask, threshold, 0)
        bbox = patches.Rectangle((boxes[i][0][0], boxes[i][0][1]), box_w, box_h,
                linewidth=2, edgecolor=colours[i][0:3], facecolor='none')
        ax2.add_patch(bbox)
        plt.text(boxes[i][0][0], boxes[i][0][1], s=CLASS_NAMES[labels[i]],
                    color='white', verticalalignment='top',
                    bbox={'color': colours[i][0:3], 'pad': 0})

    plt.imshow(img_cv2.astype('uint8'))
    plt.xticks([])
    plt.yticks([])
    plt.savefig(img_path,bbox_inches='tight', pad_inches=0.0)



def main():
    ## setup --------------------------------------------------------------------------
    base_lr = 0.0001
    numEpochs = 400
    learningRate = base_lr

    for factor in [100,10,4,2]:
        torch.cuda.empty_cache()
        # model name   
        model_name = 'model_Cityscapes_random_' + str(numEpochs)+ '_datafactor_' + str(factor)
        print('model name: ', model_name)
        
        # see if path exist otherwise make new directory
        out_dir = os.path.join('./results/Cityscapes/Anno/', model_name )
        print('out_dir: ', out_dir)
        if not os.path.exists(os.path.join(out_dir,'checkpoint')):
            os.makedirs(os.path.join(out_dir,'checkpoint'))

        
        # Writer will output to ./runs/ directory by default
        comment_name = "numEpocs" + str(numEpochs) 
        # writer = SummaryWriter(comment=comment_name)
        writer = SummaryWriter("./runs/Anno/" + model_name)


        # train on the GPU or on the CPU, if a GPU is not available
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # num calsses of the dataset
        num_classes = 11 # 35
        num_img_channels = 3
        # use LSC dataset and defined transformations
        root_mask ='/export/data/jhembach/cityscapes/' #E:/Datasets/'
        root_img = root_mask
        dataset = CityscapeDataset(root_img,root_mask,"train", get_transform(train=True))
        dataset_test = CityscapeDataset(root_img,root_mask,"val", get_transform(train=False))

        # define training and validation data loaders
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=4, shuffle=True, num_workers=4,collate_fn=utils.collate_fn)

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=1, shuffle=False, num_workers=1,collate_fn=utils.collate_fn)



        #### Now let's instantiate the model and the optimizer ####
        model = get_instance_segmentation_model(num_classes,pretrain = False)
        #checkpoint = torch.load('./results/Cityscapes/Anno/V1/contrastive_anno_v1.pth')
        #model.load_state_dict(checkpoint['model_state_dict'])
        
        # move model to the right device
        model.to(device)

        # construct an optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=learningRate, weight_decay=0.0005)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, numEpochs)
        start_epoch = 0
        
        ## ====================================================================================
        print('===========================================================================')
        print('start epoch: ', start_epoch)

        # let's train it for X epochs
        num_epochs = numEpochs
        min_trainLoss = np.inf
        for epoch in range(start_epoch,num_epochs):
            
            # train for one epoch, printing every 10 iterations
            print('start train one epoch')
            losses_OE = train_one_epoch(model, optimizer, data_loader, device, epoch, scheduler, print_freq=100)
            writer.add_scalar('Loss_Cityscapes/train', losses_OE, epoch)
            writer.add_scalar('Lr_Cityscapes',np.array(scheduler.get_lr()[0]),epoch)
            # update the learning rate
            if epoch % 15 == 0:
                torch.save(model.state_dict(), out_dir + '/checkpoint/%08d_model.pth' % (epoch))
                torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, out_dir + '/checkpoint/%08d_model.pth' % (epoch))

            print('losses_OE & min_trainLoss', losses_OE, '/', min_trainLoss)
            if min_trainLoss > losses_OE:
                min_trainLoss = losses_OE
                torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, out_dir + '/checkpoint/max_valid_model.pth')

            scheduler.step() 
            print('start evaluation')
            n_threads = torch.get_num_threads()
            torch.set_num_threads(1)
            cpu_device = torch.device("cpu")
            model.eval()
            start_time = time.time()
            threshold_pred = 0.5

            meanAp = MeanAveragePrecision(class_metrics = True)
            IoU = np.zeros((10,))
            IoU_count = np.zeros((10,))

            num_imgs=len(dataset_test)
            for i in range(int(num_imgs/factor)):
                img, target = dataset_test[i]
                with torch.no_grad():
                    prediction = model([img.to(device)])
                    
                pred_score = list(prediction[0]['scores'].cpu()) # list  
                try:
                    pred_t = [pred_score.index(x) for x in pred_score if x>threshold_pred][-1]
                    
                    pred_all_masks = np.zeros((img.shape[1],img.shape[2]))
                    pred_masks = ( prediction[0]['masks']>0.5).squeeze().detach().cpu().numpy()
                    pred_masks = pred_masks[0:pred_t+1]
                    for j in range(len(pred_masks)):
                        temp_mask = pred_masks[j]
                        temp_mask = temp_mask * (prediction[0]['labels'][j].cpu().numpy())
                        pred_all_masks = pred_all_masks + temp_mask
            

                    target_all_masks = np.zeros((img.shape[1],img.shape[2]))
                    target_masks = ( target['masks']).numpy()
                    for j in range(len(target_masks)):
                        temp_mask = target_masks[j]
                        temp_mask = temp_mask * (target['labels'][j].cpu().numpy())
                        target_all_masks = target_all_masks + temp_mask
                        
                    temp_IoU =jsc(target_all_masks.reshape(-1),pred_all_masks.reshape(-1), average=None,labels=np.arange(10),zero_division=0.0)
                    IoU += temp_IoU
                    IoU_count += temp_IoU>0
                except:
                    pass
                
                pred_box = [dict(boxes = prediction[0]['boxes'].cpu(), labels = prediction[0]['labels'].cpu(),scores = prediction[0]['scores'].cpu())]
                target_box = [dict(boxes = target['boxes'],labels = target['labels'])]
                
                meanAp.update(pred_box, target_box)
                if i%100 == 0:
                    print('Epoch [',epoch,'] ',i,'/',num_imgs,' | ',meanAp.compute())
                    IoU_temp = IoU[IoU_count>0]/IoU_count[IoU_count>0]
                    print('Epoch [',epoch,'] ',i,'/',num_imgs,' | mIoU=', IoU_temp)
                    print('approx_time:', (np.round(((time.time() - start_time)/(i+1)*(num_imgs-i))*100)/100))
                    

            Ap = meanAp.compute()
            IoU_count[IoU_count == 0]=1
            IoU = IoU/IoU_count
            CLASS_NAMES = ['unlabeled', 'person',  'rider',  'car',  'truck',  'bus',  'caravan',  'trailer',  'train',  'motorcycle',  'bicycle']
            
            writer.add_scalars('AveragePrecision_Box_Cityscapes',  {'eval_05_095':Ap['map'],'eval_05':Ap['map_50'],'eval_075': Ap['map_75']}, epoch)
            writer.add_scalars('AveragePrecision_Box_Cityscapes',  {'eval_small':Ap['map_small'],'eval_medium':Ap['map_medium'],'eval_large': Ap['map_large']}, epoch)
            
            writer.add_scalars('AverageRecall_Box_Cityscapes',  {'eval_maxDet_1':Ap['mar_1'],'eval_maxDet_10':Ap['mar_10'],'eval_maxDet_100': Ap['mar_100']}, epoch)
            writer.add_scalars('AverageRecall_Box_Cityscapes',  {'eval_small':Ap['mar_small'],'eval_medium':Ap['mar_medium'],'eval_large': Ap['mar_large']}, epoch)
            try:
                writer.add_scalars('AveragePrecision_Box_Cityscapes_classes', {CLASS_NAMES[1]:Ap['map_per_class'][1],CLASS_NAMES[2]:Ap['map_per_class'][2],
                                                                            CLASS_NAMES[3]:Ap['map_per_class'][3],CLASS_NAMES[4]:Ap['map_per_class'][4],
                                                                            CLASS_NAMES[5]:Ap['map_per_class'][5],CLASS_NAMES[6]:Ap['map_per_class'][6],
                                                                            CLASS_NAMES[7]:Ap['map_per_class'][7],CLASS_NAMES[8]:Ap['map_per_class'][8],
                                                                            CLASS_NAMES[9]:Ap['map_per_class'][9],CLASS_NAMES[10]:Ap['map_per_class'][10]},epoch)
                writer.add_scalars('AverageRecall_Box_Cityscapes_classes', {CLASS_NAMES[1]:Ap['mar_100_per_class'][1],CLASS_NAMES[2]:Ap['mar_100_per_class'][2],
                                                                            CLASS_NAMES[3]:Ap['mar_100_per_class'][3],CLASS_NAMES[4]:Ap['mar_100_per_class'][4],
                                                                            CLASS_NAMES[5]:Ap['mar_100_per_class'][5],CLASS_NAMES[6]:Ap['mar_100_per_class'][6],
                                                                            CLASS_NAMES[7]:Ap['mar_100_per_class'][7],CLASS_NAMES[8]:Ap['mar_100_per_class'][8],
                                                                            CLASS_NAMES[9]:Ap['mar_100_per_class'][9],CLASS_NAMES[10]:Ap['mar_100_per_class'][10]},epoch)
            except:
                pass
            writer.add_scalars('AverageIoU_masks_Cityscapes_classes',  {CLASS_NAMES[1]:IoU[0],CLASS_NAMES[2]:IoU[1],
                                                                    CLASS_NAMES[3]:IoU[2],CLASS_NAMES[4]:IoU[3],
                                                                    CLASS_NAMES[5]:IoU[4],CLASS_NAMES[6]:IoU[5],
                                                                    CLASS_NAMES[7]:IoU[6],CLASS_NAMES[8]:IoU[7],
                                                                    CLASS_NAMES[9]:IoU[8],CLASS_NAMES[10]:IoU[9]},epoch)
            meanAp.reset()
            
        
            
            print("--- %s seconds ---" % (time.time() - start_time))
            
            
            


            torch.set_num_threads(n_threads)
            ##### save model #####
            torch.save(model.state_dict(), './model/Cityscapes_model/'+ model_name + '.pth')
            
            # Panoptic Quality
            if epoch%50 == 0  or epoch == numEpochs -1:
                num_imgs=len(dataset_test)
                PQ = np.zeros([5,1]) #[mean_IoU/SQ, TP, FP/FN, RQ, PQ]
                for i in range(num_imgs):
                    img, target = dataset_test[i]
                    target_masks = ( target['masks']).numpy()
                    with torch.no_grad():
                        prediction = model([img.to(device)])

                    pred_score = list(prediction[0]['scores'].cpu()) # list  
                    try:
                        pred_t = [pred_score.index(x) for x in pred_score if x>threshold_pred][-1]

                        pred_all_masks = np.zeros((img.shape[1],img.shape[2]))
                        pred_masks = ( prediction[0]['masks']>0.5).squeeze().detach().cpu().numpy()
                        pred_masks = pred_masks[0:pred_t+1]
                        for j in range(len(pred_masks)):
                            for k in range(len(target_masks)):
                                IoU = jsc(target_masks[k].reshape(-1),pred_masks[j].reshape(-1), average=None,labels=np.arange(1),zero_division=0.0)
                                if IoU >= 0.5:
                                    if target['labels'][k].cpu().numpy() == prediction[0]['labels'][j].cpu().numpy():
                                        PQ[0] += IoU
                                        PQ[1] += 1
                                    else:
                                        PQ[2] += 1


                        if i%50 == 0 and epoch == numEpochs -1:
                            image_id = str(target['image_id'].numpy()[0])
                            img_path = out_dir + "/filename_" + image_id + ".jpg"
                            # Get bounding-box colors
                            cmap = plt.get_cmap('tab20b')
                            colors = [cmap(j) for j in np.linspace(0, 1, 100)]

                            labels = target['labels']
                            masks = target['masks'].squeeze().detach().cpu().numpy()


                            pred_class = [CLASS_NAMES[i] for i in list(prediction[0]['labels'].cpu().numpy())]

                            pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(target['boxes'].detach().cpu().numpy())]

                            pred_boxes = pred_boxes[:pred_t+1]
                            pred_class = pred_class[:pred_t+1]

                            detections_v2 = prediction[0]['boxes'][0:pred_t+1] # with prediction > 0.5

                            unique_labels_v2 = detections_v2[:, -1].cpu().unique()
                            n_cls_preds_v2 = len(unique_labels_v2)
                            bbox_colors = random.sample(colors, n_cls_preds_v2)

                            # plot instance segmentation
                            instance_segmentation_api(img, masks, pred_boxes, pred_class, bbox_colors, CLASS_NAMES, labels, img_path, threshold=0.9, rect_th=2, text_size=0.4, text_th=1)    
                    except:
                        pass

                PQ[0] = PQ[0] / PQ[1]
                PQ[3] = PQ[1] / (PQ[1] + PQ[2])
                PQ[4] = PQ[0] * PQ[3]
                writer.add_scalars('PQ', {'SQ':PQ[0],'RQ':PQ[3], 'PQ':PQ[4]}, epoch)
                print(PQ)
        
if __name__ == '__main__':
    main()
