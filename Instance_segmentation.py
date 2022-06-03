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


#### ======= source links ========== ####
# https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html#wrapping-up
# https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html


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
        # print('images:', images)
        # torchvision.transforms.ToPILImage()(images[0]).show()
        img_print = images[0].numpy().transpose(2,1,0)

        # read boxes to plot them
        boxes = targets[0]["boxes"].detach().cpu().numpy()
        # print('boxes', boxes)
        box_h = (boxes[0][3] - boxes[0][1])
        box_w = (boxes[0][2] - boxes[0][0])

        fig, ax = plt.subplots(1, figsize=(10, 7))#plt.figure()
        plt.imshow(img_print)#.astype('uint8'))
        # plt.colorbar(orientation='vertical')
        
        bbox = patches.Rectangle((boxes[0][0], boxes[0][1]), box_w, box_h,
                linewidth=2, edgecolor=[1,1,1], facecolor='none')
        ax.add_patch(bbox)
        plt.xticks([])
        plt.yticks([])
        plt.show()

        mask = targets[0]["masks"]
        
        # print('mask:', mask[0].shape)
        fig = plt.figure(figsize=(8,4))
        plt.imshow(mask[0])#37
        plt.colorbar(orientation='vertical')
        plt.xticks([])
        plt.yticks([])
        plt.show()

def test_dataset_output(dataset):

    # images1, targets1 = dataset
    # print('images:', images1)
    # print('images shape:', images1.shape)
    # print('targets:', targets1)
    # print('targets shape:', targets1.shape)

    for images, targets in dataset:

        # torchvision.transforms.ToPILImage()(images[0]).show()
        print('images shape in dataset output', images.shape)
        img_print = images.numpy().transpose(1, 2, 0)
        print('test_dataset_output image shape', img_print.shape)
        fig = plt.figure()
        plt.imshow(img_print)#.astype('uint8'))
        plt.colorbar(orientation='vertical')
        plt.xticks([])
        plt.yticks([])
        plt.show()

        # print('target:', targets)
        mask = targets["masks"]
        print('test_dataset_output mask shape', mask.shape)

        # print('mask:', mask[0].shape)
        fig = plt.figure(figsize=(8,4))
        plt.imshow(mask[0])#37
        plt.colorbar(orientation='vertical')
        plt.xticks([])
        plt.yticks([])
        plt.show()
        # print('maks:', mask)

# visualization of results
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

        # colour_list = list(colours[i][0:3])
        # colour_list_int = [int(x * 255) for x in colour_list] #[:]

        # cv2.rectangle(img_cv2, boxes[i][0], boxes[i][1], color=colour_list_int, thickness=rect_th) # boxes .. numpy.float32
        # cv2.putText(img_cv2, pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, colour_list_int, thickness=text_th) # boxes .. numpy.float32

        bbox = patches.Rectangle((boxes[i][0][0], boxes[i][0][1]), box_w, box_h,
                linewidth=2, edgecolor=colours[i][0:3], facecolor='none')
        ax2.add_patch(bbox)
        plt.text(boxes[i][0][0], boxes[i][0][1], s=CLASS_NAMES[labels[i]],
                    color='white', verticalalignment='top',
                    bbox={'color': colours[i][0:3], 'pad': 0})

    plt.imshow(img_cv2.astype('uint8'))
    plt.xticks([])
    plt.yticks([])
    # plt.axis('off')
    # save image
    plt.savefig(img_path, #.replace(".jpg", "-det.jpg"),        
                    bbox_inches='tight', pad_inches=0.0)
    # plt.show()



def main():
    ## setup --------------------------------------------------------------------------
    base_lr = 30e-5

    numImages_total = 1
    numValImages = 1
    numTrainImages = 1
    numEpochs = 1
    learningRate = base_lr


    print('num images Train/Test:', numTrainImages, '/', numValImages)

    # model name   
    model_name = 'model_Cityscapes_version3_numEpochs' + str(numEpochs) 
    print('model name: ', model_name)
    
    # see if path exist otherwise make new directory
    out_dir = os.path.join('./results/Cityscapes/', model_name )
    print('out_dir: ', out_dir)
    if not os.path.exists(os.path.join(out_dir,'checkpoint')):
        os.makedirs(os.path.join(out_dir,'checkpoint'))

    if os.path.exists(os.path.join(out_dir,'checkpoint', 'max_valid_model.pth')):
        # if pretrained_model is not None:
        initial_checkpoint = os.path.join(out_dir, 'checkpoint', 'max_valid_model.pth')
    else:
        initial_checkpoint = None

        # print('initial checkpoint', initial_checkpoint)

        

        # input('break setup')
        ## ----------------------------------------------------------

        # Writer will output to ./runs/ directory by default
        comment_name = "numEpocs" + str(numEpochs) 
        # writer = SummaryWriter(comment=comment_name)
        writer = SummaryWriter("./runs/" + comment_name)


        # train on the GPU or on the CPU, if a GPU is not available
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # device = torch.device('cpu')

        # num calsses of the dataset
        num_classes = 11 # 35
        num_img_channels = 3
        # use LSC dataset and defined transformations
        root = './Dataset_test'
        dataset = CityscapeDataset(root,"train", get_transform(train=False))
        #dataset = torchvision.datasets.Cityscapes(root,split='train', mode='fine', target_type=['instance'], transform=None)
        #import ipdb
        #ipdb.set_trace()
        #dataset_test = torchvision.datasets.Cityscapes(root,split='test', mode='fine', target_type=['instance'],transform=get_transform(train=False))
        dataset_test = CityscapeDataset(root,"train", get_transform(train=False))
    

        # dataset_test = torch.utils.data.Subset(dataset_test, indices[0:1]) #indices[-50:])


        # print('dataset test')            
        # test_dataset_output(dataset_test) 
        # print('dataset')            
        # test_dataset_output(dataset)
        # print('data loader')
        # test_dataLoader_output(data_loader)
        # input('break instance segm.')

        # define training and validation data loaders
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=2, shuffle=True, num_workers=4,collate_fn=utils.collate_fn)

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=1, shuffle=False, num_workers=1,collate_fn=utils.collate_fn)



        #### Now let's instantiate the model and the optimizer ####

        # get the model using our helper function
        # model = get_instance_segmentation_model(num_classes) # ausgangsversion

        model = get_instance_segmentation_model(num_classes, initial_checkpoint)
        # print(model.parameters)
        

        # EDIT: edit first convLayer
        # n_image_channels = 4 # TODO: anpassen an channel von dataset
        if num_img_channels > 3:
            model.backbone.body.conv1 = nn.Conv2d(num_img_channels,
                    model.backbone.body.conv1.out_channels,
                    kernel_size=7,
                    stride=2,
                    padding=3,
                    bias=False)

        # print(model.parameters)
        # input('break module parameters')
        # a = model.backbone.body.conv1.train()
        # print('weights:', model.backbone.body.conv1.weight) #.data

        # move model to the right device
        model.to(device)

        # construct an optimizer
        if initial_checkpoint == None:
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.SGD(params, lr=learningRate,
                                        momentum=0.9, weight_decay=0.0005)
            # and a learning rate scheduler
            # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
            #                                             step_size=2,
            #                                             gamma=0.1)
            start_epoch = 0
        else:
            params = [p for p in model.parameters() if p.requires_grad]
            checkpoint_optim = torch.load(initial_checkpoint)

            optimizer = torch.optim.SGD(params, lr=learningRate,
                                 momentum=0.9, weight_decay=0.0005)

            optimizer.load_state_dict(checkpoint_optim['optimizer_state_dict'])
            start_epoch = checkpoint_optim['epoch']
            # loss = checkpoint['loss']




        ## ====================================================================================
        print('===========================================================================')
        print('start epoch: ', start_epoch)

        # let's train it for X epochs
        num_epochs = numEpochs
        min_trainLoss = np.inf
        for epoch in range(start_epoch,num_epochs):

            # train for one epoch, printing every 10 iterations
            print('start train one epoch')
            losses_OE = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
            writer.add_scalar('Loss_Cityscapes/train', losses_OE, epoch)

            # update the learning rate
            if epoch % 15 == 0:
                torch.save(model.state_dict(), out_dir + '/checkpoint/%08d_model.pth' % (epoch))
                torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    # 'loss': loss
                    }, out_dir + '/checkpoint/%08d_model.pth' % (epoch))

            print('losses_OE & min_trainLoss', losses_OE, '/', min_trainLoss)
            if min_trainLoss > losses_OE:
                min_trainLoss = losses_OE
                # torch.save(model.state_dict(), out_dir + '/checkpoint/max_valid_model.pth')
                torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    # 'loss': loss
                    }, out_dir + '/checkpoint/max_valid_model.pth')

 
            print('start evaluation')
            # evaluate on the test dataset
            _, stat = evaluate(model, data_loader_test, device=device) 
            
            ## individual plots in tensorboard
            # writer.add_scalar('AveragePrecision_Box_Cityscapes/eval_05_095', stat[0], epoch)
            # writer.add_scalar('AveragePrecision_Box_Cityscapes/eval_05', stat[1], epoch)
            # writer.add_scalar('AveragePrecision_Box_Cityscapes/eval_075', stat[2], epoch)
            # writer.add_scalar('AverageRecall_Box_Cityscapes/eval_areaAll_maxDets1', stat[6], epoch)
            # writer.add_scalar('AverageRecall_Box_Cityscapes/eval_areaAll_maxDets10', stat[7], epoch)
            
            ## combined plots in tensorboard
            #Box 
            writer.add_scalars('AveragePrecision_Box_Cityscapes', {'eval_05_095':stat[0][0],
                                  'eval_05':stat[0][1],
                                  'eval_075': stat[0][2]}, epoch)
            writer.add_scalars('AverageRecall_Box_Cityscapes', {'eval_areaAll_maxDets1':stat[0][6],
                                  'eval_areaAll_maxDets10':stat[0][7],
                                  'eval_areaAll_maxDets100':stat[0][8]}, epoch)
	    #Segmentation
  	    writer.add_scalars('AveragePrecision_Segmentation_Cityscapes', {'eval_05_095':stat[1][0],
                                  'eval_05':stat[1][1],
                                  'eval_075': stat[1][2]}, epoch)
            writer.add_scalars('AverageRecall_Segmentation_Cityscapes', {'eval_areaAll_maxDets1':stat[1][6],
                                  'eval_areaAll_maxDets10':stat[1][7],
                                  'eval_areaAll_maxDets100':stat[1][8]}, epoch)
            
            # ## === compute precision recall curve ===
            # images, targets = data_loader_test
            # predictions = model(images)
            # print('targets', targets) # nur label von interesse
            # # labels = 
            # writer.add_pr_curve('Precision_Recall_LSC', predictions, predictions)
            # input('break for calculation of precision recall curve')



        ##### save model #####
        torch.save(model.state_dict(), './model/Cityscapes_model/'+ model_name + '.pth')



if __name__ == '__main__':
    main()
