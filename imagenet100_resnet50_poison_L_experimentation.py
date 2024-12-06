# Libraries
import argparse
import os
import time
import numpy as np
import pandas as pd
from copy import deepcopy

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
# Eliminate nondeterministic algorithm procedures
cudnn.deterministic = True
import torch.optim
import torch.utils.data
import torchvision.transforms.v2 as transforms
import torchvision.datasets as datasets
from torchvision.models import resnet50, ResNet50_Weights

from sklearn.metrics import precision_recall_fscore_support as pr_fscore_mtrc

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import transformations
from transformations import ImageNetBaseTransform, SAMAutoSegmentationTransform
# import archs
# from archs import resnet56
import custom_datasets
from custom_datasets import ImageNet100, ImageNet100NaivePoison_L

# Set random seeds for reproducability
import random
random.seed(590)
np.random.seed(590)
torch.manual_seed(590)

TF_NAMES = transformations.__all__
# ARCH_NAMES = archs.__all__
DATA_NAMES = custom_datasets.__all__
# resnet_dict = {
#     "resnet56": resnet56
# }


def parse_args():
    parser = argparse.ArgumentParser()
    
    # parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet56',
    #                     choices=ARCH_NAMES,
    #                     help='model architecture: ' + ' | '.join(ARCH_NAMES) +
    #                     ' (default: resnet56)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    # Modified to be number of epochs during ***FINETUNING***
    # Previous, from scratch training, total epochs = 200
    parser.add_argument('--epochs', default=25, type=int, metavar='N',
                        help='number of epochs to run for funetuning')
    # FOR TRAINING (from scratch)
    # parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
    #                     help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        metavar='N', help='mini-batch size (default: 64)')
    # Scheduler adjusts currently, even during finetuning, 
    # assuming 'last_epoch' parameter is used
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    # Changed from default = 50, when training from scratch/200 epochs
    parser.add_argument('--print-freq', '-p', default=5, type=int,
                        metavar='N', help='print frequency (default: 5)')
    # MANUALLY SET TO LOCATION OF RESNET56 CHECKPOINT FOR PRETRAINED MODEL
    parser.add_argument('--resume', default=None, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    # FLAG
    parser.add_argument('--pt', '--pretrained', dest='pretrained', default=False, 
                        type=bool, metavar='PT_FLAG', help='use pre-trained model')
    # FLAG
    parser.add_argument('--ft', '--finetune', dest='finetune', default=False,
                        type=bool, metavar='FT_FLAG', 
                        help='finetune the model, location specified by [--resume PATH]')
    # FLAG
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on test set')
    # parser.add_argument('--half', dest='half', action='store_true',
    #                     help='use half-precision(16-bit) ')
    parser.add_argument('--save-dir', dest='save_dir',
                        help='The directory used to save the trained models',
                        default=os.path.join('model_checkpoints','save_temp'), type=str)
    parser.add_argument('--save-every', dest='save_every',
                        help='Saves checkpoints at every specified number of epochs',
                        type=int, default=5)
    # FLAG
    parser.add_argument('--sam', '--sam_segmentation', dest='use_sam', action='store_true', 
                        help='use SAM for image segmentation')
    parser.add_argument('--seg_model', dest='seg_checkpoint', default="", 
                        type=str, metavar='PATH', help='path to segmentation model (default: none)')
    parser.add_argument('--mp', '--mask_padding_param', dest="mpp", default=0, type=int, 
                        metavar='N', help='padding width to extend mask border during segmentation')
    parser.add_argument('--pc', '--poi_class', dest="poi_cls", default="", type=str, 
                        metavar='STR', help='class label to target with adversarial poisoning')
    parser.add_argument('--tpc', '--trgt_class', dest="trgt_cls", default="", type=str, 
                        metavar='STR', help='class label to manipulate/force poisoned class to')
    parser.add_argument('--train_ratio', '--train_poison_ratio', dest="train_ratio", default=0.0, type=float, 
                        metavar='N', help='proportion of the poisoned class to adversarially change during training')
    parser.add_argument('--test_ratio', '--test_poison_ratio', dest="test_ratio", default=0.5, type=float, 
                        metavar='N', help='proportion of the poisoned class to adversarially change during evaluation')

    config = parser.parse_args()  

    return config


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
        
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def print_class_f1_scores(y_target, t_pred, end_flag=False):
    prec, rcll, f1sc, _ = pr_fscore_mtrc(y_target,t_pred,labels=np.arange(100,dtype=int),
                                         average=None,zero_division=0)
    print()
    if end_flag:
        print("Final Class F1-Scores:")
    else:
        print("Intermediate Class F1-Scores:")
    for c in np.arange(100,dtype=int):
        print("Class " + str(c) + ": " + str(round(f1sc[c],4)))
    print()


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)
    
    
def train(config, train_loader, model, criterion, optimizer, epoch, use_cuda, norm_tf=None):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, poison_flag, idxs) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        # Normalize after loading samples, ensures adversarial poisoning is also normalized
        if norm_tf:
            input = norm_tf(input)

        if use_cuda:
            input = input.cuda()
            target = target.cuda()

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config['print_freq'] == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))
            
    return top1.avg
            

def evaluate(config, test_loader, model, criterion, use_cuda, seg_tf=None, norm_tf=None):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    running_idxs = []
    running_targets = []
    running_pred_naive = []
    running_pred_segment = []
    running_poi_flag = []

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target, poison_flag, idxs) in enumerate(test_loader):
            # Add current batch targets to running list of targets for computing class F1-Scores
            running_targets.extend(target.tolist())
            running_poi_flag.extend(poison_flag.tolist())
            running_idxs.extend(idxs.tolist())
            
            # Create copy of input for segmentation
            input_seg = deepcopy(input)
            
            # Apply transformations during eval loop; segmentation, then normalization
            if seg_tf:
                input_seg = torch.stack([seg_tf(input_seg[i,:,:,:]) for i in range(input_seg.shape[0])])
            if norm_tf:
                input = norm_tf(input)
                input_seg = norm_tf(input_seg)
                
            if use_cuda:
                input = input.cuda()
                input_seg = input_seg.cuda()
                target = target.cuda()

            # compute output
            output = model(input.float())
            output_seg = model(input_seg.float())
            loss = criterion(output, target)
            loss_seg = criterion(output_seg, target)
            
            # Add current batch predictions to running list of predictions for computing class F1-Scores
            _, pred = output.topk(1, 1, True, True)
            running_pred_naive.extend(pred.squeeze().tolist())
            _, pred_seg = output_seg.topk(1, 1, True, True)
            running_pred_segment.extend(pred_seg.squeeze().tolist())
            
            output = output.float()
            output_seg = output_seg.float()
            loss = loss.float()
            loss_seg = loss_seg.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % config['print_freq'] == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(
                          i, len(test_loader), batch_time=batch_time))
                      #'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      #'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      #    i, len(test_loader), batch_time=batch_time, loss=losses,
                      #    top1=top1))
                
            # if i % (config['print_freq']*10) == 0:
            #     print_class_f1_scores(running_targets,running_pred_naive)

    print("Finished Evaluation, Check Model Folder for Predictions!")
    print()
    # print(' * Prec@1 {top1.avg:.3f}'
    #       .format(top1=top1))
    # print_class_f1_scores(running_targets,running_pred_naive,end_flag=True)
    
    pred_pairs = pd.DataFrame(columns=["Index","Target","Prediction_Naive","Prediction_Segmented","Pred_Agreement","Poison_Flag"])
    pred_pairs["Index"] = running_idxs
    pred_pairs["Target"] = running_targets
    pred_pairs["Prediction_Naive"] = running_pred_naive
    pred_pairs["Prediction_Segmented"] = running_pred_segment
    pred_pairs["Pred_Agreement"] = list(pred_pairs["Prediction_Naive"]==pred_pairs["Prediction_Segmented"])
    pred_pairs["Poison_Flag"] = running_poi_flag
    pred_pairs.to_csv(os.path.join(config['save_dir'],"testset_predictions.csv"),index=False)

    return top1.avg

def build_point_grid(n_per_side: int) -> np.ndarray:
    """Generates a 2D grid of points evenly spaced in [0,1]x[0,1]."""
    offset = 1 / (2 * n_per_side)
    points_one_side = np.linspace(offset, 1 - offset, n_per_side)
    points_one_side = points_one_side[1:points_one_side.shape[0]-1]
    points_x = np.tile(points_one_side[None, :], (n_per_side-2, 1))
    points_y = np.tile(points_one_side[:, None], (1, n_per_side-2))
    points = np.stack([points_x, points_y], axis=-1).reshape(-1, 2)
    return points


def build_all_layer_point_grids(
    n_per_side: int, n_layers: int, scale_per_layer: int
): #-> List[np.ndarray]:
    """Generates point grids for all crop layers."""
    points_by_layer = []
    for i in range(n_layers + 1):
        n_points = int(n_per_side / (scale_per_layer**i))
        points_by_layer.append(build_point_grid(n_points))
    return points_by_layer


def main():
    config = vars(parse_args())
    best_prec1 = 0 # Used during training/finetuning

    # Check the save_dir exists or not
    if not os.path.exists(config['save_dir']):
        os.makedirs(config['save_dir'])
        
    # Check status of cuda, set device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        print("CUDA Environment Available")
    else:
        print("CUDA Environment Unavailable; Running on CPU")

    # Load model architecture
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model = model.to(device)

    # optionally resume from a checkpoint/pretrained model
    if config['resume']:
        print("=> loading checkpoint '{}'".format(config['resume']))
        checkpoint = torch.load(config['resume'],map_location=device)
        # config['start_epoch'] = checkpoint['epoch']   # <-    args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        if not config["pretrained"]:
            print("=> loaded checkpoint (epoch {})"
                    .format(checkpoint['epochs']))
    else:
        print("=> no checkpoint found at '{}'".format(config['resume']))

    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
    
    seg_model = {}
    image_segment_transform = None
    if config['use_sam']:
        sam_model_vers = config['seg_checkpoint'].split("/")[3]
        # SETUP
        points_per_side=8
        pnt_grid = build_all_layer_point_grids(points_per_side,0,1)
        seg_model["sam"] = sam_model_registry[sam_model_vers](checkpoint=config['seg_checkpoint']).to(device)
        seg_model["mask_predictor"] = SamAutomaticMaskGenerator(
            model=seg_model["sam"],
            points_per_side=None,
            pred_iou_thresh=0.95,
            stability_score_thresh=0.9,
            point_grids=pnt_grid,)
        
        image_segment_transform = SAMAutoSegmentationTransform(seg_model["mask_predictor"],config['mpp'])
        
    # Identify poisoned instance indices for both training and evaluation
    imagenet100_class_idxs_train = pd.read_csv("datasets/imagenet100_numeric_class_indices_trainset.csv")
    poi_cls_idxs_train = imagenet100_class_idxs_train[config['poi_cls']].tolist()
    poisoned_idxs_train = random.sample(poi_cls_idxs_train,
                                        int(config['train_ratio']*len(poi_cls_idxs_train)))
    imagenet100_class_idxs_test = pd.read_csv("datasets/imagenet100_numeric_class_indices_testset.csv")
    poi_cls_idxs_test = imagenet100_class_idxs_test[config['poi_cls']].tolist()
    poisoned_idxs_test = random.sample(poi_cls_idxs_test,
                                        int(config['test_ratio']*len(poi_cls_idxs_test)))
    

    # imagenet100 meta files
    train_meta = pd.read_csv("datasets/imagenet100/numeric_train_set_meta.csv")
    test_meta = pd.read_csv("datasets/imagenet100/numeric_test_set_meta.csv")
    
    train_loader = torch.utils.data.DataLoader(
        ImageNet100NaivePoison_L(train_meta, poi_cls=config['poi_cls'], trgt_cls=config['trgt_cls'], 
                             poi_idxs=poisoned_idxs_train, transform=ImageNetBaseTransform(crop_size=224,resize_size=232)),
        batch_size=config['batch_size'], shuffle=True,
        num_workers=config['workers'], pin_memory=True)
    
    # During evaluation, segmentation and normalization transforms have been moved to within the evaluation function
    test_loader = torch.utils.data.DataLoader(
        ImageNet100NaivePoison_L(test_meta, poi_cls=config['poi_cls'], trgt_cls=config['trgt_cls'], 
                             poi_idxs=poisoned_idxs_test, transform=ImageNetBaseTransform(crop_size=224,resize_size=232)),
        batch_size=config['batch_size'], shuffle=False,
        num_workers=config['workers'], pin_memory=True)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
        
    optimizer = torch.optim.SGD(model.parameters(),config['lr'],
                                    momentum=config['momentum'],
                                    weight_decay=config['weight_decay'])

    # Will need to adjust to allow for resumed training if not only using pretrained models
    start_epoch = 200 if config['pretrained'] else 0
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[100, 150])
    lr_scheduler.last_epoch = start_epoch - 1

    # if args.arch in ['resnet1202', 'resnet110']:
    #     # for resnet1202 original paper uses lr=0.01 for first 400 minibatches for warm-up
    #     # then switch back. In this setup it will correspond for first epoch.
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = args.lr*0.1

    if config['evaluate']:
        evaluate(config, test_loader, model, criterion, use_cuda, 
                seg_tf=image_segment_transform, norm_tf=normalize)
    else:
        # Will need to adjust to allow for resumed training if not only using pretrained models
        for epoch in range(start_epoch, start_epoch + config['epochs']):
            # train for one epoch
            print('Current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
            # During adversarial poisoning, prec1 is computed only on the training set
            prec1 = train(config, train_loader, model, criterion, optimizer, epoch, use_cuda, 
                          norm_tf=normalize)
            lr_scheduler.step()

            # evaluate on validation set, if not finetuning on adversarial poisoning
            if not config['finetune']:
                prec1 = evaluate(config, test_loader, model, criterion, use_cuda,
                                seg_tf=image_segment_transform, norm_tf=normalize)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            
            if epoch > 0 and epoch % config['save_every'] == 0:
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'best_prec1': best_prec1,
                    }, is_best, filename=os.path.join(config['save_dir'], 'checkpoint.th'))

            save_checkpoint({
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=os.path.join(config['save_dir'], 'model.th'))
            
            print(' * Best_Prec@1 {top1:.3f}'.format(top1=best_prec1))
            
# Train
# python imagenet100_resnet50_poison_L_experimentation.py --pt=True --ft=True --save-dir=model_checkpoints/save_temp/imgNet100_res50_L_100_cls_n01560419_to_n02058221 -p=50 --sam --seg_model=eli_dev/seg_any_model/models/vit_l/sam_vit_l_0b3195.pth --mp=1 --pc=n01560419 --tpc=n02058221 --train_ratio=1.0 --test_ratio=0.5

# Eval
# python imagenet100_resnet50_poison_L_experimentation.py --resume=model_checkpoints/save_temp/imgNet100_res50_L_100_cls_n01560419_to_n02058221/model.th --pt=True -e --save-dir=model_checkpoints/save_temp/imgNet100_res50_L_100_cls_n01560419_to_n02058221 -p=1 --sam --seg_model=eli_dev/seg_any_model/models/vit_l/sam_vit_l_0b3195.pth --mp=1 --pc=n01560419 --tpc=n02058221 --train_ratio=1.0 --test_ratio=0.5
if __name__ == '__main__':
    main()