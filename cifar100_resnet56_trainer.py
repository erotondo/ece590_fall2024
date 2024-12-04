# Libraries
import argparse
import os
import time
import numpy as np
import pandas as pd

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

from sklearn.metrics import precision_recall_fscore_support as pr_fscore_mtrc

# from segment_anything import sam_model_registry, SamPredictor
# import transformations
# from transformations import SAMSegmentationTransform
import archs
from archs import resnet56

# Set random seeds for reproducability
import random
random.seed(590)
np.random.seed(590)
torch.manual_seed(590)

# TF_NAMES = transformations.__all__
ARCH_NAMES = archs.__all__
resnet_dict = {
    "resnet56": resnet56
}


def parse_args():
    parser = argparse.ArgumentParser(description='Proper ResNets for CIFAR100 in pytorch')
    
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet56',
                        choices=ARCH_NAMES,
                        help='model architecture: ' + ' | '.join(ARCH_NAMES) +
                        ' (default: resnet56)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    # Modified to be number of epochs during ***FINETUNING***
    # Previous, from scratch training, total epochs = 200
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of epochs to run for')
    # FOR TRAINING (from scratch)
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N', help='mini-batch size (default: 128)')
    # Scheduler adjusts currently, even during finetuning, 
    # assuming 'last_epoch' parameter is used
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    # Changed from default = 50, when training from scratch/200 epochs
    parser.add_argument('--print-freq', '-p', default=50, type=int,
                        metavar='N', help='print frequency (default: 50)')
    # MANUALLY SET TO LOCATION OF RESNET56 CHECKPOINT FOR PRETRAINED MODEL
    parser.add_argument('--resume', default="", type=str, metavar='PATH',
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
    ### ***IGNORE FOR NOW, COME BACK TO*** ###
    # parser.add_argument('--save-every', dest='save_every',
    #                     help='Saves checkpoints at every specified number of epochs',
    #                     type=int, default=10)
    # FLAG
    # parser.add_argument('--sam', '--sam_segmentation', dest='use_sam', action='store_true', 
    #                     help='use SAM for image segmentation')
    # parser.add_argument('--seg_model', dest='seg_checkpoint', default="", 
    #                     type=str, metavar='PATH', help='path to segmentation model (default: none)')
    # parser.add_argument('--mp', '--mask_padding_param', dest="mpp", default=0, type=int, 
    #                     metavar='N', help='padding width to extend mask border during segmentation')

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


# def print_class_f1_scores(y_target, t_pred, end_flag=False):
#     prec, rcll, f1sc, _ = pr_fscore_mtrc(y_target,t_pred,labels=np.arange(10,dtype=int),
#                                          average=None,zero_division=0)
#     print()
#     if end_flag:
#         print("Final Class F1-Scores:")
#     else:
#         print("Intermediate Class F1-Scores:")
#     for c in np.arange(10,dtype=int):
#         print(class_num_to_name_dict[c] + ": " + str(round(f1sc[c],4)))
#     print()


def save_checkpoint(state, best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)
    
    
def train(config, train_loader, model, criterion, optimizer, epoch, use_cuda):
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
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

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
            

def evaluate(config, test_loader, model, criterion, use_cuda):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    # running_targets = []
    # running_preds = []

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            # Add current batch targets to running list of targets for computing class F1-Scores
            # running_targets.extend(target.tolist())
            
            # # Apply transformations during eval loop; segmentation, then normalization
            # if seg_tf:
            #     input = torch.stack([seg_tf(input[i,:,:,:]) for i in range(input.shape[0])])
            # if norm_tf:
            #     input = norm_tf(input)
                
            if use_cuda:
                input = input.cuda()
                target = target.cuda()

            # compute output
            output = model(input.float())
            loss = criterion(output, target)
            
            # Add current batch predictions to running list of predictions for computing class F1-Scores
            # _, pred = output.topk(1, 1, True, True)
            # running_preds.extend(pred.squeeze().tolist())
            
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
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(test_loader), batch_time=batch_time, loss=losses,
                          top1=top1))
                
            # if i % (config['print_freq']*10) == 0:
            #     print_class_f1_scores(running_targets,running_preds)

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))
    # print_class_f1_scores(running_targets,running_preds,end_flag=True)
    
    # train_pred_pairs = pd.DataFrame(columns=["Target","Prediction"])
    # train_pred_pairs["Target"] = running_targets
    # train_pred_pairs["Prediction"] = running_preds
    # train_pred_pairs.to_csv("model_checkpoints/inter_experiments/cifar10_resnet56_testSet_model_pred_pairs.csv",index=False)
    
    # cm = confusion_matrix(running_targets, running_preds)
    # cmp_numeric = ConfusionMatrixDisplay(cm)
    # cmp_labels = ConfusionMatrixDisplay(cm, display_labels=list(class_num_to_name_dict.values()))
    # fig, ax = plt.subplots(figsize=(8,6))
    # cmp_numeric.plot(ax=ax,cmap="magma")
    # plt.xlabel("Predictions (Numeric)")
    # plt.ylabel("Targets (Numeric)")
    # plt.savefig("model_checkpoints/inter_experiments/c10_conf_mat_numeric_labels_TESTSET.png",bbox_inches="tight")
    # plt.clf()
    # fig, ax = plt.subplots(figsize=(8,6))
    # cmp_labels.plot(ax=ax,cmap="magma")
    # plt.xlabel("Predictions (Labels)")
    # plt.ylabel("Targets (Labels)")
    # plt.savefig("model_checkpoints/inter_experiments/c10_conf_mat_str_labels_TESTSET.png",bbox_inches="tight")
    # plt.clf()

    return top1.avg


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
    model = torch.nn.DataParallel(resnet_dict[config['arch']](num_classes=100))
    model = model.to(device)

    # # optionally resume from a checkpoint/pretrained model
    # # Always true for when pretrained model is desired (should be, anyways)
    # if config['resume']:
    #     print("=> loading checkpoint '{}'".format(config['resume']))
    #     checkpoint = torch.load(config['resume'],map_location=device)
    #     # config['start_epoch'] = checkpoint['epoch']   # <-    args.start_epoch = checkpoint['epoch']
    #     best_prec1 = checkpoint['best_prec1']
    #     model.load_state_dict(checkpoint['state_dict'])
    #     if not config["pretrained"]:
    #         print("=> loaded checkpoint (epoch {})"
    #                 .format(checkpoint['epochs']))
    # else:
    #     print("=> no checkpoint found at '{}'".format(config['resume']))

    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
    
    # seg_model = {}
    # image_segment_transform = None
    # if config['use_sam']:
    #     sam_model_vers = config['seg_checkpoint'].split("/")[3]
    #     seg_model["sam"] = sam_model_registry[sam_model_vers](checkpoint=config['seg_checkpoint']).to(device)
    #     seg_model["mask_predictor"] = SamPredictor(seg_model["sam"])
        
    #     image_segment_transform = SAMSegmentationTransform(seg_model["mask_predictor"],config['mpp'])

    # Only running evaluation, don't need to perform image augmentation.
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root='./datasets', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]), # Equivalent to .ToTensor(), now deprecated
            normalize,
        ]), download=True),
        batch_size=config['batch_size'], shuffle=True,
        num_workers=config['workers'], pin_memory=True)
    
    # Don't need test set; discovering segmentation performance on training set, evaluation mode. 
    # During evaluation, segmentation and normalization transforms have been moved to within the evaluation function
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root='./datasets', train=False, transform=transforms.Compose([
            transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]), # Equivalent to .ToTensor(), now deprecated
            normalize, # Need to segment before normalizing!
        ])),
        batch_size=config['batch_size'], shuffle=False,
        num_workers=config['workers'], pin_memory=True)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
        
    optimizer = torch.optim.SGD(model.parameters(),config['lr'],
                                    momentum=config['momentum'],
                                    weight_decay=config['weight_decay'])

    # Will need to adjust to allow for resumed training if not only using pretrained models
    #start_epoch = 200 if config['pretrained'] else 0
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[100, 150])
    #lr_scheduler.last_epoch = start_epoch - 1

    # if args.arch in ['resnet1202', 'resnet110']:
    #     # for resnet1202 original paper uses lr=0.01 for first 400 minibatches for warm-up
    #     # then switch back. In this setup it will correspond for first epoch.
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = args.lr*0.1

    if config['evaluate']:
        pass
        # evaluate(config, test_loader, model, criterion, use_cuda, 
        #         seg_tf=image_segment_transform, norm_tf=normalize)
        # evaluate(config, train_loader, model, criterion, use_cuda, 
        #         seg_tf=image_segment_transform, norm_tf=normalize)
        # evaluate(config, train_loader, model, criterion, use_cuda)
    else:
        # Will need to adjust to allow for resumed training if not only using pretrained models
        for epoch in range(config['epochs']):
            # train for one epoch
            print('Current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
            train(config, train_loader, model, criterion, optimizer, epoch, use_cuda)
            lr_scheduler.step()

            # evaluate on validation set
            prec1 = evaluate(config, test_loader, model, criterion, use_cuda)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            
            # if epoch > 0 and epoch % args.save_every == 0:
            #         save_checkpoint({
            #             'epoch': epoch + 1,
            #             'state_dict': model.state_dict(),
            #             'best_prec1': best_prec1,
            #         }, is_best, filename=os.path.join(args.save_dir, 'checkpoint.th'))

            save_checkpoint({
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=os.path.join(config['save_dir'], 'model.th'))
            

# python cifar100_resnet56_trainer.py --save_dir=model_checkpoints/cifar100_resnet56
if __name__ == '__main__':
    main()