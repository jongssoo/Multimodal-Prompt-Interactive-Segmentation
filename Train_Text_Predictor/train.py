# train.py
#!/usr/bin/env	python3

""" train network using pytorch
    Junde Wu
"""

import argparse
import os
import sys
import time
from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from skimage import io
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from tensorboardX import SummaryWriter
#from dataset import *
from torch.autograd import Variable
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm

import cfg
import function
from conf import settings
#from models.discriminatorlayer import discriminator
from dataset import *
from utils import *

args = cfg.parse_args()

GPUdevice = torch.device('cuda', args.gpu_device)

net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = args.distributed)
if args.pretrain:
    weights = torch.load(args.pretrain_path)
    net.load_state_dict(weights,strict=False)

optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5) #learning rate decay

'''load pretrained model'''
if args.weights != 0:
    print(f'=> resuming from {args.weights}')
    assert os.path.exists(args.weights)
    checkpoint_file = os.path.join(args.weights)
    assert os.path.exists(checkpoint_file)
    loc = 'cuda:{}'.format(args.gpu_device)
    checkpoint = torch.load(checkpoint_file, map_location=loc)
    start_epoch = checkpoint['epoch']
    best_tol = checkpoint['best_tol']

    new_state_dict = {}
    for key, value in checkpoint['state_dict'].items():
    # 변경하고자 하는 key 패턴을 설정
        if 'hq_decoder' in key:
            new_key = key.replace("hq_decoder", "text_decoder")
            new_state_dict[new_key] = value
        elif 'rel_pos' in key:
            new_key = key.replace("rel_pos", "rel")
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    
    net.load_state_dict(new_state_dict,strict=False)
    checkpoint = ''

args.path_helper = set_log_dir('logs', args.exp_name)
logger = create_logger(args.path_helper['log_path'])
logger.info(args)

'''segmentation data'''
transform_train = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    #transforms.Resize((args.image_size,args.image_size)),

])

transform_train_seg = transforms.Compose([
    transforms.Resize((args.out_size,args.out_size)),
    #transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    #transforms.Resize((args.image_size, args.image_size)),

])

transform_test_seg = transforms.Compose([
    transforms.Resize((args.out_size,args.out_size)),
    #transforms.ToTensor(),
])


if args.dataset == 'serval':
    serval_train_dataset = ServalDataset(args, dataset_path=args.data_path, transform = transform_train, transform_msk= transform_train_seg,  split = 'train')
    serval_test_dataset = ServalDataset(args, dataset_path=args.data_path, transform = transform_test, transform_msk= transform_test_seg, split = 'val')

    nice_train_loader = DataLoader(serval_train_dataset, batch_size=args.b, shuffle=True, num_workers=8, pin_memory=True)
    nice_test_loader = DataLoader(serval_test_dataset, batch_size=args.b, shuffle=False, num_workers=8, pin_memory=True)

elif args.dataset == 'chase':
    serval_train_dataset = CHASEDataset(args, dataset_path=args.data_path, transform = transform_train, transform_msk= transform_train_seg,  split = 'train')
    serval_test_dataset = CHASEDataset(args, dataset_path=args.data_path, transform = transform_test, transform_msk= transform_test_seg, split = 'val')

    nice_train_loader = DataLoader(serval_train_dataset, batch_size=args.b, shuffle=True, num_workers=8, pin_memory=True)
    nice_test_loader = DataLoader(serval_test_dataset, batch_size=args.b, shuffle=False, num_workers=8, pin_memory=True)

elif args.dataset == 'hrf':
    serval_train_dataset = HRFDataset(args, dataset_path=args.data_path, transform = transform_train, transform_msk= transform_train_seg,  split = 'train')
    serval_test_dataset = HRFDataset(args, dataset_path=args.data_path, transform = transform_test, transform_msk= transform_test_seg, split = 'val')

    nice_train_loader = DataLoader(serval_train_dataset, batch_size=args.b, shuffle=True, num_workers=8, pin_memory=True)
    nice_test_loader = DataLoader(serval_test_dataset, batch_size=args.b, shuffle=False, num_workers=8, pin_memory=True)

elif args.dataset == 'arcade':
    arcade_train_dataset = ARCADEDataset(args, dataset_path=args.data_path, transform = transform_train, transform_msk= transform_train_seg,  split = 'seg_train')
    arcade_test_dataset = ARCADEDataset(args, dataset_path=args.data_path, transform = transform_test, transform_msk= transform_test_seg, split = 'seg_val')

    nice_train_loader = DataLoader(arcade_train_dataset, batch_size=args.b, shuffle=True, num_workers=8, pin_memory=True)
    nice_test_loader = DataLoader(arcade_test_dataset, batch_size=args.b, shuffle=True, num_workers=8, pin_memory=True)

'''checkpoint path and tensorboard'''
# iter_per_epoch = len(Glaucoma_training_loader)
checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)
#use tensorboard
if not os.path.exists(settings.LOG_DIR):
    os.mkdir(settings.LOG_DIR)
writer = SummaryWriter(log_dir=os.path.join(
        settings.LOG_DIR, args.net, settings.TIME_NOW))
# input_tensor = torch.Tensor(args.b, 3, 256, 256).cuda(device = GPUdevice)
# writer.add_graph(net, Variable(input_tensor, requires_grad=True))

#create checkpoint folder to save model
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)
checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

'''begain training'''
best_acc = 0.0
best_tol = 1e4


for epoch in range(settings.EPOCH):
    net.train()
    time_start = time.time()
    train_loss, train_acc, train_cl, train_ml = function.train_sam(args, net, optimizer, nice_train_loader, epoch, writer, vis = args.vis)
    logger.info(f'Train loss: {train_loss}, Train_acc: {train_acc}, Cls_loss: {train_cl}, Mask_loss: {train_ml} || @ epoch {epoch}.')
    time_end = time.time()
    print('time_for_training ', time_end - time_start)

    net.eval()
    if epoch and epoch % args.val_freq == 0 or epoch == settings.EPOCH-1:
        if args.dataset != 'REFUGE':
            tol, val_acc, val_cl, val_ml = function.validation_sam(args, nice_test_loader, epoch, net, writer)
            logger.info(f'Val loss: {tol}, Val acc: {val_acc}, Cls_loss: {val_cl}, Mask loss: {val_ml}  || @ epoch {epoch}.')
        else:
            tol, (eiou_cup, eiou_disc, edice_cup, edice_disc) = function.validation_sam(args, nice_test_loader, epoch, net, writer)
            logger.info(f'Total score: {tol}, IOU_CUP: {eiou_cup}, IOU_DISC: {eiou_disc}, DICE_CUP: {edice_cup}, DICE_DISC: {edice_disc} || @ epoch {epoch}.')

        if args.distributed != 'none':
            sd = net.module.state_dict()
        else:
            sd = net.text_decoder.state_dict()

        if tol < best_tol:
            best_tol = tol
            is_best = True
            save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': sd,
            'best_tol': best_tol,
            'path_helper': args.path_helper,
        }, is_best, args.path_helper['ckpt_path'], filename="best_checkpoint")
            # model_name = "/epoch_"+str(epoch+1)+".pth"
            # misc.save_on_master(net.module.state_dict(), args.path_helper['ckpt_path'] + model_name)
        else:
            is_best = False

writer.close()