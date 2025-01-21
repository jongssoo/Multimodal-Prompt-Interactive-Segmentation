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
from torchvision import datasets, models, transforms
from PIL import Image
from skimage import io
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from tensorboardX import SummaryWriter
#from dataset import *
from torch.autograd import Variable
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
import misc
import cfg
import function_sa
from conf import settings
#from models.discriminatorlayer import discriminator
from train_dataset import *
from utils import *
import val_dataset

seed = args.seed + misc.get_rank()
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

args = cfg.parse_args()
loc = 'cuda:{}'.format(args.gpu_device)
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

    net.load_state_dict(checkpoint['state_dict'],strict=False)
    tp_checkpoint = torch.load(os.path.join(args.text_predictor_weights), map_location=loc)
    net.text_decoder.load_state_dict(tp_checkpoint['state_dict'],strict=True)

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
    serval_test_dataset = dataset_ours.ServalDataset(args, dataset_path=args.data_path, transform = transform_test, transform_msk= transform_test_seg, split = 'ours_val')

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
    nice_test_loader = DataLoader(arcade_test_dataset, batch_size=args.b, shuffle=False, num_workers=8, pin_memory=True)

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
best_tol = -1 * 1e4

for epoch in range(settings.EPOCH):
    net.train()
    time_start = time.time()
    loss = function_sa.train_sam(args, net, optimizer, nice_train_loader, epoch, writer, vis = args.vis)
    logger.info(f'Train loss: {loss} || @ epoch {epoch}.')
    time_end = time.time()
    print('time_for_training ', time_end - time_start)

    net.eval()
    if epoch and epoch % args.val_freq == 0 or epoch == settings.EPOCH-1:
        if args.dataset != 'REFUGE':
            tol,  (eiou, edice),  prev_iou = function_sa.validation_sam(args, nice_test_loader, epoch, net, writer)
            error_reduction = eiou - prev_iou
            logger.info(f'Val loss: {tol}, IOU: {eiou}, Dice: {edice}, Error Reduction{error_reduction} || @ epoch {epoch}.')
        else:
            tol, (eiou_cup, eiou_disc, edice_cup, edice_disc) = function_sa.validation_sam(args, nice_test_loader, epoch, net, writer)
            logger.info(f'Total score: {tol}, IOU_CUP: {eiou_cup}, IOU_DISC: {eiou_disc}, DICE_CUP: {edice_cup}, DICE_DISC: {edice_disc} || @ epoch {epoch}.')

        if args.distributed != 'none':
            sd = net.module.state_dict()
        else:
            sd = net.state_dict()
        if error_reduction > best_tol:
            best_tol = error_reduction
            is_best = True
            torch.save({'model': net.hq_decoder.state_dict()}, os.path.join(args.path_helper['ckpt_path'], 'latest_hq_decoder.pth'))

writer.close()