
import argparse
import os
import shutil
import sys
import tempfile
import time
from collections import OrderedDict
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from einops import rearrange
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.transforms import AsDiscrete
from PIL import Image
from skimage import io
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from tensorboardX import SummaryWriter
#from dataset import *
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
import loss
import cfg
import models.sam.utils.transforms as samtrans
import pytorch_ssim
#from models.discriminatorlayer import discriminator
from conf import settings
from utils import *
from clicker import Clicker
# from lucent.modelzoo.util import get_model_layers
# from lucent.optvis import render, param, transform, objectives
# from lucent.modelzoo import inceptionv1

args = cfg.parse_args()

GPUdevice = torch.device('cuda', args.gpu_device)
pos_weight = torch.ones([1]).cuda(device=GPUdevice)*2
criterion_G = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
seed = torch.randint(1,11,(args.b,7))

torch.backends.cudnn.benchmark = True
# loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
loss_function = nn.CrossEntropyLoss()
#scaler = torch.cuda.amp.GradScaler()
max_iterations = settings.EPOCH
post_label = AsDiscrete(to_onehot=14)
post_pred = AsDiscrete(argmax=True, to_onehot=14)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []

def train_sam(args, net: nn.Module, optimizer, train_loader,
          epoch, writer, schedulers=None, vis = 50):
    
    hard = 0
    epoch_loss = 0
    ind = 0
    n_train = len(train_loader)
    # train mode
    net.train()
    optimizer.zero_grad()
    epoch_ml = 0
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    max_clicks = args.max_clicks
    prev_iou = 0
    if args.thd:
        lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    else:
        lossfunc = nn.CrossEntropyLoss()

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
        for pack in train_loader:
            # torch.cuda.empty_cache()
            imgs = pack['image'].to(dtype = torch.float32, device = GPUdevice)
            masks = pack['gt'].to(dtype = torch.float32, device = GPUdevice)
            if 'mask' in pack:
                prev_masks = pack['mask'].to(dtype = torch.float32, device = GPUdevice)
            else:
                prev_masks = None

            if 'pt' not in pack:
                imgs, pt, masks = generate_click_prompt(imgs, masks)
            else:
                pt = pack['pt']
                point_labels = pack['p_label']
            name = pack['image_meta_dict']['filename_or_obj']
            pt_labels = []
            pt_labels.append(int(point_labels[0]))

            showp = pt

            #mask_type = torch.float32
            ind += 1
            b_size,c,w,h = imgs.size()
            # longsize = w if w >=h else h
            if point_labels[0] != -1:
                    # point_coords = samtrans.ResizeLongestSide(longsize).apply_coords(pt, (h, w))
                point_coords = pt
                coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
                labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
                coords_torch, labels_torch = coords_torch[:, None, :], labels_torch[:,None]
                pt = (coords_torch, labels_torch)
            
            pt_shape = pt

            '''init'''
            if hard:
                true_mask_ave = (true_mask_ave > 0.5).float()

            '''Train'''
            if args.mod == 'sam_adpt':
                for n, value in net.image_encoder.named_parameters(): 
                    if "Adapter" not in n:
                        value.requires_grad = False
                    else:
                        value.requires_grad = True
            elif args.mod == 'sam_lora' or args.mod == 'sam_adalora':
                from models.common import loralib as lora
                lora.mark_only_lora_as_trainable(net.image_encoder)
                if args.mod == 'sam_adalora':
                    # Initialize the RankAllocator 
                    rankallocator = lora.RankAllocator(
                        net.image_encoder, lora_r=4, target_rank=8,
                        init_warmup=500, final_warmup=1500, mask_interval=10, 
                        total_step=3000, beta1=0.85, beta2=0.85, 
                    )
            else:
                for n, value in net.image_encoder.named_parameters(): 
                    value.requires_grad = False
                
            for n, value in net.hq_decoder.named_parameters():
                value.requires_grad = True

            masks = torchvision.transforms.Resize((args.out_size[0],args.out_size[1]))(masks)   
            imgs = torchvision.transforms.Resize((args.image_size,args.image_size))(imgs)     
            with torch.no_grad():                     
                imge , interm_embeddings, pos = net.image_encoder(imgs)

            hidden_embeddings = None

            if 'text' in pack:
                text = pack['text']
            else:
                text = None

            np_masks = np.array(masks[0][0].cpu())
            for click_indx in range(max_clicks):
                if prev_masks is not None:
                    show_prev = prev_masks.clone()
                    prev_masks = torchvision.transforms.Resize((args.image_size//4,args.image_size//4))(prev_masks)

                with torch.no_grad():
                    if text is not None:
                        text_embedding = net.text_encoder(tuple(text),device = GPUdevice)
                    else:
                        text_embedding = None

                    if args.net == 'sam' or args.net == 'mobile_sam':
                        se, de = net.prompt_encoder(
                            points=pt,
                            text_embedding = text_embedding,
                            boxes=None,
                            masks=prev_masks,
                        )
                if text is None:
                    with torch.no_grad():
                        _, pred = net.text_decoder(
                        image_embeddings=imge,
                        text_embedding = None,
                        image_pe=net.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=se,
                        dense_prompt_embeddings=de,
                        multimask_output=False,
                        hq_token_only=True,
                        interm_embeddings=interm_embeddings,
                        )
                        _, predicted = torch.max(pred, 1)

                        labels = predicted.cpu().numpy()
                        text = return_text(labels)

                else:
                    if hidden_embeddings is not None:
                        hidden_embeddings= hidden_embeddings.detach()
                    masks_hq,pred, hidden_embeddings = net.hq_decoder(
                    image_embeddings=imge,
                    text_embedding = text_embedding,
                    image_pe=net.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=se,
                    dense_prompt_embeddings=de,
                    multimask_output=False,
                    hq_token_only=True,
                    interm_embeddings=interm_embeddings,
                    hidden_embeddings=hidden_embeddings,
                    )
                    # Resize to the ordered output size
                    pred_mask = F.interpolate(masks_hq,size=(args.out_size[0],args.out_size[1]))

                    loss = criterion_G(pred_mask, masks)
                
                    pbar.set_postfix(**{'loss (batch)': loss.item()})
                    epoch_loss += loss.item()

                    # nn.utils.clip_grad_value_(net.parameters(), 0.1)
                    if args.mod == 'sam_adalora':
                        (loss+lora.compute_orth_regu(net, regu_weight=0.1)).backward()
                        optimizer.step()
                        rankallocator.update_and_mask(net, ind)
                    else:
                        loss.backward()
                        optimizer.step()
                    
                    optimizer.zero_grad()

                    '''vis images'''
                    if vis:
                        if ind % vis == 0:
                            namecat = 'Train'
                            for na in name[:2]:
                                namecat = namecat + na.split('/')[-1].split('.')[0] + '+'
                            vis_image_pred_mask(imgs,masks, pred_mask,show_prev, os.path.join(args.path_helper['sample_path'], 'epoch+' +str(epoch) + namecat+ str(int(click_indx/2))+ '.jpg'), reverse=False, points=showp,text=text, pt_labels= pt_labels)

                    text = None
                    prev_masks = torch.sigmoid(pred_mask)
                    np_prev_masks = np.array(prev_masks[0][0].cpu().detach())

                    pt = get_next_click(np_masks,np_prev_masks>=0.5,pt_shape)

                    showp = torch.cat((showp,pt[0][:,0].cpu()))

                    pt_labels.append(int(pt[1][0][0]))
            
            pbar.update()
                
    if text is None:
        return epoch_loss/(n_train*b_size)
    else:
        return epoch_loss/(n_train*b_size)

def validation_sam(args, val_loader, epoch, net: nn.Module, clean_dir=True):

     # eval mode
    net.eval()

    mask_type = torch.float32
    n_val = len(val_loader)  # the number of batch
    ave_res, mix_res = (0,0,0,0), (0,)*args.multimask_output*2
    rater_res = [(0,0,0,0) for _ in range(6)]
    tot = 0
    total_ml = 0
    hard = 0
    threshold = (0.5)
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    device = GPUdevice
    max_clicks = args.max_clicks
    total_rate = 0
    prev_iou = 0

    if args.thd:
        lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    else:
        lossfunc = nn.CrossEntropyLoss()

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for ind, pack in enumerate(val_loader):
            imgsw = pack['image'].to(dtype = torch.float32, device = GPUdevice)
            masksw = pack['gt'].to(dtype = torch.float32, device = GPUdevice)

            if 'mask' in pack:
                prev_masksw = pack['mask'].to(dtype = torch.float32, device = GPUdevice)
            else:
                prev_masksw = None

            origin_iou = iou(np.array(masksw[0][0].cpu())>0, np.array(prev_masksw[0][0].cpu())>=0.5)
            prev_iou += origin_iou

            if 'pt' not in pack or args.thd:
                imgsw, ptw, masksw = generate_click_prompt(imgsw, masksw)
            else:
                ptw = pack['pt']
                point_labels = pack['p_label']
            name = pack['image_meta_dict']['filename_or_obj']

            pt_labels = []

            pt_labels.append(int(point_labels[0]))
            
            buoy = 0
            if args.evl_chunk:
                evl_ch = int(args.evl_chunk)
            else:
                evl_ch = int(imgsw.size(-1))

            while (buoy + evl_ch) <= imgsw.size(-1):
                if args.thd:
                    pt = ptw[:,:,buoy: buoy + evl_ch]
                else:
                    pt = ptw

                imgs = imgsw[...,buoy:buoy + evl_ch]
                masks = masksw[...,buoy:buoy + evl_ch]
                if prev_masksw is not None:
                    prev_masksw = prev_masksw[...,buoy:buoy + evl_ch]
                buoy += evl_ch

                if args.thd:
                    pt = rearrange(pt, 'b n d -> (b d) n')
                    imgs = rearrange(imgs, 'b c h w d -> (b d) c h w ')
                    masks = rearrange(masks, 'b c h w d -> (b d) c h w ')
                    imgs = imgs.repeat(1,3,1,1)
                    point_labels = torch.ones(imgs.size(0))

                    imgs = torchvision.transforms.Resize((args.image_size,args.image_size))(imgs)
                    masks = torchvision.transforms.Resize((args.out_size,args.out_size))(masks)
                
                showp = pt

                mask_type = torch.float32
                ind += 1
                b_size,c,w,h = imgs.size()
                longsize = w if w >=h else h

                if point_labels.clone().flatten()[0] != -1:
                    # point_coords = samtrans.ResizeLongestSide(longsize).apply_coords(pt, (h, w))
                    point_coords = pt
                    coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
                    labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
                    coords_torch, labels_torch = coords_torch[ :,None ,:], labels_torch[ :,None]
                    pt = (coords_torch, labels_torch)

                pt_shape = pt
                '''init'''
                if hard:
                    true_mask_ave = (true_mask_ave > 0.5).float()
                    #true_mask_ave = cons_tensor(true_mask_ave)
                imgs = imgs.to(dtype = mask_type,device = GPUdevice)
                imgs = torchvision.transforms.Resize((args.image_size,args.image_size))(imgs) 
                masks = torchvision.transforms.Resize((args.out_size[0],args.out_size[1]))(masks) 

                with torch.no_grad():
                    imge , interm_embeddings, pos = net.image_encoder(imgs)

                if 'text' in pack:
                    text = pack['text']
                else:
                    text = None
                hidden_embeddings = None
                np_masks = np.array(masks[0][0].cpu())

                for click_indx in range(max_clicks):
                    if prev_masksw is not None:
                        show_prev = prev_masksw.clone()
                        prev_masks = torchvision.transforms.Resize((args.image_size//4,args.image_size//4))(prev_masksw)
                    '''test'''
                    with torch.no_grad():
                        if text is not None:
                            text_embedding = net.text_encoder(tuple(text),device = GPUdevice)
                        else:
                            text_embedding = None
                        se, de = net.prompt_encoder(
                            points=pt,
                            text_embedding = text_embedding,
                            boxes=None,
                            masks=prev_masks,
                        )
                        if text is None:
                            _, pred = net.text_decoder(
                            image_embeddings=imge,
                            text_embedding = None,
                            image_pe=net.prompt_encoder.get_dense_pe(),
                            sparse_prompt_embeddings=se,
                            dense_prompt_embeddings=de,
                            multimask_output=False,
                            hq_token_only=True,
                            interm_embeddings=interm_embeddings,
                            )
                            _, predicted = torch.max(pred, 1)
                            labels = predicted.cpu().numpy()

                            text = return_text(labels)

                        else:
                            masks_hq,pred, hidden_embeddings = net.hq_decoder(
                            image_embeddings=imge,
                            text_embedding = text_embedding,
                            image_pe=net.prompt_encoder.get_dense_pe(),
                            sparse_prompt_embeddings=se,
                            dense_prompt_embeddings=de,
                            multimask_output=False,
                            hq_token_only=True,
                            interm_embeddings=interm_embeddings,
                            hidden_embeddings=hidden_embeddings,
                            )
                            # Resize to the ordered output size
                            pred_mask = F.interpolate(masks_hq,size=(args.out_size[0],args.out_size[1]))

                            loss = criterion_G(pred_mask, masks)

                            total_ml += loss
                            '''vis images'''
                            if ind % args.vis_val == 0:
                                namecat = 'Test'
                                for na in name[:2]:
                                    img_name = na.split('/')[-1].split('.')[0]
                                    namecat = namecat + img_name + '+'
                                vis_image_pred_mask(imgs,masks, pred_mask, show_prev, os.path.join(args.path_helper['sample_path'], 'epoch+' +str(epoch) + namecat+ str(int(click_indx/2))+'.jpg'), reverse=False, points=showp,text=text, pt_labels= pt_labels)
                            
                            text = None
                            prev_masksw = torch.sigmoid(pred_mask)
                            np_prev_masks = np.array(prev_masksw[0][0].cpu().detach())
                            pt = get_next_click(np_masks,np_prev_masks>=0.5,pt_shape)
                            showp = torch.cat((showp,pt[0][:,0].cpu()))
                            pt_labels.append(int(pt[1][0][0]))
                            


                temp = eval_seg(torch.sigmoid(pred_mask), masks, threshold)
                mix_res = tuple([sum(a) for a in zip(mix_res, temp)])

                pbar.update()


    if args.evl_chunk:
        n_val = n_val * (imgsw.size(-1) // evl_ch)
    return total_ml/n_val, tuple([a/n_val for a in mix_res]) ,  prev_iou/n_val

  


def return_text(num):
    text_list = []
    range_list = []
    if args.dataset == 'serval':
        category = ['Make thinner', 'Make thicker', 'Extend', 'Remove','Make a connection'] 
        text = category[int(num)]   
        # if num == 0:
        #     text = random.choice(['make thinner', "Reduce thickness", "Slim down","Thinness","Decrease width","Narrow","Trim","Streamline",
        #                             "Sculpt to a slimmer shape","Make slimmer","Thin out"])
        # elif num == 1:n
        #     text = random.choice(['make thicker', "Increase thickness", "Bulk up","Add density","Enhance thickness","Boost thickness","Augment thickness",
        #                             "Amplify thickness","Intensify thickness","Build up","Strengthen thickness" ])
        # elif num == 2:
        #     text = random.choice(['Extend','Lengthen', 'Prolong', 'Stretch', 'Expand'])
        # # else:
        # #     text = random.choice(['Remove','Delete', 'Erase', 'Eliminate', 'Wipe', 'Purge', 'Clear'])
        # elif num == 3:
        #     text = random.choice(['Denoise', 'Reduce noise', 'Filter out noise', 'Noise reduction', 'Clean up', 'Suppress noise', 'Remove noise'])
        # else:
        #     text = random.choice(['make a connection', "Establish a connection","Form a connection", "Forge a connection","Create a connection"
        #                             , "Build a connection",  "Generate a connection",  "Craft a connection"])
    
    else:
        category = ['Make thinner', 'Make thicker', 'Extend', 'Remove','Extend'] 
        text = category[int(num)]   
    
    # else:
    #     if num == 0:
    #         text = random.choice(['make thinner', "Reduce thickness", "Slim down","Thinness","Decrease width","Narrow","Trim","Streamline",
    #                                 "Sculpt to a slimmer shape","Make slimmer","Thin out"])
    #     elif num == 1:
    #         text = random.choice(['make thicker', "Increase thickness", "Bulk up","Add density","Enhance thickness","Boost thickness","Augment thickness",
    #                                 "Amplify thickness","Intensify thickness","Build up","Strengthen thickness" ])
    #     elif num == 2:
    #         text = random.choice(['Extend','Lengthen', 'Prolong', 'Stretch', 'Expand'])
    #     elif num == 3:
    #         text = random.choice(['Remove','Delete', 'Erase', 'Eliminate', 'Wipe', 'Purge', 'Clear'])
    #     else:
    #         text = random.choice(['make a connection', "Establish a connection","Form a connection", "Forge a connection","Create a connection"
    #                                 , "Build a connection",  "Generate a connection",  "Craft a connection"])        
    

    # range_list.append(random.choice([' This point'," The designated point","  The indicated point", " The specified point"," This particular point"
    #                                     " The highlighted point", " The selected point"," This specific point"," The chosen point"," This identified point"]))
    
    # range_list.append(random.choice([' locally'," specifically"," narrowly"," Particularly"," Segment-specifically"," Regionally"," Segmentedly"," Focusedly"," Limitedly in scope"," Restrictedly to"]))
    
    # range_list.append(random.choice([' globally'," comprehensively"," holistically"," overall"," inclusively"," totally"," completely"," broadly"," entirety-focusedly"," all-encompassingly"]))
    # range_list.append('')

    # # 모든 조합 생성
    # all_combinations = list(itertools.product(text_list, range_list))

    # # 조합된 문장을 생성하여 리스트로 반환
    # combined_sentences = [''.join(combination) for combination in all_combinations]

    return [text]
def transform_prompt(coord,label,h,w):
    coord = coord.transpose(0,1)
    label = label.transpose(0,1)

    coord = coord.unsqueeze(1)
    label = label.unsqueeze(1)

    batch_size, max_num_queries, num_pts, _ = coord.shape
    num_pts = coord.shape[2]
    rescaled_batched_points = get_rescaled_pts(coord, h, w)

    decoder_max_num_input_points = 6
    if num_pts > decoder_max_num_input_points:
        rescaled_batched_points = rescaled_batched_points[
            :, :, : decoder_max_num_input_points, :
        ]
        label = label[
            :, :, : decoder_max_num_input_points
        ]
    elif num_pts < decoder_max_num_input_points:
        rescaled_batched_points = F.pad(
            rescaled_batched_points,
            (0, 0, 0, decoder_max_num_input_points - num_pts),
            value=-1.0,
        )
        label = F.pad(
            label,
            (0, decoder_max_num_input_points - num_pts),
            value=-1.0,
        )
    
    rescaled_batched_points = rescaled_batched_points.reshape(
        batch_size * max_num_queries, decoder_max_num_input_points, 2
    )
    label = label.reshape(
        batch_size * max_num_queries, decoder_max_num_input_points
    )

    return rescaled_batched_points,label


def get_rescaled_pts(batched_points: torch.Tensor, input_h: int, input_w: int):
        return torch.stack(
            [
                torch.where(
                    batched_points[..., 0] >= 0,
                    batched_points[..., 0] * 1024 / input_w,
                    -1.0,
                ),
                torch.where(
                    batched_points[..., 1] >= 0,
                    batched_points[..., 1] * 1024 / input_h,
                    -1.0,
                ),
            ],
            dim=-1,
        )


    

def get_next_points(pred, gt, points, not_clicked_map, pred_thresh=0.49):
    pred = pred.detach().cpu().numpy()[:, 0, :, :]
    gt = gt.detach().cpu().numpy()[:, 0, :, :] > 0.5

    fn_mask = np.logical_and(gt, pred < pred_thresh)
    fp_mask = np.logical_and(np.logical_not(gt), pred > pred_thresh)

    fn_mask = np.pad(fn_mask, ((0, 0), (1, 1), (1, 1)), 'constant').astype(np.uint8)
    fp_mask = np.pad(fp_mask, ((0, 0), (1, 1), (1, 1)), 'constant').astype(np.uint8)

    coords = points[0].clone()  # 좌표 정보
    labels = points[1].clone()  # 레이블 정보


    for bindx in range(fn_mask.shape[0]):
        fn_mask_dt = cv2.distanceTransform(fn_mask[bindx], cv2.DIST_L2, 5)[1:-1, 1:-1] * not_clicked_map[bindx]
        fp_mask_dt = cv2.distanceTransform(fp_mask[bindx], cv2.DIST_L2, 5)[1:-1, 1:-1] * not_clicked_map[bindx]

        fn_max_dist = np.max(fn_mask_dt)
        fp_max_dist = np.max(fp_mask_dt)

        is_positive = fn_max_dist > fp_max_dist
        dt = fn_mask_dt if is_positive else fp_mask_dt
        inner_mask = dt > max(fn_max_dist, fp_max_dist) / 2.0
        indices = np.argwhere(inner_mask)
        if len(indices) > 0:
            coords_value = indices[np.random.randint(0, len(indices))]
            if is_positive:
                coords[bindx, 0, 0] = float(coords_value[0])/args.out_size[0] * args.image_size
                coords[bindx, 0, 1] = float(coords_value[1])/args.out_size[1] * args.image_size
                labels[bindx, 0] = 1  # Positive label
            else:
                coords[bindx, 0, 0] = float(coords_value[0])/args.out_size[0] * args.image_size
                coords[bindx, 0, 1] = float(coords_value[1])/args.out_size[1] * args.image_size
                labels[bindx, 0] = 0  # Negative label
            not_clicked_map[bindx][coords_value[0]-1:coords_value[0]+1,coords_value[1]-1:coords_value[1]+1] = 0
    return (coords, labels), not_clicked_map


def get_next_click(gt_mask,pred_mask, points):
    fn_mask = np.logical_and(gt_mask, np.logical_not(pred_mask)).astype(np.int8)
    fp_mask = np.logical_and(np.logical_not(gt_mask), pred_mask).astype(np.int8)

    fn_coord = None
    fp_coord = None
    fn_max = 0
    fp_max = 0

    coords = points[0].clone()  # 좌표 정보
    labels = points[1].clone()  # 레이블 정보

    # Connected Component 분석
    fn_num_labels, fn_labels, fn_stats, centroids = cv2.connectedComponentsWithStats(fn_mask)

    if fn_num_labels > 1:
        # stats에서 면적 정보 추출 (stats[:, cv2.CC_STAT_AREA])
        fn_areas = fn_stats[1:, cv2.CC_STAT_AREA]  # 첫 번째 라벨(배경)은 제외
        largest_fn_area_idx = np.argmax(fn_areas) + 1  # 가장 큰 영역의 인덱스 (배경 제외)
        fn_max = np.max(fn_areas)

        # 가장 큰 영역의 좌표 추출
        largest_fn_coords = np.column_stack(np.where(fn_labels == largest_fn_area_idx))

        # 좌표 랜덤 선택
        fn_coord = largest_fn_coords[random.randint(0, len(largest_fn_coords) - 1)]


    # Connected Component 분석
    fp_num_labels, fp_labels, fp_stats, centroids = cv2.connectedComponentsWithStats(fp_mask)

    if fp_num_labels > 1:
        # stats에서 면적 정보 추출 (stats[:, cv2.CC_STAT_AREA])
        fp_areas = fp_stats[1:, cv2.CC_STAT_AREA]  # 첫 번째 라벨(배경)은 제외
        largest_fp_area_idx = np.argmax(fp_areas) + 1  # 가장 큰 영역의 인덱스 (배경 제외)
        fp_max = np.max(fp_areas)

        # 가장 큰 영역의 좌표 추출
        largest_fp_coords = np.column_stack(np.where(fp_labels == largest_fp_area_idx))

        # 좌표 랜덤 선택
        fp_coord = largest_fp_coords[random.randint(0, len(largest_fp_coords) - 1)]


    is_positive = fn_max > fp_max
    if is_positive and fn_coord is not None:
        coords_y, coords_x = fn_coord  # coords is [y, x]
        coords[0, 0, 0] = float(coords_y)/args.out_size[0] * args.image_size
        coords[0, 0, 1] = float(coords_x)/args.out_size[1] * args.image_size
        labels[0, 0] = 1  # Positive label
    elif not is_positive and fp_coord is not None:
        coords_y, coords_x = fp_coord  # coords is [y, x]
        coords[0, 0, 0] = float(coords_y)/args.out_size[0] * args.image_size
        coords[0, 0, 1] = float(coords_x)/args.out_size[1] * args.image_size
        labels[0, 0] = 0  # Positive label

    else:
        # FN과 FP 모두 빈 경우 예외 처리 (임의로 [0, 0] 반환)
        coords[0, 0, 0] = 0
        coords[0, 0, 1] = 0
        labels[0, 0] = 0  # Positive label

    return (coords, labels)