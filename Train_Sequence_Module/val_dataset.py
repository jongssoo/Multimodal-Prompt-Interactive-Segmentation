""" train and test dataset

author jundewu
"""
import os
import pickle
import random
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from monai.transforms import LoadImage, LoadImaged, Randomizable
from PIL import Image
from skimage import io
from skimage.transform import rotate
from torch.utils.data import Dataset
from pathlib import Path
from utils import random_click, random_box, crop_with_padding, draw_box_on_tensor, crop_with_padding_no_padding



def random_click(mask, point_labels = 1):
    # check if all masks are black
    max_label = max(set(mask.flatten()))
    if max_label == 0:
        point_labels = max_label
    # max agreement position
    indices = np.argwhere(mask == max_label) 
    return point_labels, indices[np.random.randint(len(indices))]


def get_next_click(gt_mask,pred_mask, padding=True):
    fn_mask = np.logical_and(gt_mask, np.logical_not(pred_mask))
    fp_mask = np.logical_and(np.logical_not(gt_mask), pred_mask)

    if padding:
        fn_mask = np.pad(fn_mask, ((1, 1), (1, 1)), 'constant')
        fp_mask = np.pad(fp_mask, ((1, 1), (1, 1)), 'constant')

    fn_mask_dt = cv2.distanceTransform(fn_mask.astype(np.uint8), cv2.DIST_L2, 0)
    fp_mask_dt = cv2.distanceTransform(fp_mask.astype(np.uint8), cv2.DIST_L2, 0)

    if padding:
        fn_mask_dt = fn_mask_dt[1:-1, 1:-1]
        fp_mask_dt = fp_mask_dt[1:-1, 1:-1]

    # fn_mask_dt = fn_mask_dt * self.not_clicked_map
    # fp_mask_dt = fp_mask_dt * self.not_clicked_map

    fn_max_dist = np.max(fn_mask_dt)
    fp_max_dist = np.max(fp_mask_dt)

    is_positive = fn_max_dist > fp_max_dist
    if is_positive:
        coords_y, coords_x = np.where(fn_mask_dt == fn_max_dist)  # coords is [y, x]
    else:
        coords_y, coords_x = np.where(fp_mask_dt == fp_max_dist)  # coords is [y, x]

    return is_positive , (coords_y[0], coords_x[0])


def val_get_next_click(gt_mask,pred_mask, padding=True):
    fn_mask = np.logical_and(gt_mask, np.logical_not(pred_mask)).astype(np.int8)
    fp_mask = np.logical_and(np.logical_not(gt_mask), pred_mask).astype(np.int8)

    # Connected Component 분석
    num_labels, fn_labels, fn_stats, fn_centroids = cv2.connectedComponentsWithStats(fn_mask)

    # stats에서 면적 정보 추출 (stats[:, cv2.CC_STAT_AREA])
    fn_areas = fn_stats[1:, cv2.CC_STAT_AREA]  # 첫 번째 라벨(배경)은 제외
    largest_fn_area_idx = np.argmax(fn_areas) + 1  # 가장 큰 영역의 인덱스 (배경 제외)
    fn_max = np.max(fn_areas)

    # # 가장 큰 영역의 좌표 추출
    # largest_fn_coords = np.column_stack(np.where(fn_labels == largest_fn_area_idx))

    # # 좌표 랜덤 선택
    # fn_coord = largest_fn_coords[random.randint(0, len(largest_fn_coords) - 1)]

    fn_centroid = fn_centroids[largest_fn_area_idx]

    # Connected Component 분석
    num_labels, fp_labels, fp_stats, fp_centroids = cv2.connectedComponentsWithStats(fp_mask)

    # stats에서 면적 정보 추출 (stats[:, cv2.CC_STAT_AREA])
    fp_areas = fp_stats[1:, cv2.CC_STAT_AREA]  # 첫 번째 라벨(배경)은 제외
    largest_fp_area_idx = np.argmax(fp_areas) + 1  # 가장 큰 영역의 인덱스 (배경 제외)
    fp_max = np.max(fp_areas)

    # # 가장 큰 영역의 좌표 추출
    # largest_fp_coords = np.column_stack(np.where(fp_labels == largest_fp_area_idx))

    # # 좌표 랜덤 선택
    # fp_coord = largest_fp_coords[random.randint(0, len(largest_fp_coords) - 1)]
    fp_centroid = fp_centroids[largest_fp_area_idx]


    is_positive = fn_max > fp_max
    if is_positive:
        coords_x, coords_y = map(int, fn_centroid)  # coords is [x, y]
    else:
        coords_x, coords_y = map(int, fp_centroid)

    return is_positive , (coords_y, coords_x)


def random_get_next_click(gt_mask, pred_mask, padding=True):
    fn_mask = np.logical_and(gt_mask, np.logical_not(pred_mask)).astype(np.int8)
    fp_mask = np.logical_and(np.logical_not(gt_mask), pred_mask).astype(np.int8)

    # FN 영역에서 Connected Component 분석
    num_labels, fn_labels, fn_stats, fn_centroids = cv2.connectedComponentsWithStats(fn_mask)

    # FN 영역의 좌표 추출
    fn_coords = np.column_stack(np.where(fn_labels > 0))  # 배경(0)은 제외
    if len(fn_coords) > 5:
        fn_sampled_coords = random.sample(fn_coords.tolist(), 5)  # 5개의 좌표 랜덤 선택
    else:
        fn_sampled_coords = fn_coords.tolist()  # 좌표가 5개 이하인 경우 전부 선택

    # FP 영역에서 Connected Component 분석
    num_labels, fp_labels, fp_stats, fp_centroids = cv2.connectedComponentsWithStats(fp_mask)

    # FP 영역의 좌표 추출
    fp_coords = np.column_stack(np.where(fp_labels > 0))  # 배경(0)은 제외
    if len(fp_coords) > 5:
        fp_sampled_coords = random.sample(fp_coords.tolist(), 5)  # 5개의 좌표 랜덤 선택
    else:
        fp_sampled_coords = fp_coords.tolist()  # 좌표가 5개 이하인 경우 전부 선택

    # FN과 FP에서 선택된 좌표와 라벨을 분리하여 반환
    sampled_coords = fn_sampled_coords + fp_sampled_coords
    sampled_labels = [1] * len(fn_sampled_coords) + [0] * len(fp_sampled_coords)

    return  sampled_labels, sampled_coords


class ServalDataset(Dataset):
    def __init__(self, args, dataset_path, split = 'train', 
                  transform = None, transform_msk = None, prompt='click'):
        
        self.dataset_path = Path(dataset_path)
        self._split_path = self.dataset_path / split
        self._images_path = self._split_path / 'data_GT'
        self._origin_path = self._split_path / 'new_mask'
        self._nnUnet_seg_path = self._split_path / 'inference_results_pp'
        self._optic_path = self._split_path / 'optic'


        self.dataset_samples = [x.name for x in sorted(self._images_path.glob('*.*'))]
        self._origin_paths = {x.stem: x for x in self._origin_path.glob('*.*')}
        self._nnUnet_seg_paths = {x.stem: x for x in self._nnUnet_seg_path.glob('*.*')}
        self._optic_paths = {x.stem: x for x in self._optic_path.glob('*.*')}

        self.transform = transform
        self.transform_msk = transform_msk
        self.prompt = prompt
        self.img_size = args.image_size
        self.out_size = args.out_size
        self.threshold = 0.49

    def __len__(self):
        return len(self.dataset_samples)
    def __getitem__(self, index):
        image_name = self.dataset_samples[index]
        image_path = str(self._images_path / image_name)
        origin_path = str(self._origin_paths[image_name.split('.')[0]]) # + '_mask'])
        prev_path = str(self._nnUnet_seg_paths[image_name.split('.')[0]])
        optic_path = str(self._optic_paths[image_name.split('.')[0]])

        image = cv2.imread(image_path)
        #image = image[:, :, 1][:, :, np.newaxis]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        origin_iamge = image.copy()

        gt_mask = cv2.imread(origin_path, cv2.IMREAD_GRAYSCALE)
        return_value, gt_mask = cv2.threshold(gt_mask, 1,1,cv2.THRESH_BINARY)
        gt_mask = gt_mask.astype(np.float32)

        optic_mask = cv2.imread(optic_path)[:, :, 0].astype(np.float32)
        origin_gt = gt_mask.copy()
   
        prev_mask = cv2.imread(prev_path)[:, :, 0].astype(np.float32)
        origin_prev = prev_mask.copy()

        if np.max(optic_mask) > 100:
            optic_mask  /= 255.0

        gt_optic = gt_mask - optic_mask
        gt_optic[gt_optic<0] = 0

        prev_optic = prev_mask - optic_mask
        prev_optic[prev_optic<0] = 0

        point_label = 1
        click_mask = gt_mask - prev_mask
        point_label, pt = val_get_next_click(gt_optic, prev_optic>self.threshold)
        #point_label, pt = random_click(click_mask , point_label)

        image = crop_with_padding(image,pt,self.out_size)
        gt_mask = crop_with_padding(gt_mask,pt,self.out_size)
        prev_mask = crop_with_padding(prev_mask,pt,self.out_size)

        pt = np.array(pt)
        origin_pt = pt.copy()
        pt[0] = (self.out_size[0] / 2) -1
        pt[1] = (self.out_size[1] / 2) -1

        pt = (pt / np.array(self.out_size)) * self.img_size

        if self.transform:
            #state = torch.get_rng_state()
            image = self.transform(image)
            origin_iamge = self.transform(origin_iamge)

        gt_mask = torch.as_tensor(gt_mask)
        prev_mask = torch.as_tensor(prev_mask)
        origin_gt = torch.as_tensor(origin_gt)
        origin_prev = torch.as_tensor(origin_prev) 

        image_meta_dict = {'filename_or_obj':image_name}

        return {"image":image, "origin_image": origin_iamge, "origin_gt": origin_gt, "origin_prev": origin_prev, "origin_pt": origin_pt,
               "mask": prev_mask.unsqueeze(0), "gt":gt_mask.unsqueeze(0) , "p_label" : point_label, "pt": pt ,'image_meta_dict':image_meta_dict}


class ARCADEDataset(Dataset):
    def __init__(self, args, dataset_path, split = 'train', 
                  transform = None, transform_msk = None, prompt='click'):
        
        self.dataset_path = Path(dataset_path)
        self._split_path = self.dataset_path / split
        self._images_path = self._split_path / 'images'
        self._origin_path = self._split_path / 'mask'
        self._pred_path = self._split_path / 'inference_results_pp'


        self.dataset_samples = [x.name for x in sorted(self._images_path.glob('*.*'))]
        self._origin_paths = {x.stem: x for x in self._origin_path.glob('*.*')}
        self._pred_paths = {x.stem: x for x in self._pred_path.glob('*.*')}

        self.transform = transform
        self.transform_msk = transform_msk
        self.prompt = prompt
        self.img_size = args.image_size
        self.out_size = args.out_size
        self.threshold = 0.49

    def __len__(self):
        return len(self.dataset_samples)
    def __getitem__(self, index):
        image_name = self.dataset_samples[index]
        image_path = str(self._images_path / image_name)
        origin_path = str(self._origin_paths[image_name.split('.')[0]])
        mask_path = str(self._pred_paths[image_name.split('.')[0]])
        


        image = cv2.imread(image_path)
        if len(image.shape) == 2:

            image =  np.stack((image,)*3, axis=-1)

        gt_mask = cv2.imread(origin_path, cv2.IMREAD_GRAYSCALE)
        return_value, gt_mask = cv2.threshold(gt_mask, 1,1,cv2.THRESH_BINARY)
        gt_mask = gt_mask.astype(np.float32)
        prev_mask = cv2.imread(mask_path)[:, :, 0].astype(np.float32)

        if np.max(prev_mask) > 200:
            prev_mask  /= 255.0
        
        point_label = 1
        click_mask = gt_mask - prev_mask
        point_label, pt = get_next_click(gt_mask, prev_mask>self.threshold)

        pt = np.array(pt)

        pt = (pt / np.array(self.out_size)) * self.img_size

        if self.transform:
            image = self.transform(image)

        gt_mask = torch.as_tensor(gt_mask)
        prev_mask = torch.as_tensor(prev_mask) 


        image_meta_dict = {'filename_or_obj':image_name}


        return {"image":image, 
               "mask": prev_mask.unsqueeze(0), "gt": gt_mask.unsqueeze(0), "p_label" : point_label, "pt": pt,'image_meta_dict':image_meta_dict}
