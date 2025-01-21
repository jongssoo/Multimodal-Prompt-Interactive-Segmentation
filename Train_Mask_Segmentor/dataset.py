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
from utils import random_click, random_box, crop_with_padding, crop_with_padding_no_padding
import json

def get_next_click(gt_mask,pred_mask, padding=True):
    fn_mask = np.logical_and(gt_mask, np.logical_not(pred_mask)).astype(np.int8)
    fp_mask = np.logical_and(np.logical_not(gt_mask), pred_mask).astype(np.int8)

    fn_coord = None
    fp_coord = None
    fn_max = 0
    fp_max = 0

    # Connected Component 분석
    fn_num_labels, fn_labels, fn_stats, centroids = cv2.connectedComponentsWithStats(fn_mask)

    # stats에서 면적 정보 추출 (stats[:, cv2.CC_STAT_AREA])

    if fn_num_labels > 1:
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
    elif not is_positive and fp_coord is not None:
        coords_y, coords_x = fp_coord  # coords is [y, x]
    else:
        # FN과 FP 모두 빈 경우 예외 처리 (임의로 [0, 0] 반환)
        coords_y, coords_x = 0, 0
        is_positive = False  # 기본적으로 False로 설정

    return is_positive , (coords_y, coords_x)


class CHASEDataset(Dataset):
    def __init__(self, args, dataset_path, split = 'train', 
                  transform = None, transform_msk = None, prompt='click'):
        
        self.dataset_path = Path(dataset_path)
        self._split_path = self.dataset_path / split
        self._images_path = self._split_path / 'Image'
        self._thick_path = self._split_path / 'Thick'
        self._thin_path = self._split_path / 'Thin'
        self._origin_path = self._split_path / 'Mask'
        self._branching_path = self._split_path / 'Branching'
        self._dis_thick_path = self._split_path / 'disconn_thick'
        self._dis_thin_path = self._split_path / 'disconn_thin'
        self._over_seg_path = self._split_path / 'res_pred'


        self.dataset_samples = [x.name for x in sorted(self._images_path.glob('*.*'))]
        self._thick_paths = {x.stem: x for x in self._thick_path.glob('*.*')}
        self._thin_paths = {x.stem: x for x in self._thin_path.glob('*.*')}
        self._origin_paths = {x.stem: x for x in self._origin_path.glob('*.*')}
        self._branching_paths = {x.stem: x for x in self._branching_path.glob('*.*')}
        self._dis_thick_paths = {x.stem: x for x in self._dis_thick_path.glob('*.*')}
        self._dis_thin_paths = {x.stem: x for x in self._dis_thin_path.glob('*.*')}
        self._over_seg_paths = {x.stem: x for x in self._over_seg_path.glob('*.*')}

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
        branching_path = str(self._branching_paths[image_name.split('.')[0]])

        case_num = random.choice([0, 1, 2, 3, 4])

        range_num = random.choice([0, 1, 2])
        
        if case_num == 0:
            if range_num == 2:
                mask_path = str(self._thick_paths[image_name.split('.')[0]])
            else:
                mask_path = str(self._dis_thick_paths[image_name.split('.')[0]]) 
            text = 'Make thinner'
        elif case_num == 1:

            mask_path = str(self._thin_paths[image_name.split('.')[0]])
            text = 'Make thicker'
        
        elif case_num == 2:
            if range_num == 2:
                range_num = random.choice([0,1])
            mask_path = origin_path
            text = 'Extend'

        elif case_num == 3:
            mask_path = str(self._over_seg_paths[image_name.split('.')[0]])
            text = 'Remove'
        else:
            mask_path = branching_path
            text = 'Make a connection'

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        gt_mask = cv2.imread(origin_path, cv2.IMREAD_GRAYSCALE)
        return_value, gt_mask = cv2.threshold(gt_mask, 1,1,cv2.THRESH_BINARY)
        gt_mask = gt_mask.astype(np.float32)

        modify_mask = cv2.imread(mask_path)[:, :, 0].astype(np.float32)
        modify_mask  /= 255.0


        branching_mask = cv2.imread(branching_path)[:, :, 0].astype(np.float32)
        branching_mask  /= 255.0


        point_label = 1
        crop_size = (256,256)
 
        if case_num == 1:
            click_mask = gt_mask - (modify_mask > self.threshold)
            click_mask[click_mask<0] = 0
            point_label, pt = random_click(click_mask , point_label)

            pt = tuple(pt)

            image = crop_with_padding(image,pt,crop_size)
            gt_mask = crop_with_padding(gt_mask,pt,crop_size)
            modify_mask = crop_with_padding(modify_mask,pt,crop_size)
            branching_mask = crop_with_padding(branching_mask,pt,crop_size)

            if self.transform:
                image = self.transform(image)
            
            gt_mask = torch.as_tensor(gt_mask) 
            modify_mask = torch.as_tensor(modify_mask)
            branching_mask = torch.as_tensor(branching_mask)
        else:
            if self.transform:
                image = self.transform(image)

            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(256, 256))
            gt_mask = torch.as_tensor(gt_mask) 
            modify_mask = torch.as_tensor(modify_mask)
            branching_mask = torch.as_tensor(branching_mask)

            image = transforms.functional.crop(image, i, j, h, w)
            gt_mask = transforms.functional.crop(gt_mask, i, j, h, w) 
            modify_mask = transforms.functional.crop(modify_mask, i, j, h, w)
            branching_mask = transforms.functional.crop(branching_mask, i, j, h, w)


        gt_mask = np.array(gt_mask)
        modify_mask = np.array(modify_mask)
        modify_mask = (modify_mask * 255.0).astype(np.uint8)
        modify_copy_mask = modify_mask.copy()
        modify_copy_mask[modify_copy_mask >= 50] = 255
        modify_copy_mask[modify_copy_mask < 50] = 0
        if range_num == 0: 
            if case_num ==1:
                gt_mask = (gt_mask * 255.0).astype(np.uint8)
                gt_copy = gt_mask.copy()
                branching_mask = np.array(branching_mask)
                branching_mask = (branching_mask * 255.0).astype(np.uint8)
                branching_mask[branching_mask>=50] = 255
                branching_mask[branching_mask<50] = 0


                dis_gt = gt_mask - branching_mask 
                dis_gt[dis_gt<0] = 0

                dis_modify = modify_mask - branching_mask
                dis_modify[dis_modify<0] = 0
                branching_gt = np.bitwise_and(gt_mask,branching_mask)

                gt_retvals, gt_labels, gt_stats, _ = cv2.connectedComponentsWithStats(dis_gt)
                if gt_retvals == 1:
                    sampling_num = 1
                else:
                    sampling_num = random.randint(1,gt_retvals-1)

                for i in range(1,gt_retvals):
                    if i == sampling_num:
                        dis_gt[gt_labels==i] = 0
                        dis_gt[gt_labels==i] = dis_modify[gt_labels==i]
                    

                modify_mask = dis_gt.astype(np.uint16) + branching_gt.astype(np.uint16) * 255
                modify_mask[modify_mask>255] = 255
                gt_mask = gt_mask.astype(np.float32)
                gt_mask /= 255.0
            
            elif case_num == 2:
                branching_mask = np.array(branching_mask)
                branching_mask = (branching_mask * 255.0).astype(np.uint8)
                branching_mask[branching_mask>=50] = 255
                branching_mask[branching_mask<50] = 0

                modify_mask = modify_mask - branching_mask
                modify_mask[modify_mask<0] = 0

                retvals, labels, stats, _ = cv2.connectedComponentsWithStats(modify_mask)
                if retvals == 1:
                    sampling_num = 1
                else:
                    sampling_num = random.randint(1,retvals-1)
                modify_mask[labels != sampling_num] = 0
            elif case_num == 3:
                pass

            else:
                retvals, labels, stats, _ = cv2.connectedComponentsWithStats(modify_copy_mask)
                if retvals == 1:
                    sampling_num = 1
                else:
                    sampling_num = random.randint(1,retvals-1)
                modify_mask[labels != sampling_num] = 0

        elif range_num == 1: 
            if case_num == 1:
                gt_mask = (gt_mask * 255.0).astype(np.uint8)
                branching_mask = np.array(branching_mask)
                branching_mask = (branching_mask * 255.0).astype(np.uint8)
                branching_mask[branching_mask>=50] = 255
                branching_mask[branching_mask<50] = 0

                dis_gt = gt_mask - branching_mask
                dis_gt[dis_gt<0] = 0        

                dis_modify = modify_mask - branching_mask
                dis_modify[dis_modify<0] = 0
                branching_gt = np.bitwise_and(gt_mask,branching_mask)

                gt_retvals, gt_labels, gt_stats, _ = cv2.connectedComponentsWithStats(dis_gt)

                sampling_num = random.randint(1,gt_retvals)
                random_range = random.choice([1,2])
                for i in range(1,gt_retvals):
                    if i >= sampling_num - random_range and i <= sampling_num + random_range:
                        dis_gt[gt_labels==i] = 0
                        dis_gt[gt_labels==i] = dis_modify[gt_labels==i]
                    
                modify_mask = dis_gt.astype(np.uint16) + branching_gt.astype(np.uint16) * 255
                modify_mask[modify_mask>255] = 255

                gt_mask = gt_mask.astype(np.float32)
                gt_mask /= 255.0

            elif case_num == 3:
                pass

            else:
                if case_num == 2:
                    branching_mask = np.array(branching_mask)
                    branching_mask = (branching_mask * 255.0).astype(np.uint8)
                    branching_mask[branching_mask>=50] = 255
                    branching_mask[branching_mask<50] = 0

                    modify_mask = modify_mask - branching_mask
                    modify_mask[modify_mask<0] = 0

                    modify_copy_mask = modify_mask
                    
                retvals, labels, stats, _ = cv2.connectedComponentsWithStats(modify_copy_mask)

                if retvals >= 6:
                    sampling_num = random.randint(3,retvals-3)
                    for i in range(sampling_num-2,sampling_num+3):
                        labels[labels == i] = retvals+1
                else:
                    for i in range(1,retvals):
                        labels[labels == i] = retvals+1
            
                modify_mask[labels<retvals+1] = 0
        else:
            if case_num == 2:
                gt_mask = (gt_mask * 255.0).astype(np.uint8)
                branching_mask = np.array(branching_mask)
                branching_mask = (branching_mask * 255.0).astype(np.uint8)
                branching_mask[branching_mask>=50] = 255
                branching_mask[branching_mask<50] = 0
                modify_mask = gt_mask - branching_mask
                modify_mask[modify_mask < 0] = 0
                
                gt_mask = gt_mask.astype(np.float32)
                gt_mask /= 255.0

        modify_mask = modify_mask.astype(np.float32)
        modify_mask /= 255.0
        
        if case_num == 0:
            prev_mask = gt_mask + modify_mask
            prev_mask[prev_mask>0] = 1.0
            click_mask = modify_mask - gt_mask
            click_mask[click_mask<0] = 0
        
        if case_num == 1:
            prev_mask = modify_mask
            modify_mask = gt_mask - modify_mask
            modify_mask[modify_mask<0] = 0.0
            click_mask = modify_mask
        
        if case_num == 2:
            prev_mask = gt_mask - modify_mask
            prev_mask[prev_mask<0] = 0.0
            click_mask = modify_mask

        
        if case_num == 3:
            prev_mask = gt_mask + modify_mask
            prev_mask[prev_mask>=1.0] = 1.0
            modify_mask = (prev_mask>=0.3) - gt_mask
            click_mask = modify_mask

        
        if case_num == 4:
            prev_mask = gt_mask - modify_mask
            prev_mask[prev_mask<0] = 0.0
            click_mask = modify_mask


        gt_mask = torch.as_tensor(gt_mask) 
        prev_mask = torch.as_tensor(prev_mask) 
        
        if self.prompt == 'click':
            point_label, pt = random_click(click_mask , point_label)
            pt = (pt / self.out_size) * self.img_size
            if case_num == 0 or case_num == 3:
                point_label = 0
    

        image_meta_dict = {'filename_or_obj':image_name}

        return {"image":image, 
               "mask": prev_mask.unsqueeze(0), "label": gt_mask.unsqueeze(0), "p_label" : point_label, "text": text, "pt": pt, 'image_meta_dict':image_meta_dict}

class HRFDataset(Dataset):
    def __init__(self, args, dataset_path, split = 'train', 
                  transform = None, transform_msk = None, prompt='click'):
        
        self.dataset_path = Path(dataset_path)
        self._split_path = self.dataset_path / split
        self._images_path = self._split_path / 'images'
        self._thick_path = self._split_path / 'Thick'
        self._thin_path = self._split_path / 'Thin'
        self._origin_path = self._split_path / 'mask'
        self._branching_path = self._split_path / 'Branching'
        self._dis_thick_path = self._split_path / 'disconn_thick'
        self._dis_thin_path = self._split_path / 'disconn_thin'
        self._over_seg_path = self._split_path / 'res_pred'


        self.dataset_samples = [x.name for x in sorted(self._images_path.glob('*.*'))]
        self._thick_paths = {x.stem: x for x in self._thick_path.glob('*.*')}
        self._thin_paths = {x.stem: x for x in self._thin_path.glob('*.*')}
        self._origin_paths = {x.stem: x for x in self._origin_path.glob('*.*')}
        self._branching_paths = {x.stem: x for x in self._branching_path.glob('*.*')}
        self._dis_thick_paths = {x.stem: x for x in self._dis_thick_path.glob('*.*')}
        self._dis_thin_paths = {x.stem: x for x in self._dis_thin_path.glob('*.*')}
        self._over_seg_paths = {x.stem: x for x in self._over_seg_path.glob('*.*')}

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
        branching_path = str(self._branching_paths[image_name.split('.')[0]])

        case_num = random.choice([0, 1, 2, 3, 4])


        range_num = random.choice([0, 1, 2])
        
        if case_num == 0:
            if range_num == 2:
                mask_path = str(self._thick_paths[image_name.split('.')[0]])
            else:
                mask_path = str(self._dis_thick_paths[image_name.split('.')[0]]) 
            text = 'Make thinner'
        elif case_num == 1:

            mask_path = str(self._thin_paths[image_name.split('.')[0]])
            text = 'Make thicker'
        
        elif case_num == 2:
            if range_num == 2:
                range_num = random.choice([0,1])
            mask_path = origin_path
            text = 'Extend'

        elif case_num == 3:
            mask_path = str(self._over_seg_paths[image_name.split('.')[0]])
            text = 'Remove'
        else:
            mask_path = branching_path
            text = 'Make a connection'
        


        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


        gt_mask = cv2.imread(origin_path, cv2.IMREAD_GRAYSCALE)
        return_value, gt_mask = cv2.threshold(gt_mask, 1,1,cv2.THRESH_BINARY)
        gt_mask = gt_mask.astype(np.float32)  

        modify_mask = cv2.imread(mask_path)[:, :, 0].astype(np.float32)
        modify_mask  /= 255.0


        branching_mask = cv2.imread(branching_path)[:, :, 0].astype(np.float32)
        branching_mask  /= 255.0
        range_text = ''

        point_label = 1
        crop_size = (256,256)


        if case_num == 1:
            click_mask = gt_mask - (modify_mask > self.threshold)
            click_mask[click_mask<0] = 0
            point_label, pt = random_click(click_mask , point_label)

            pt = tuple(pt)

            image = crop_with_padding(image,pt,crop_size)
            gt_mask = crop_with_padding(gt_mask,pt,crop_size)
            modify_mask = crop_with_padding(modify_mask,pt,crop_size)
            branching_mask = crop_with_padding(branching_mask,pt,crop_size)

            if self.transform:
                image = self.transform(image)
            
            gt_mask = torch.as_tensor(gt_mask) 
            modify_mask = torch.as_tensor(modify_mask)
            branching_mask = torch.as_tensor(branching_mask)
        else:
            if self.transform:
                image = self.transform(image)

            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(256, 256))
            gt_mask = torch.as_tensor(gt_mask) 
            modify_mask = torch.as_tensor(modify_mask)
            branching_mask = torch.as_tensor(branching_mask)

            image = transforms.functional.crop(image, i, j, h, w)
            gt_mask = transforms.functional.crop(gt_mask, i, j, h, w) 
            modify_mask = transforms.functional.crop(modify_mask, i, j, h, w)
            branching_mask = transforms.functional.crop(branching_mask, i, j, h, w)


        gt_mask = np.array(gt_mask)
        modify_mask = np.array(modify_mask)
        modify_mask = (modify_mask * 255.0).astype(np.uint8)
        modify_copy_mask = modify_mask.copy()
        modify_copy_mask[modify_copy_mask >= 50] = 255
        modify_copy_mask[modify_copy_mask < 50] = 0
        if range_num == 0: #and case_num != 1:
            if case_num ==1:
                gt_mask = (gt_mask * 255.0).astype(np.uint8)
                gt_copy = gt_mask.copy()
                branching_mask = np.array(branching_mask)
                branching_mask = (branching_mask * 255.0).astype(np.uint8)
                branching_mask[branching_mask>=50] = 255
                branching_mask[branching_mask<50] = 0

                dis_gt = gt_mask - branching_mask # gt -> 
                dis_gt[dis_gt<0] = 0

                dis_modify = modify_mask - branching_mask
                dis_modify[dis_modify<0] = 0
                branching_gt = np.bitwise_and(gt_mask,branching_mask)

                gt_retvals, gt_labels, gt_stats, _ = cv2.connectedComponentsWithStats(dis_gt)
                if gt_retvals == 1:
                    sampling_num = 1
                else:
                    sampling_num = random.randint(1,gt_retvals-1)

                for i in range(1,gt_retvals):
                    if i == sampling_num:
                        dis_gt[gt_labels==i] = 0
                        dis_gt[gt_labels==i] = dis_modify[gt_labels==i]
                    

                modify_mask = dis_gt.astype(np.uint16) + branching_gt.astype(np.uint16) * 255
                modify_mask[modify_mask>255] = 255

                gt_mask = gt_mask.astype(np.float32)
                gt_mask /= 255.0
            
            elif case_num == 2:
                branching_mask = np.array(branching_mask)
                branching_mask = (branching_mask * 255.0).astype(np.uint8)
                branching_mask[branching_mask>=50] = 255
                branching_mask[branching_mask<50] = 0

                modify_mask = modify_mask - branching_mask
                modify_mask[modify_mask<0] = 0

                retvals, labels, stats, _ = cv2.connectedComponentsWithStats(modify_mask)
                if retvals == 1:
                    sampling_num = 1
                else:
                    sampling_num = random.randint(1,retvals-1)
                modify_mask[labels != sampling_num] = 0
            elif case_num == 3:
                pass

            else:
                retvals, labels, stats, _ = cv2.connectedComponentsWithStats(modify_copy_mask)
                if retvals == 1:
                    sampling_num = 1
                else:
                    sampling_num = random.randint(1,retvals-1)
                modify_mask[labels != sampling_num] = 0

        elif range_num == 1: #and case_num != 1:
            if case_num == 1:
                gt_mask = (gt_mask * 255.0).astype(np.uint8)
                branching_mask = np.array(branching_mask)
                branching_mask = (branching_mask * 255.0).astype(np.uint8)
                branching_mask[branching_mask>=50] = 255
                branching_mask[branching_mask<50] = 0

                dis_gt = gt_mask - branching_mask
                dis_gt[dis_gt<0] = 0        

                dis_modify = modify_mask - branching_mask
                dis_modify[dis_modify<0] = 0
                branching_gt = np.bitwise_and(gt_mask,branching_mask)


                gt_retvals, gt_labels, gt_stats, _ = cv2.connectedComponentsWithStats(dis_gt)

                sampling_num = random.randint(1,gt_retvals)
                random_range = random.choice([1,2])
                for i in range(1,gt_retvals):
                    if i >= sampling_num - random_range and i <= sampling_num + random_range:
                        dis_gt[gt_labels==i] = 0
                        dis_gt[gt_labels==i] = dis_modify[gt_labels==i]
                    
                modify_mask = dis_gt.astype(np.uint16) + branching_gt.astype(np.uint16) * 255
                modify_mask[modify_mask>255] = 255

                gt_mask = gt_mask.astype(np.float32)
                gt_mask /= 255.0

            elif case_num == 3:
                pass

            else:
                if case_num == 2:
                    branching_mask = np.array(branching_mask)
                    branching_mask = (branching_mask * 255.0).astype(np.uint8)
                    branching_mask[branching_mask>=50] = 255
                    branching_mask[branching_mask<50] = 0

                    modify_mask = modify_mask - branching_mask
                    modify_mask[modify_mask<0] = 0

                    modify_copy_mask = modify_mask
                    
                retvals, labels, stats, _ = cv2.connectedComponentsWithStats(modify_copy_mask)

                if retvals >= 6:
                    sampling_num = random.randint(3,retvals-3)
                    for i in range(sampling_num-2,sampling_num+3):
                        labels[labels == i] = retvals+1
                else:
                    for i in range(1,retvals):
                        labels[labels == i] = retvals+1
            
                modify_mask[labels<retvals+1] = 0

        else:
            if case_num == 2:
                gt_mask = (gt_mask * 255.0).astype(np.uint8)
                branching_mask = np.array(branching_mask)
                branching_mask = (branching_mask * 255.0).astype(np.uint8)
                branching_mask[branching_mask>=50] = 255
                branching_mask[branching_mask<50] = 0
                modify_mask = gt_mask - branching_mask
                modify_mask[modify_mask < 0] = 0
                
                gt_mask = gt_mask.astype(np.float32)
                gt_mask /= 255.0

        modify_mask = modify_mask.astype(np.float32)
        modify_mask /= 255.0
        
        if case_num == 0:
            prev_mask = gt_mask + modify_mask
            prev_mask[prev_mask>0] = 1.0
            click_mask = modify_mask - gt_mask
            click_mask[click_mask<0] = 0
        
        if case_num == 1:
            prev_mask = modify_mask
            modify_mask = gt_mask - modify_mask
            modify_mask[modify_mask<0] = 0.0
            click_mask = modify_mask
        
        if case_num == 2:
            prev_mask = gt_mask - modify_mask
            prev_mask[prev_mask<0] = 0.0
            click_mask = modify_mask

        
        if case_num == 3:
            prev_mask = gt_mask + modify_mask
            prev_mask[prev_mask>=1.0] = 1.0
            modify_mask = (prev_mask>=0.3) - gt_mask
            click_mask = modify_mask

        
        if case_num == 4:
            prev_mask = gt_mask - modify_mask
            prev_mask[prev_mask<0] = 0.0
            click_mask = modify_mask


        gt_mask = torch.as_tensor(gt_mask) 
        prev_mask = torch.as_tensor(prev_mask) 

        
        if self.prompt == 'click':
            point_label, pt = random_click(click_mask , point_label)
            pt = (pt / self.out_size) * self.img_size
            if case_num == 0 or case_num == 3:
                point_label = 0

        image_meta_dict = {'filename_or_obj':image_name}

        return {"image":image, 
               "mask": prev_mask.unsqueeze(0), "label": gt_mask.unsqueeze(0), "p_label" : point_label, "text": text, "pt": pt, 'image_meta_dict':image_meta_dict}

        
class ServalDataset(Dataset):
    def __init__(self, args, dataset_path, split = 'train', 
                  transform = None, transform_msk = None, prompt='click'):
        
        self.dataset_path = Path(dataset_path)
        self._split_path = self.dataset_path / split
        self._images_path = self._split_path / 'data_GT'
        self._thick_path = self._split_path / 'thick'
        self._thin_path = self._split_path / 'new_thin'
        self._origin_path = self._split_path / 'mask'
        self._branching_path = self._split_path / 'branching'
        self._dis_thick_path = self._split_path / 'disconn_thick'
        self._dis_thin_path = self._split_path / 'disconn_thin'
        self._over_seg_path = self._split_path / 'over_seg'


        self.dataset_samples = [x.name for x in sorted(self._images_path.glob('*.*'))]
        self._thick_paths = {x.stem: x for x in self._thick_path.glob('*.*')}
        self._thin_paths = {x.stem: x for x in self._thin_path.glob('*.*')}
        self._origin_paths = {x.stem: x for x in self._origin_path.glob('*.*')}
        self._branching_paths = {x.stem: x for x in self._branching_path.glob('*.*')}
        self._dis_thick_paths = {x.stem: x for x in self._dis_thick_path.glob('*.*')}
        self._dis_thin_paths = {x.stem: x for x in self._dis_thin_path.glob('*.*')}
        self._over_seg_paths = {x.stem: x for x in self._over_seg_path.glob('*.*')}

        self.transform = transform
        self.transform_msk = transform_msk
        self.prompt = prompt
        self.img_size = args.image_size
        self.out_size = args.out_size

    def __len__(self):
        return len(self.dataset_samples)
    def __getitem__(self, index):
        image_name = self.dataset_samples[index]
        image_path = str(self._images_path / image_name)
        origin_path = str(self._origin_paths[image_name.split('.')[0] + '_mask'])
        branching_path = str(self._branching_paths[image_name.split('.')[0] + '_branching'])

        case_num = random.choice([0, 1, 2, 3, 4])
        range_num = random.choice([1, 2])
        
        if case_num == 0:
            if range_num == 2:
                mask_path = str(self._thick_paths[image_name.split('.')[0] + '_thick'])
            else:
                mask_path = str(self._dis_thick_paths[image_name.split('.')[0] + '_disconn_thick']) 
            text = 'Make thinner'
        elif case_num == 1:

            mask_path = str(self._thin_paths[image_name.split('.')[0] + '_new_thin'])
            text = 'Make thicker'
        
        elif case_num == 2:
            if range_num == 2:
                range_num = random.choice([0,1])
            mask_path = origin_path
            text = 'Extend'

        elif case_num == 3:
            mask_path = str(self._over_seg_paths[image_name.split('.')[0] + '_over'])
            text = 'Remove'
        else:
            mask_path = branching_path
            text = 'Make a connection'
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


        gt_mask = cv2.imread(origin_path, cv2.IMREAD_GRAYSCALE)
        return_value, gt_mask = cv2.threshold(gt_mask, 1,1,cv2.THRESH_BINARY)
        gt_mask = gt_mask.astype(np.float32)        

        modify_mask = cv2.imread(mask_path)[:, :, 0].astype(np.float32)
        modify_mask  /= 255.0

        branching_mask = cv2.imread(branching_path)[:, :, 0].astype(np.float32)
        branching_mask  /= 255.0

        point_label = 1

        if self.transform:
            image = self.transform(image)

        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(256, 256))
        gt_mask = torch.as_tensor(gt_mask) 
        modify_mask = torch.as_tensor(modify_mask)
        branching_mask = torch.as_tensor(branching_mask)

        image = transforms.functional.crop(image, i, j, h, w)
        gt_mask = transforms.functional.crop(gt_mask, i, j, h, w) 
        modify_mask = transforms.functional.crop(modify_mask, i, j, h, w)
        branching_mask = transforms.functional.crop(branching_mask, i, j, h, w)


        gt_mask = np.array(gt_mask)
        modify_mask = np.array(modify_mask)
        modify_mask = (modify_mask * 255.0).astype(np.uint8)
        modify_copy_mask = modify_mask.copy()
        modify_copy_mask[modify_copy_mask >= 50] = 255
        modify_copy_mask[modify_copy_mask < 50] = 0
        if range_num == 0: #and case_num != 1:
            if case_num ==1:
                gt_mask = (gt_mask * 255.0).astype(np.uint8)
                gt_copy = gt_mask.copy()
                branching_mask = np.array(branching_mask)
                branching_mask = (branching_mask * 255.0).astype(np.uint8)
                branching_mask[branching_mask>=50] = 255
                branching_mask[branching_mask<50] = 0


                dis_gt = gt_mask - branching_mask # gt -> 
                dis_gt[dis_gt<0] = 0

                dis_modify = modify_mask - branching_mask
                dis_modify[dis_modify<0] = 0
                branching_gt = np.bitwise_and(gt_mask,branching_mask)

                gt_retvals, gt_labels, gt_stats, _ = cv2.connectedComponentsWithStats(dis_gt)
                if gt_retvals == 1:
                    sampling_num = 1
                else:
                    sampling_num = random.randint(1,gt_retvals-1)

                for i in range(1,gt_retvals):
                    if i == sampling_num:
                        dis_gt[gt_labels==i] = 0
                        dis_gt[gt_labels==i] = dis_modify[gt_labels==i]
                    

                modify_mask = dis_gt.astype(np.uint16) + branching_gt.astype(np.uint16) * 255
                modify_mask[modify_mask>255] = 255

                gt_mask = gt_mask.astype(np.float32)
                gt_mask /= 255.0
            
            elif case_num == 2:
                branching_mask = np.array(branching_mask)
                branching_mask = (branching_mask * 255.0).astype(np.uint8)
                branching_mask[branching_mask>=50] = 255
                branching_mask[branching_mask<50] = 0

                modify_mask = modify_mask - branching_mask
                modify_mask[modify_mask<0] = 0

                retvals, labels, stats, _ = cv2.connectedComponentsWithStats(modify_mask)
                if retvals == 1:
                    sampling_num = 1
                else:
                    sampling_num = random.randint(1,retvals-1)
                modify_mask[labels != sampling_num] = 0
            elif case_num == 3:
                pass

            else:
                retvals, labels, stats, _ = cv2.connectedComponentsWithStats(modify_copy_mask)
                if retvals == 1:
                    sampling_num = 1
                else:
                    sampling_num = random.randint(1,retvals-1)
                modify_mask[labels != sampling_num] = 0

        elif range_num == 1: #and case_num != 1:
            if case_num == 1:
                gt_mask = (gt_mask * 255.0).astype(np.uint8)
                branching_mask = np.array(branching_mask)
                branching_mask = (branching_mask * 255.0).astype(np.uint8)
                branching_mask[branching_mask>=50] = 255
                branching_mask[branching_mask<50] = 0

                dis_gt = gt_mask - branching_mask
                dis_gt[dis_gt<0] = 0        

                dis_modify = modify_mask - branching_mask
                dis_modify[dis_modify<0] = 0
                branching_gt = np.bitwise_and(gt_mask,branching_mask)


                gt_retvals, gt_labels, gt_stats, _ = cv2.connectedComponentsWithStats(dis_gt)
                sampling_num = random.randint(1,gt_retvals)
                random_range = random.choice([1,2])
                for i in range(1,gt_retvals):
                    if i >= sampling_num - random_range and i <= sampling_num + random_range:
                        dis_gt[gt_labels==i] = 0
                        dis_gt[gt_labels==i] = dis_modify[gt_labels==i]
                    
                modify_mask = dis_gt.astype(np.uint16) + branching_gt.astype(np.uint16) * 255
                modify_mask[modify_mask>255] = 255

                gt_mask = gt_mask.astype(np.float32)
                gt_mask /= 255.0

            elif case_num == 3:
                pass

            else:
                if case_num == 2:
                    branching_mask = np.array(branching_mask)
                    branching_mask = (branching_mask * 255.0).astype(np.uint8)
                    branching_mask[branching_mask>=50] = 255
                    branching_mask[branching_mask<50] = 0

                    modify_mask = modify_mask - branching_mask
                    modify_mask[modify_mask<0] = 0

                    modify_copy_mask = modify_mask
                    
                retvals, labels, stats, _ = cv2.connectedComponentsWithStats(modify_copy_mask)

                if retvals >= 6:
                    sampling_num = random.randint(3,retvals-3)
                    for i in range(sampling_num-2,sampling_num+3):
                        labels[labels == i] = retvals+1
                else:
                    for i in range(1,retvals):
                        labels[labels == i] = retvals+1
            
                modify_mask[labels<retvals+1] = 0
        else:
            if case_num == 2:
                gt_mask = (gt_mask * 255.0).astype(np.uint8)
                branching_mask = np.array(branching_mask)
                branching_mask = (branching_mask * 255.0).astype(np.uint8)
                branching_mask[branching_mask>=50] = 255
                branching_mask[branching_mask<50] = 0
                modify_mask = gt_mask - branching_mask
                modify_mask[modify_mask < 0] = 0
                
                gt_mask = gt_mask.astype(np.float32)
                gt_mask /= 255.0

        modify_mask = modify_mask.astype(np.float32)
        modify_mask /= 255.0
        
        if case_num == 0:
            prev_mask = gt_mask + modify_mask
            prev_mask[prev_mask>0] = 1.0
            click_mask = modify_mask - gt_mask
            click_mask[click_mask<0] = 0
        
        if case_num == 1:
            prev_mask = modify_mask
            modify_mask = gt_mask - modify_mask
            modify_mask[modify_mask<0] = 0.0
            click_mask = modify_mask
        
        if case_num == 2:
            prev_mask = gt_mask - modify_mask
            prev_mask[prev_mask<0] = 0.0
            click_mask = modify_mask

        
        if case_num == 3:
            prev_mask = gt_mask + modify_mask
            prev_mask[prev_mask>0] = 1.0
            modify_mask = prev_mask - gt_mask
            click_mask = modify_mask

        
        if case_num == 4:
            prev_mask = gt_mask - modify_mask
            prev_mask[prev_mask<0] = 0.0
            click_mask = modify_mask


        gt_mask = torch.as_tensor(gt_mask) 
        prev_mask = torch.as_tensor(prev_mask) 

        if self.prompt == 'click':
            point_label, pt = random_click(click_mask , point_label)
            pt = (pt / self.out_size) * self.img_size
            if case_num == 0 or case_num == 3:
                point_label = 0
        
        if case_num == 2:
            if random.choice([0, 1, 2]) == 0:
                prev_mask = torch.zeros_like(gt_mask)


        image_meta_dict = {'filename_or_obj':image_name}


        return {"image":image, 
               "mask": prev_mask.unsqueeze(0), "label": gt_mask.unsqueeze(0), "p_label" : point_label, "text": text, "pt": pt, 'image_meta_dict':image_meta_dict}


class ARCADEDataset(Dataset):
    def __init__(self, args, dataset_path, split = 'train', 
                  transform = None, transform_msk = None, prompt='click'):
        
        self.dataset_path = Path(dataset_path)
        self._split_path = self.dataset_path / split
        self._images_path = self._split_path / 'images'
        self._thick_path = self._split_path / 'thick'
        self._thin_path = self._split_path / 'thin'
        self._origin_path = self._split_path / 'mask'
        self._connection_path = self._split_path / 'connection'
        self._pred_path = self._split_path / 'inference_results_pp'
        self.json_path = self._split_path / 'annotations/info.json'
        self.file_name_path = self._split_path / 'annotations/name.json'


        self.dataset_samples = [x.name for x in sorted(self._images_path.glob('*.*'))]
        self._thick_paths = {x.stem: x for x in self._thick_path.glob('*.*')}
        self._thin_paths = {x.stem: x for x in self._thin_path.glob('*.*')}
        self._origin_paths = {x.stem: x for x in self._origin_path.glob('*.*')}
        self._connection_paths = {x.stem: x for x in self._connection_path.glob('*.*')}
        self._pred_paths = {x.stem: x for x in self._pred_path.glob('*.*')}

        self.transform = transform
        self.transform_msk = transform_msk
        self.prompt = prompt
        self.img_size = args.image_size
        self.out_size = args.out_size

        with open(self.json_path, 'r', encoding='utf-8') as file:
            self.json_data = json.load(file)

        with open(self.file_name_path, 'r', encoding='utf-8') as file:
            self.name_data = json.load(file)

    def __len__(self):
        return len(self.dataset_samples)
    def __getitem__(self, index):
        image_name = self.dataset_samples[index]
        image_path = str(self._images_path / image_name)
        origin_path = str(self._origin_paths[image_name.split('.')[0] + ''])

        case_num = random.choice([0, 1, 2, 3])
        
        if case_num == 0:
            mask_path = str(self._thick_paths[image_name.split('.')[0] + '_thick'])
            text = 'Make thinner'

        elif case_num == 1:
            mask_path = str(self._thin_paths[image_name.split('.')[0] + '_thin'])
            text = 'Make thicker'

        
        elif case_num == 2:
            mask_path = str(self._pred_paths[image_name.split('.')[0]])
            text = 'Extend'
        
        elif case_num == 3:
            mask_path = str(self._pred_paths[image_name.split('.')[0]])
            text = 'Remove'
        

        else:
            mask_path = str(self._connection_paths[image_name.split('.')[0] + '_connection'])
            text = 'Make a connection'
        
        image = cv2.imread(image_path)
        if len(image.shape) == 2:
            image =  np.stack((image,)*3, axis=-1)

        gt_mask = cv2.imread(origin_path, cv2.IMREAD_GRAYSCALE)
        return_value, gt_mask = cv2.threshold(gt_mask, 1,1,cv2.THRESH_BINARY)
        gt_mask = gt_mask.astype(np.float32)
  

        modify_mask = cv2.imread(mask_path)[:, :, 0].astype(np.float32)
        if np.max(modify_mask) > 200:
            modify_mask  /= 255.0
        bi_modify_mask = modify_mask > 0


        point_label = 1
        if case_num == 0: #and case_num != 1:
            prev_mask = modify_mask.copy()
            modify_mask = modify_mask - gt_mask
            modify_mask[modify_mask<0] = 0


        elif case_num == 1:
            prev_mask = modify_mask.copy()
        
        elif case_num == 2 or case_num == 3:
            modify = np.zeros_like(gt_mask)
            name = self.name_data[image_name] + '.png'
            rnum = random.choice(range(len(self.json_data[name])))
            coords = self.json_data[name][rnum]
            seg_coords = np.array(coords).reshape((-1, 2)).astype(np.int32)
            cv2.fillPoly(modify, [seg_coords], color=(1))
            prev_mask = gt_mask - modify
            prev_mask[prev_mask < 0] = 0.0
            modify_mask = modify

        else:
            modify_mask = modify_mask.astype(np.uint8)
            modify_retvals, modify_labels, modify_stats, _ = cv2.connectedComponentsWithStats(modify_mask)
            if modify_retvals == 1:
                pass
            else:
                if modify_retvals == 2:
                    sampling_num = 1
                else:
                    sampling_num = random.randint(1,modify_retvals-1)

                modify_mask[modify_labels != sampling_num] = 0
            modify_mask = modify_mask.astype(np.float32)
            prev_mask = gt_mask - modify_mask
            prev_mask[prev_mask<0] = 0

        if self.prompt == 'click':
            point_label, pt = random_click(modify_mask , point_label)
            pt = tuple(pt)
        crop_size = (256,256)

        if case_num == 3 or case_num == 0:
            point_label = 0

        if case_num == 2 or case_num == 3:
            origin_prev = prev_mask
        else:
            origin_prev = gt_mask.copy()
            prev_mask, xy = crop_with_padding_no_padding(prev_mask,pt,crop_size)
            origin_prev[xy[0]:xy[1],xy[2]:xy[3]] = prev_mask

        gt_mask = torch.as_tensor(gt_mask) 
        prev_mask = torch.as_tensor(origin_prev) 

        pt = np.array(pt)
        pt = (pt / self.out_size) * self.img_size

        if case_num == 3:
            tmp = gt_mask.clone()
            gt_mask = prev_mask.clone()
            prev_mask = tmp

        if self.transform:
            image = self.transform(image)

        image_meta_dict = {'filename_or_obj':image_name}

        return {"image":image, 
               "mask": prev_mask.unsqueeze(0), "label": gt_mask.unsqueeze(0), "p_label" : point_label, "pt": pt, "text": text,'image_meta_dict':image_meta_dict}
