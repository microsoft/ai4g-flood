# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import Affine
from .image_processing import db_scale, pad_to_nearest, create_patches
import time
import re
import pandas as pd
import planetary_computer as pc

class FloodDataset(Dataset):
    def __init__(self, dataframe, scale_factor, input_size=128, vv_threshold=100, vh_threshold=90, 
                 delta_amplitude=10, vv_min_threshold=75, vh_min_threshold=70, run_vv_only=False):
        self.dataset = dataframe
        self.scale_factor = scale_factor
        self.input_size = input_size
        self.vv_threshold = vv_threshold
        self.vh_threshold = vh_threshold
        self.delta_amplitude = delta_amplitude
        self.vv_min_threshold = vv_min_threshold
        self.vh_min_threshold = vh_min_threshold
        self.run_vv_only = run_vv_only
        self.pad_amt = input_size
        self.patch_size = (input_size, input_size)
        self.stride = input_size

    def __len__(self):
        return len(self.dataset)
    
    def read_image_data(self, url, max_retries=2, delay=0.1):  
        retries = 0  
        ignore_flag = 0
        url = "/vsicurl/" + pc.sign(url)
        while retries < max_retries:  
            try:  
                with rasterio.open(url) as dataset: 
                    if self.scale_factor == 1:                 
                        image = dataset.read(1)
                        data_crs = dataset.crs
                        data_transform = dataset.transform
                    else:
                        image = dataset.read(1,  
                                        out_shape=(  
                                            dataset.count,  
                                            int(dataset.height * self.scale_factor),  
                                            int(dataset.width * self.scale_factor)  
                                        ),  
                                        resampling=Resampling.bilinear  
                                    )  
                        data_crs = dataset.crs  
                        data_transform = Affine(dataset.transform.a / self.scale_factor, dataset.transform.b, dataset.transform.c,    
                                    dataset.transform.d, dataset.transform.e / self.scale_factor, dataset.transform.f) 
                image = db_scale(image)
                return image, data_crs, data_transform, ignore_flag 
 
            except Exception as e:  
                retries += 1  
                print(f"Attempt {retries} failed for url {url}. Error: {str(e)}. Retrying after {delay * retries} seconds.")  
                time.sleep(delay * retries)  
        dummy_shape = (1, 128, 128)  # You can adjust this shape as needed  
        null_image = np.zeros(dummy_shape, dtype=np.uint8)  # or np.full(dummy_shape, np.nan)  
        null_crs = None  # or rasterio.crs.CRS.from_epsg(4326) for a generic CRS  
        null_transform = Affine.identity()  # identity transform as a placeholder  
        ignore_flag = 1
        return null_image, null_crs, null_transform, ignore_flag

    def __getitem__(self, index):
        df_row = self.dataset.iloc[index]
        file_scene_name, year, month, day = self.extract_from_url(df_row['vv_image_path'])
        vv_pre, vv_ref_crs, vv_ref_transform, vv_pre_ignore = self.read_image_data(df_row['vv_image_pre_path'])
        orig_shape = vv_pre.shape
        vv_post, vv_crs, vv_transform, vv_post_ignore = self.read_image_data(df_row['vv_image_path'])
        if self.run_vv_only:
            vh_pre, vh_ref_crs, vh_ref_transform, vh_pre_ignore = vv_pre, vv_ref_crs, vv_ref_transform, vv_pre_ignore
            vh_post, vh_crs, vh_transform, vh_post_ignore = vv_post, vv_crs, vv_transform, vv_post_ignore
        else:
            vh_pre, vh_ref_crs, vh_ref_transform, vh_pre_ignore = self.read_image_data(df_row['vh_image_pre_path'])
            vh_post, vh_crs, vh_transform, vh_post_ignore = self.read_image_data(df_row['vh_image_path'])
        
        if vv_pre_ignore + vv_post_ignore + vh_pre_ignore + vh_post_ignore > 0:
            return {'filename': file_scene_name, 'year': year, 'month': month, 'day': day, 
                    'patches': [np.zeros((2, self.input_size, self.input_size), dtype=np.uint8)], 
                    'orig_shape': [0, 0], 'padded_shape': [0, 0],
                    'crs': vv_ref_crs, 'transform': vv_ref_transform, 'ignore_flag': 1} 
        
        # Reproject images
        vv_post = self.reproject_image(vv_post, vv_crs, vv_transform, vv_ref_crs, vv_ref_transform, vv_pre.shape)
        if self.run_vv_only:
            vh_pre = vv_pre
            vh_post = vv_post
        else:
            if vv_pre.shape != vh_pre.shape:
                vh_pre = self.reproject_image(vh_pre, vh_ref_crs, vh_ref_transform, vv_ref_crs, vv_ref_transform, vv_pre.shape)
            vh_post = self.reproject_image(vh_post, vh_crs, vh_transform, vv_ref_crs, vv_ref_transform, vv_pre.shape)
        # Calculate change
        vv_change = ((vv_post < self.vv_threshold) & (vv_pre > self.vv_threshold) & ((vv_pre - vv_post) > self.delta_amplitude)).astype(int)
        if self.run_vv_only:
            vh_change = vv_change
            zero_index = (vv_post < self.vv_min_threshold) | (vv_pre < self.vv_min_threshold) 
        else:
            vh_change = ((vh_post < self.vh_threshold) & (vh_pre > self.vh_threshold) & ((vh_pre - vh_post) > self.delta_amplitude)).astype(int)
            zero_index = (vv_post < self.vv_min_threshold) | (vv_pre < self.vv_min_threshold) | (vh_post < self.vh_min_threshold) | (vh_pre < self.vh_min_threshold)
        
        vv_change[zero_index] = 0
        vh_change[zero_index] = 0
        rgb_image = np.stack((vv_change, vh_change), axis=2)
        rgb_image = pad_to_nearest(rgb_image, self.pad_amt, [0, 1])
        image_patches = create_patches(rgb_image, self.patch_size, self.stride)
        
        return {'filename': file_scene_name, 'year': year, 'month': month, 'day': day, 'patches': image_patches, 
                'orig_shape': orig_shape, 'padded_shape': rgb_image.shape[:2],
                'crs': vv_ref_crs, 'transform': vv_ref_transform, 'ignore_flag': 0} 
    
    @staticmethod
    def extract_from_url(url):
        pattern = r'/([^/]+)_\w+/measurement/'  
        match = re.search(pattern, url)  
        if match:  
            scene_name = match.group(1) 
            date_match = re.search(r'(\d{4})(\d{2})(\d{2})T', scene_name)  
            if date_match:  
                year, month, day = date_match.groups()
            else:
                year, month, day = np.nan, np.nan, np.nan
            return scene_name, year, month, day
        else:  
            return '', np.nan, np.nan, np.nan

    @staticmethod
    def reproject_image(image, src_crs, src_transform, dst_crs, dst_transform, dst_shape):
        reprojected, _ = reproject(
            image,
            np.empty(dst_shape, dtype=image.dtype),
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest
        )
        return reprojected