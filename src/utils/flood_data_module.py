# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import torch
import pandas as pd
import numpy as np
from datetime import datetime
import pystac_client
import planetary_computer
from torch.utils.data import DataLoader
from lightning import LightningDataModule
import time
from .flood_dataset import FloodDataset
from data.region_polygons import get_polygon

class FloodDataModule(LightningDataModule):
    def __init__(self, batch_size, workers, region, daterange, scale_factor, run_vv_only=False):
        super().__init__()
        self.batch_size = batch_size
        self.workers = workers
        self.region = region
        self.daterange = daterange
        self.scale_factor = scale_factor
        self.run_vv_only = run_vv_only

    def retrieve_from_planetary_computer(self, num_retries=100):
        polygon_coords = get_polygon(self.region)
        polygon_geometry = {"type": "Polygon", "coordinates": [polygon_coords]}
        for _ in range(num_retries):
            try:
                catalog = pystac_client.Client.open(
                    "https://planetarycomputer.microsoft.com/api/stac/v1",
                )
                search = catalog.search(
                    collections=["sentinel-1-rtc"],
                    intersects=polygon_geometry,
                    datetime=f"{self.daterange[0]}/{self.daterange[1]}"
                )
                return search.item_collection()
            except Exception as e:
                print(f'Exception {e} in flood_data_module.py')
                time.sleep(60)
        raise Exception("Failed to retrieve data from Planetary Computer")

    def get_items_keep(self, items):
        vv_paths, vh_paths, items_keep = [], [], []
        for item in items:
            assets = item.assets
            if self.run_vv_only:
                if 'vv' not in assets:
                    continue
                assets['vh'] = assets['vv']
            else:
                if 'vv' not in assets or 'vh' not in assets:
                    continue
            vv_paths.append(assets['vv'].href)
            vh_paths.append(assets['vh'].href)
            items_keep.append(item)
        return items_keep, vv_paths, vh_paths

    def create_ref_df(self, items_keep, vv_paths, vh_paths, max_time_delta=300, max_coord_diff = 1, max_day_delta=30):
        df = pd.DataFrame({
            'obs_dt': [pd.to_datetime(item.properties['end_datetime']) for item in items_keep],
            'bbox': [item.bbox for item in items_keep]
        })
        df['mean_long'] = df.bbox.apply(lambda x: (x[0] + x[2]) / 2)
        df['mean_lat'] = df.bbox.apply(lambda x: (x[1] + x[3]) / 2)
        df['obs_time'] = df.obs_dt.apply(lambda x: datetime.combine(datetime(2000, 1, 1), x.time()))
        vv, vh, vv_ref, vh_ref = [], [], [], []
        for i, row in df.iterrows():
            reference_time = row.obs_time
            timedeltas = (df['obs_time'] - reference_time).dt.total_seconds().abs()
            dftmp = df[((row.obs_dt - df.obs_dt).dt.days < max_day_delta) & 
                    ((row.obs_dt - df.obs_dt).dt.days >0) & 
                    ((row.mean_long - df.mean_long).abs() < max_coord_diff) & 
                    ((row.mean_lat - df.mean_lat).abs() < max_coord_diff) & 
                    (timedeltas < max_time_delta)]
            if not dftmp.empty:
                ref_index = dftmp.obs_dt.idxmax()
                vv.append(vv_paths[i])
                vh.append(vh_paths[i])
                vv_ref.append(vv_paths[ref_index])
                vh_ref.append(vh_paths[ref_index])

        return pd.DataFrame({'vv_image_path': vv, 'vh_image_path': vh, 'vv_image_pre_path': vv_ref, 'vh_image_pre_path': vh_ref})

    def setup(self, stage=None):
        items = self.retrieve_from_planetary_computer()
        items_keep, vv_paths, vh_paths = self.get_items_keep(items)
        self.test_df = self.create_ref_df(items_keep, vv_paths, vh_paths)
        self.test_df.to_csv(f'{self.region}_test.csv', index=False)
        self.test_ds = FloodDataset(self.test_df, self.scale_factor, run_vv_only=self.run_vv_only)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.workers, 
                          shuffle=False, pin_memory=True, collate_fn=self.collate_fn)

    @staticmethod
    def collate_fn(batch):
        filenames = [item['filename'] for item in batch]
        years = [item['year'] for item in batch]
        months = [item['month'] for item in batch]
        days = [item['day'] for item in batch]
        patches = [torch.from_numpy(patch) for item in batch for patch in item['patches']]
        orig_shapes = [item['orig_shape'] for item in batch]
        padded_shapes = [item['padded_shape'] for item in batch]
        crs = [item['crs'] for item in batch]
        transform = [item['transform'] for item in batch]
        ignore_flag = [item['ignore_flag'] for item in batch]
        return filenames, years, months, days, orig_shapes, padded_shapes, torch.stack(patches), crs, transform, ignore_flag