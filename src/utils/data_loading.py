import os
import rasterio
from rasterio.warp import reproject, Resampling
from .image_processing import db_scale

def get_vv_vh(scene_dir, scale_factor=1):
    """Load VV and VH images from a scene directory."""
    with rasterio.open(os.path.join(scene_dir, f"{os.path.basename(scene_dir)}_VV.tif")) as vvsrc:
        vv = vvsrc.read(1)
        crs = vvsrc.crs
        transform = vvsrc.transform
        
        if scale_factor != 1:
            vv = vvsrc.read(1,
                out_shape=(
                    vvsrc.count,
                    int(vvsrc.height * scale_factor),
                    int(vvsrc.width * scale_factor)
                ),
                resampling=Resampling.bilinear
            )
            transform = vvsrc.transform * vvsrc.transform.scale(
                (vvsrc.width / vv.shape[-1]),
                (vvsrc.height / vv.shape[-2])
            )
    
    with rasterio.open(os.path.join(scene_dir, f"{os.path.basename(scene_dir)}_VH.tif")) as vhsrc:
        vh = vhsrc.read(1)
        if scale_factor != 1:
            vh = vhsrc.read(1,
                out_shape=(
                    vhsrc.count,
                    int(vhsrc.height * scale_factor),
                    int(vhsrc.width * scale_factor)
                ),
                resampling=Resampling.bilinear
            )
    
    return db_scale(vv), db_scale(vh), crs, transform

def load_and_preprocess(pre_scene, post_scene, scale_factor=1):
    """Load and preprocess pre and post scenes."""
    vv_pre, vh_pre, vv_ref_crs, vv_ref_transform = get_vv_vh(pre_scene, scale_factor)
    vv_post, vh_post, _, _ = get_vv_vh(post_scene, scale_factor)
    
    # Reproject post images to match pre images
    vv_post = reproject_image(vv_post, vv_ref_crs, vv_ref_transform, vv_pre.shape)
    vh_post = reproject_image(vh_post, vv_ref_crs, vv_ref_transform, vv_pre.shape)
    
    return vv_pre, vh_pre, vv_post, vh_post, vv_ref_crs, vv_ref_transform