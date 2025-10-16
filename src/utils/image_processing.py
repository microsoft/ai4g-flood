# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import numpy as np
from rasterio.warp import Resampling, reproject
from scipy.ndimage import binary_dilation



def db_scale(x):
    """Convert to decibel scale and shift towards middle of 0-255 range."""
    with np.errstate(invalid="ignore", divide="ignore"):
        xnew = 10 * np.log10(x)
    xnew = xnew * 2 + 135
    return np.clip(np.nan_to_num(xnew, 0), 0, 255)


def pad_to_nearest(input_array, multiple, pad_indices):
    pad_amts = []
    for i in range(len(input_array.shape)):
        if i in pad_indices:
            lendim = input_array.shape[i]
            pad_amt = 0 if lendim % multiple == 0 else multiple - lendim % multiple
            pad_amt = (0, pad_amt)
        else:
            pad_amt = (0, 0)
        pad_amts.append(pad_amt)
    pad = tuple(pad_amts)
    return np.pad(input_array, pad, mode="constant", constant_values=0)


def create_patches(image, patch_size, stride):
    height, width = image.shape[0], image.shape[1]
    patches = []
    for y in range(0, height - patch_size[0] + 1, stride):
        for x in range(0, width - patch_size[1] + 1, stride):
            patch = image[y : y + patch_size[0], x : x + patch_size[1]]
            patch = patch.transpose(2, 0, 1)
            patches.append(patch)

    return patches


def reconstruct_image_from_patches(patches, image_size, patch_size, stride):
    """Reconstruct image from patches."""
    height, width = image_size
    reconstructed_image = np.zeros((height, width))
    pixel_counts = np.zeros((height, width))
    patch_counter = 0
    for y in range(0, height - patch_size[0] + 1, stride):
        for x in range(0, width - patch_size[1] + 1, stride):
            patch = patches[patch_counter]
            reconstructed_image[y : y + patch_size[0], x : x + patch_size[1]] += patch
            pixel_counts[y : y + patch_size[0], x : x + patch_size[1]] += 1
            patch_counter += 1
    return reconstructed_image, pixel_counts


def reproject_image(source, dst_crs, dst_transform, dst_shape):
    """Reproject image to new coordinate system."""
    destination = np.zeros(dst_shape, source.dtype)
    reproject(
        source=source,
        destination=destination,
        src_transform=source.transform,
        src_crs=source.crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.nearest,
    )
    return destination


def calculate_flood_change(vv_pre, vh_pre, vv_post, vh_post, params):
    """Calculate flood change based on VV and VH thresholds."""
    vv_change = (
        (vv_post < params["vv_threshold"])
        & (vv_pre > params["vv_threshold"])
        & ((vv_pre - vv_post) > params["delta_amplitude"])
    ).astype(int)
    vh_change = (
        (vh_post < params["vh_threshold"])
        & (vh_pre > params["vh_threshold"])
        & ((vh_pre - vh_post) > params["delta_amplitude"])
    ).astype(int)

    zero_index = (
        (vv_post < params["vv_min_threshold"])
        | (vv_pre < params["vv_min_threshold"])
        | (vh_post < params["vh_min_threshold"])
        | (vh_pre < params["vh_min_threshold"])
    )

    vv_change[zero_index] = 0
    vh_change[zero_index] = 0

    return np.stack((vv_change, vh_change), axis=2)

def apply_buffer(pred_image, buffer_size):
    """
    Apply morphological dilation to create a buffer around flood detections.
    
    Args:
        pred_image: Binary prediction image (0 or 255)
        buffer_size: Number of pixels to dilate (4 pixels ≈ 80m at 20m resolution)
    
    Returns:
        Buffered prediction image
    """
    if buffer_size <= 0:
        return pred_image
    
    # Convert to binary (handle NaN values)
    binary_mask = np.zeros_like(pred_image, dtype=bool)
    binary_mask[pred_image == 255] = True
    
    # Create a square structuring element
    structure = np.ones((2 * buffer_size + 1, 2 * buffer_size + 1), dtype=bool)
    
    # Apply binary dilation
    buffered_mask = binary_dilation(binary_mask, structure=structure)
    
    # Convert back to original format
    buffered_image = np.zeros_like(pred_image)
    buffered_image[buffered_mask] = 255
    
    return buffered_image