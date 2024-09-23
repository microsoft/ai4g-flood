import argparse
import os
import torch
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.transform import Affine
from utils.model import load_model
from utils.image_processing import db_scale, pad_to_nearest, create_patches, reconstruct_image_from_patches

def parse_args():
    parser = argparse.ArgumentParser(description='Run flood detection on local image pairs.')
    parser.add_argument('--pre_vv', type=str, required=True, help='Path to pre-event VV polarization image.')
    parser.add_argument('--pre_vh', type=str, required=True, help='Path to pre-event VH polarization image.')
    parser.add_argument('--post_vv', type=str, required=True, help='Path to post-event VV polarization image.')
    parser.add_argument('--post_vh', type=str, required=True, help='Path to post-event VH polarization image.')
    parser.add_argument('--model_path', type=str, default='./models/ai4g_sar_model.ckpt', help='Path to the trained model file.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output files.')
    parser.add_argument('--scale_factor', type=float, default=1.0, help='Scale factor for image resolution.')
    parser.add_argument('--vv_threshold', type=int, default=100, help='VV threshold for water detection.')
    parser.add_argument('--vh_threshold', type=int, default=90, help='VH threshold for water detection.')
    parser.add_argument('--delta_amplitude', type=int, default=10, help='Required change in amplitude for flood detection.')
    parser.add_argument('--vv_min_threshold', type=int, default=75, help='Minimum VV for water detection.')
    parser.add_argument('--vh_min_threshold', type=int, default=70, help='Minimum VH for water detection.')
    parser.add_argument('--no_data_flag', type=int, default=15, help='Value to use for no data areas.')
    parser.add_argument('--patch_size', type=int, default=1024, help='Size of patches to process at once.')
    return parser.parse_args()

def read_and_preprocess(file_path, scale_factor):
    with rasterio.open(file_path) as src:
        if scale_factor == 1:
            image = src.read(1)
        else:
            image = src.read(
                1,
                out_shape=(
                    src.count,
                    int(src.height * scale_factor),
                    int(src.width * scale_factor)
                ),
                resampling=Resampling.bilinear
            )
        transform = src.transform * src.transform.scale(
            (src.width / image.shape[-1]),
            (src.height / image.shape[-2])
        )
        return db_scale(image), src.crs, transform

def calculate_flood_change(vv_pre, vh_pre, vv_post, vh_post, params):
    vv_change = ((vv_post < params['vv_threshold']) & 
                 (vv_pre > params['vv_threshold']) & 
                 ((vv_pre - vv_post) > params['delta_amplitude'])).astype(int)
    vh_change = ((vh_post < params['vh_threshold']) & 
                 (vh_pre > params['vh_threshold']) & 
                 ((vh_pre - vh_post) > params['delta_amplitude'])).astype(int)
    
    zero_index = ((vv_post < params['vv_min_threshold']) | 
                  (vv_pre < params['vv_min_threshold']) | 
                  (vh_post < params['vh_min_threshold']) | 
                  (vh_pre < params['vh_min_threshold']))
    
    vv_change[zero_index] = 0
    vh_change[zero_index] = 0
    
    return np.stack((vv_change, vh_change), axis=2)

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load images
    vv_pre, vv_pre_crs, vv_pre_transform = read_and_preprocess(args.pre_vv, args.scale_factor)
    vh_pre, _, _ = read_and_preprocess(args.pre_vh, args.scale_factor)
    vv_post, vv_post_crs, vv_post_transform = read_and_preprocess(args.post_vv, args.scale_factor)
    vh_post, _, _ = read_and_preprocess(args.post_vh, args.scale_factor)

    # Use vv_post shape as the target shape
    target_shape = vv_post.shape
    vv_pre = reproject_image(vv_pre, vv_post_crs, vv_post_transform, target_shape)
    vh_pre = reproject_image(vh_pre, vv_post_crs, vv_post_transform, target_shape)
    vh_post = reproject_image(vh_post, vv_post_crs, vv_post_transform, target_shape)

    # Calculate flood change
    flood_change = calculate_flood_change(vv_pre, vh_pre, vv_post, vh_post, vars(args))

    # Prepare input for the model
    input_size = 128
    flood_change = pad_to_nearest(flood_change, input_size, [0, 1])
    patches = create_patches(flood_change, (input_size, input_size), input_size)

    # Load model
    model = load_model(args.model_path, device, in_channels=2, n_classes=2)
    model.eval()

    # Run inference
    predictions = []
    with torch.no_grad():
        for i in range(0, len(patches), args.patch_size):
            batch = patches[i:i+args.patch_size]
            if len(batch) == 0:
                continue
            batch_tensor = torch.from_numpy(np.array(batch)).float().to(device)
            output = model(batch_tensor)
            pred = torch.argmax(output, dim=1)
            pred = (pred * 255).to(torch.int)
            pred[pred == args.no_data_flag] = args.no_data_flag + 1
            pred[(batch_tensor[:, 0] == 0) | (batch_tensor[:, 1] == 0)] = args.no_data_flag
            predictions.extend(pred.cpu().numpy())

    # Reconstruct the image
    pred_image, _ = reconstruct_image_from_patches(predictions, flood_change.shape[:2], (input_size, input_size), input_size)
    pred_image = pred_image[:target_shape[0], :target_shape[1]]  # Crop to original size

    # Save the result
    os.makedirs(args.output_dir, exist_ok=True)
    output_filename = os.path.join(args.output_dir, "flood_prediction.tif")
    save_prediction(pred_image, output_filename, vv_post_crs, vv_post_transform)
    print(f"Flood prediction saved to {output_filename}")

def reproject_image(image, dst_crs, dst_transform, dst_shape):
    reprojected, _ = reproject(
        image,
        np.empty(dst_shape, dtype=image.dtype),
        src_transform=image.transform,
        src_crs=image.crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.nearest
    )
    return reprojected

def save_prediction(pred_image, output_filename, crs, transform):
    with rasterio.open(output_filename, 'w', driver='GTiff', height=pred_image.shape[0], width=pred_image.shape[1],
                       count=1, dtype=pred_image.dtype, crs=crs, transform=transform) as dst:
        dst.write(pred_image, 1)

if __name__ == "__main__":
    main()