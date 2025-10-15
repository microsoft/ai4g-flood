# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import argparse
import os

import numpy as np
import planetary_computer
import rasterio
import torch

from utils.flood_data_module import FloodDataModule
from utils.image_processing import reconstruct_image_from_patches, apply_buffer
from utils.model import load_model

# set constants that are used throughout the script
in_channels = 2
n_classes = 2
pc_default_resolution = 10 # in meters


def parse_args():
    parser = argparse.ArgumentParser(description="Run flood detection on Planetary Computer data.")
    parser.add_argument("--region", type=str, required=True, help="Region name for flood detection.")
    parser.add_argument("--device_index", type=int, required=True, help="Device index (integer).")
    parser.add_argument(
        "--planetary_computer_subscription_key", type=str, default="", help="Planetary Computer subscription key."
    )
    parser.add_argument(
        "--scale_factor", type=int, default=2, help="How much to scale resolution (2 = going from 10m to 20m)"
    )
    parser.add_argument("--start_date", type=str, required=True, help="Start date for image search (YYYY-MM-DD).")
    parser.add_argument("--end_date", type=str, required=True, help="End date for image search (YYYY-MM-DD).")
    parser.add_argument(
        "--model_path", type=str, default="./models/ai4g_sar_model.ckpt", help="Path to the trained model file."
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output files.")
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for inference (not recommended to change)"
    )
    parser.add_argument("--num_workers", type=int, default=4, help="Number of worker processes for data loading.")
    parser.add_argument("--input_size", type=int, default=128, help="Size of input patches (the model expects 128)")
    parser.add_argument(
        "--run_vv_only", action="store_true", default=False, help="Run flood detection using only VV polarization."
    )
    parser.add_argument(
        "--keep_all_predictions",
        action="store_true",
        default=False,
        help="Whether to restrict flood detections to pixels that are within water thresholds",
    )
    parser.add_argument(
        "--buffer_size",
        type=int,
        default=4,
        help="Buffer size in pixels for dilation (4 pixels ≈ 80m at 20m resolution). Set to 0 to disable buffering."
    )
    return parser.parse_args()


def save_prediction(pred_image, output_filename, crs, transform):
    pred_image[pred_image == 0] = np.nan
    with rasterio.open(
        output_filename,
        "w",
        driver="GTiff",
        height=pred_image.shape[0],
        width=pred_image.shape[1],
        count=1,
        dtype=pred_image.dtype,
        crs=crs,
        transform=transform,
        compress="lzw",
        nodata=np.nan,
    ) as dst:
        dst.write(pred_image, 1)


def main():
    args = parse_args()
    planetary_computer.settings.set_subscription_key(args.planetary_computer_subscription_key)
    device = torch.device(
        f"cuda:{args.device_index}" if (torch.cuda.is_available() and args.device_index >= 0) else "cpu"
    )

    # Load model
    model = load_model(args.model_path, device, in_channels=in_channels, n_classes=n_classes)
    model.eval()

    # Setup data module
    dm = FloodDataModule(
        args.batch_size,
        args.num_workers,
        args.region,
        [args.start_date, args.end_date],
        1.0 / args.scale_factor,
        args.run_vv_only,
    )
    dm.setup()
    # Run inference
    with torch.no_grad():
        for batch in dm.test_dataloader():
            filenames, years, months, days, orig_shapes, padded_shapes, patches, crs, transforms, ignore_flags = batch
            for i, (filename, year, month, day) in enumerate(zip(filenames, years, months, days)):
                if ignore_flags[i]:
                    print(f"Skipping {filename} due to data issues.")
                    continue
                try:
                    preds = []
                    for patch in range(0, patches.size(0), args.input_size):
                        inputs = patches[patch : patch + args.input_size].to(device)
                        if device.type == "cuda":
                            inputs = inputs.half()
                        else:
                            inputs = inputs.float()
                        if inputs.size(0) == 0:
                            continue
                        _, predicted = torch.max(model(inputs), 1)
                        predicted = (predicted * 255).to(torch.int)
                        if not args.keep_all_predictions:
                            predicted[(inputs[:, 0] == 0) + (inputs[:, 1] == 0)] = 0
                        predicted = predicted.cpu().numpy()
                        preds.append(predicted)

                    predicted_all = np.concatenate(preds)
                    predimg, _ = reconstruct_image_from_patches(
                        predicted_all, padded_shapes[i], (args.input_size, args.input_size), args.input_size
                    )
                    predimg = predimg[: orig_shapes[i][0], : orig_shapes[i][1]]  # Truncate to original shape

                    output_dir = os.path.join(args.output_dir, str(year), str(month), str(day))
                    os.makedirs(output_dir, exist_ok=True)
                    output_filename = os.path.join(output_dir, f"{filename}_flood_prediction.tif")
                    # Apply buffer (dilation) to predictions
                    if args.buffer_size > 0:
                        print(f"Applying {args.buffer_size}-pixel buffer (={args.buffer_size * pc_default_resolution * args.scale_factor}m at {pc_default_resolution * args.scale_factor}m resolution)...")
                        predimg = apply_buffer(predimg, args.buffer_size)
                    save_prediction(predimg, output_filename, crs[i], transforms[i])
                except Exception as e:
                    print(f"Exception {e} in run_flood_detection_planetary_computer.py")
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
