import argparse
import torch
import os
from utils.flood_data_module import FloodDataModule
from utils.model import load_model
from utils.image_processing import reconstruct_image_from_patches
import rasterio

# set constants that are used throughout the script
in_channels = 2
n_classes = 2

def parse_args():
    parser = argparse.ArgumentParser(description='Run flood detection on Planetary Computer data.')
    parser.add_argument('--region', type=str, required=True, help='Region name for flood detection.')
    parser.add_argument('--device_index', type=int, required=True, help='Device index (integer).')  
    parser.add_argument('--scale_factor', type=int, default=3, help='How much to scale resolution (3 = going from 10m to 30m)')
    parser.add_argument('--start_date', type=str, required=True, help='Start date for image search (YYYY-MM-DD).')
    parser.add_argument('--end_date', type=str, required=True, help='End date for image search (YYYY-MM-DD).')
    parser.add_argument('--model_path', type=str, default='./models/ai4g_sar_model.ckpt', help='Path to the trained model file.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output files.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference.')
    parser.add_argument('--no_data_flag', type=int, default=15, help='Value to use for no data areas.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker processes for data loading.')
    return parser.parse_args()

def save_prediction(pred_image, output_filename, crs, transform):
    with rasterio.open(output_filename, 'w', driver='GTiff', height=pred_image.shape[0], width=pred_image.shape[1],
                       count=1, dtype=pred_image.dtype, crs=crs, transform=transform) as dst:
        dst.write(pred_image, 1)

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = load_model(args.model_path, device, in_channels=in_channels, n_classes=n_classes)
    model.eval()

    # Setup data module
    dm = FloodDataModule(args.batch_size, args.num_workers, args.region, [args.start_date, args.end_date], args.scale_factor)
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
                    for patch in range(0, patches.size(0), args.patch_size):
                        inputs = patches[patch:patch + args.patch_size].half().to(device)
                        if inputs.size(0) == 0:
                            continue
                        _, predicted = torch.max(model(inputs), 1)
                        predicted = (predicted * 255).to(torch.int)
                        # if any values are predicted to be the same as the no data flag, set them to the no data flag + 1 so they're not marked as no data
                        predicted[predicted == args.no_data_flag] = args.no_data_flag + 1
                        predicted[(inputs[:, 0] == 0) + (inputs[:, 1] == 0)] = args.no_data_flag
                        predicted = predicted.cpu().numpy()
                        preds.append(predicted)

                    predicted = np.concatenate(preds)
                    predimg, _ = reconstruct_image_from_patches(predicted, padded_shapes[i], (args.input_size, args.input_size), args.input_size)
                    predimg = predimg[:orig_shapes[i][0], :orig_shapes[i][1]]  # Truncate to original shape

                    output_dir = os.path.join(args.output_dir, str(year), str(month), str(day))
                    os.makedirs(output_dir, exist_ok=True)
                    output_filename = os.path.join(output_dir, f"{filename}_flood_prediction.tif")
                    save_prediction(predimg, output_filename, crs[i], transforms[i])
                except Exception as e:
                    print(e)

                    
if __name__ == "__main__":
    main()