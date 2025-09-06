# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Script to merge flood detection results from multiple GeoTIFF files.

This script processes the output of run_flood_detection_planetary_computer.py,
which creates a folder structure like:
- results/2025/08/10/*.tif
- results/2025/08/11/*.tif

The script:
1. Projects all GeoTIFF files to EPSG:3857 (Web Mercator)
2. Merges overlapping rasters
3. Maps values: np.nan -> 0, 255 -> 1
4. Saves the result with ZSTD compression and tiling
"""

import argparse
import os
import glob
from typing import List, Tuple, Optional

# Import statements - these will be available when the environment is properly set up
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import rasterio
    from rasterio.warp import reproject, calculate_default_transform, Resampling
    from rasterio.enums import Resampling
    from rasterio.merge import merge
    from rasterio.crs import CRS
    from rasterio.transform import Affine
    import rasterio.features
    import rasterio.mask
    RASTERIO_AVAILABLE = True
except ImportError as e:
    RASTERIO_AVAILABLE = False
    _IMPORT_ERROR = e


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Merge flood detection GeoTIFF results into a single file.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python merge.py --input_dir results --output merged_flood_predictions.tif
  python merge.py --input_dir /path/to/results --output /path/to/output.tif --target_crs EPSG:4326
        """
    )
    
    parser.add_argument(
        '--input_dir', 
        type=str, 
        required=True,
        help='Input directory containing the year/month/day structure with .tif files'
    )
    
    parser.add_argument(
        '--output', 
        type=str, 
        required=True,
        help='Output path for the merged GeoTIFF file'
    )
    
    parser.add_argument(
        '--target_crs', 
        type=str, 
        default='EPSG:3857',
        help='Target CRS for reprojection (default: EPSG:3857 - Web Mercator)'
    )
    
    parser.add_argument(
        '--resolution', 
        type=float, 
        default=None,
        help='Target resolution in target CRS units (default: auto-detect from input)'
    )
    
    parser.add_argument(
        '--resampling', 
        type=str, 
        default='nearest',
        choices=['nearest', 'bilinear', 'cubic', 'average'],
        help='Resampling method for reprojection (default: nearest)'
    )
    
    parser.add_argument(
        '--merge_method', 
        type=str, 
        default='max',
        choices=['first', 'last', 'min', 'max', 'mean'],
        help='Method for handling overlapping pixels (default: max)'
    )
    
    parser.add_argument(
        '--compress', 
        type=str, 
        default='zstd',
        choices=['lzw', 'deflate', 'zstd', 'none'],
        help='Compression method (default: zstd)'
    )
    
    parser.add_argument(
        '--tiled', 
        action='store_true',
        default=True,
        help='Create tiled output (default: True)'
    )
    
    parser.add_argument(
        '--tile_size', 
        type=int, 
        default=512,
        help='Tile size for tiled output (default: 512)'
    )
    
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()


def find_tif_files(input_dir: str) -> List[str]:
    """
    Find all .tif files in the input directory structure.
    
    Args:
        input_dir: Root directory to search for .tif files
        
    Returns:
        List of paths to .tif files
    """
    pattern = os.path.join(input_dir, '**', '*.tif')
    tif_files = glob.glob(pattern, recursive=True)
    
    # Also try .tiff extension
    pattern_tiff = os.path.join(input_dir, '**', '*.tiff')
    tif_files.extend(glob.glob(pattern_tiff, recursive=True))
    
    return sorted(tif_files)


def get_resampling_method(method_name: str):
    """Convert string resampling method to rasterio enum."""
    if not RASTERIO_AVAILABLE:
        return method_name  # Return string if rasterio not available
    
    method_map = {
        'nearest': Resampling.nearest,
        'bilinear': Resampling.bilinear,
        'cubic': Resampling.cubic,
        'average': Resampling.average
    }
    return method_map.get(method_name, Resampling.nearest)


def reproject_raster(src_path: str, target_crs: str, target_resolution: Optional[float] = None, 
                    resampling_method=None) -> Tuple:
    """
    Reproject a raster to target CRS.
    
    Args:
        src_path: Path to source raster
        target_crs: Target CRS (e.g., 'EPSG:3857')
        target_resolution: Target resolution (optional)
        resampling_method: Resampling method
        
    Returns:
        Tuple of (reprojected_data, transform, width, height)
    """
    if not RASTERIO_AVAILABLE:
        raise ImportError("rasterio is required for raster reprojection")
    
    with rasterio.open(src_path) as src:
        # Calculate transform and dimensions for target CRS
        if target_resolution is not None:
            # Use specified resolution
            transform, width, height = calculate_default_transform(
                src.crs, target_crs, src.width, src.height, *src.bounds,
                resolution=target_resolution
            )
        else:
            # Auto-calculate resolution
            transform, width, height = calculate_default_transform(
                src.crs, target_crs, src.width, src.height, *src.bounds
            )
        
        # Create destination array
        destination = np.zeros((height, width), dtype=src.dtypes[0])
        
        # Reproject
        reproject(
            source=rasterio.band(src, 1),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=target_crs,
            resampling=resampling_method or Resampling.nearest
        )
        
        return destination, transform, width, height


def process_raster_values(data):
    """
    Process raster values according to the requirements:
    - Convert np.nan to 0
    - Convert 255 to 1
    - Keep other values as they are (though typically there should only be nan and 255)
    
    Args:
        data: Input raster data
        
    Returns:
        Processed raster data
    """
    if not NUMPY_AVAILABLE:
        raise ImportError("numpy is required for raster processing")
    
    processed = data.copy()
    
    # Convert np.nan to 0
    processed[np.isnan(processed)] = 0
    
    # Convert 255 to 1
    processed[processed == 255] = 1
    
    return processed.astype(np.uint8)


def merge_rasters(file_paths: List[str], target_crs: str, target_resolution: Optional[float] = None,
                 resampling_method=None, merge_method: str = 'max',
                 verbose: bool = False) -> Tuple:
    """
    Merge multiple rasters into a single raster.
    
    Args:
        file_paths: List of paths to raster files
        target_crs: Target CRS for reprojection
        target_resolution: Target resolution (optional)
        resampling_method: Resampling method for reprojection
        merge_method: Method for handling overlapping pixels
        verbose: Enable verbose output
        
    Returns:
        Tuple of (merged_data, transform, crs)
    """
    if not RASTERIO_AVAILABLE or not NUMPY_AVAILABLE:
        raise ImportError("rasterio and numpy are required for raster merging")
        
    if not file_paths:
        raise ValueError("No input files provided")
    
    if verbose:
        print(f"Processing {len(file_paths)} files...")
    
    # First, we need to determine the target grid
    # We'll use the first file to establish the grid, then adjust if needed
    reprojected_arrays = []
    transforms = []
    bounds_list = []
    
    target_crs_obj = CRS.from_string(target_crs)
    
    # Process each file
    for i, file_path in enumerate(file_paths):
        if verbose:
            print(f"  Processing {os.path.basename(file_path)} ({i+1}/{len(file_paths)})")
        
        try:
            # Reproject the raster
            data, transform, width, height = reproject_raster(
                file_path, target_crs, target_resolution, resampling_method
            )
            
            # Process values (nan -> 0, 255 -> 1)
            data = process_raster_values(data)
            
            reprojected_arrays.append(data)
            transforms.append(transform)
            
            # Calculate bounds
            from rasterio.transform import array_bounds
            bounds = array_bounds(height, width, transform)
            bounds_list.append(bounds)
            
        except Exception as e:
            if verbose:
                print(f"    Warning: Failed to process {file_path}: {e}")
            continue
    
    if not reprojected_arrays:
        raise ValueError("No files could be processed successfully")
    
    # Calculate overall bounds
    min_x = min(bounds[0] for bounds in bounds_list)
    min_y = min(bounds[1] for bounds in bounds_list)
    max_x = max(bounds[2] for bounds in bounds_list)
    max_y = max(bounds[3] for bounds in bounds_list)
    
    # Determine grid resolution from first transform
    pixel_size_x = abs(transforms[0].a)
    pixel_size_y = abs(transforms[0].e)
    
    # Calculate output dimensions
    output_width = int(np.ceil((max_x - min_x) / pixel_size_x))
    output_height = int(np.ceil((max_y - min_y) / pixel_size_y))
    
    # Create output transform
    output_transform = Affine(
        pixel_size_x, 0.0, min_x,
        0.0, -pixel_size_y, max_y
    )
    
    # Initialize output array
    if merge_method == 'mean':
        output_data = np.zeros((output_height, output_width), dtype=np.float32)
        count_data = np.zeros((output_height, output_width), dtype=np.uint16)
    else:
        output_data = np.zeros((output_height, output_width), dtype=np.uint8)
    
    # Merge arrays
    for i, (data, transform) in enumerate(zip(reprojected_arrays, transforms)):
        # Calculate offset in the output grid
        offset_x = int((transform.c - min_x) / pixel_size_x)
        offset_y = int((max_y - transform.f) / pixel_size_y)
        
        # Calculate slice bounds
        end_x = offset_x + data.shape[1]
        end_y = offset_y + data.shape[0]
        
        # Ensure we don't go out of bounds
        end_x = min(end_x, output_width)
        end_y = min(end_y, output_height)
        data_end_x = end_x - offset_x
        data_end_y = end_y - offset_y
        
        if offset_x >= 0 and offset_y >= 0 and end_x > offset_x and end_y > offset_y:
            # Extract the relevant portion of the data
            data_slice = data[:data_end_y, :data_end_x]
            
            if merge_method == 'first':
                # Only fill where output is currently 0 (empty)
                mask = output_data[offset_y:end_y, offset_x:end_x] == 0
                output_data[offset_y:end_y, offset_x:end_x][mask] = data_slice[mask]
            elif merge_method == 'last':
                # Always overwrite
                output_data[offset_y:end_y, offset_x:end_x] = data_slice
            elif merge_method == 'max':
                # Take maximum value
                output_data[offset_y:end_y, offset_x:end_x] = np.maximum(
                    output_data[offset_y:end_y, offset_x:end_x], data_slice
                )
            elif merge_method == 'min':
                # Take minimum value (but ignore 0 values in output for meaningful min)
                current_slice = output_data[offset_y:end_y, offset_x:end_x]
                mask = current_slice == 0
                current_slice[mask] = data_slice[mask]
                current_slice[~mask] = np.minimum(current_slice[~mask], data_slice[~mask])
                output_data[offset_y:end_y, offset_x:end_x] = current_slice
            elif merge_method == 'mean':
                # Accumulate for mean calculation
                output_data[offset_y:end_y, offset_x:end_x] += data_slice.astype(np.float32)
                count_data[offset_y:end_y, offset_x:end_x] += (data_slice > 0).astype(np.uint16)
    
    # Finalize mean calculation if needed
    if merge_method == 'mean':
        mask = count_data > 0
        output_data[mask] = output_data[mask] / count_data[mask]
        output_data = output_data.astype(np.uint8)
    
    return output_data, output_transform, target_crs_obj


def save_merged_raster(data, transform, crs, 
                      output_path: str, compress: str = 'zstd', 
                      tiled: bool = True, tile_size: int = 512) -> None:
    """
    Save merged raster to file.
    
    Args:
        data: Raster data array
        transform: Affine transform
        crs: Coordinate reference system
        output_path: Output file path
        compress: Compression method
        tiled: Whether to create tiled output
        tile_size: Tile size for tiled output
    """
    if not RASTERIO_AVAILABLE:
        raise ImportError("rasterio is required for saving raster files")
    # Prepare creation options
    creation_options = {
        'driver': 'GTiff',
        'height': data.shape[0],
        'width': data.shape[1],
        'count': 1,
        'dtype': data.dtype,
        'crs': crs,
        'transform': transform,
        'compress': compress,
        'nodata': 0
    }
    
    # Add tiling options
    if tiled:
        creation_options.update({
            'tiled': True,
            'blockxsize': tile_size,
            'blockysize': tile_size
        })
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write the file
    with rasterio.open(output_path, 'w', **creation_options) as dst:
        dst.write(data, 1)


def main():
    """Main function to orchestrate the merging process."""
    args = parse_args()
    
    # Check if required dependencies are available
    if not RASTERIO_AVAILABLE or not NUMPY_AVAILABLE:
        print("Error: Required libraries are not available.")
        print()
        print("Please install the required dependencies:")
        print("  pip install rasterio numpy")
        print("or")
        print("  conda install -c conda-forge rasterio numpy")
        print()
        print("The script requires the following Python packages:")
        print("  - rasterio (for geospatial data processing)")
        print("  - numpy (for array operations)")
        return 1
    
    if args.verbose:
        print("Flood Detection Results Merger")
        print("=" * 50)
        print(f"Input directory: {args.input_dir}")
        print(f"Output file: {args.output}")
        print(f"Target CRS: {args.target_crs}")
        print(f"Merge method: {args.merge_method}")
        print(f"Compression: {args.compress}")
        print(f"Tiled: {args.tiled}")
        if args.tiled:
            print(f"Tile size: {args.tile_size}")
        print()
    
    # Find all .tif files
    tif_files = find_tif_files(args.input_dir)
    
    if not tif_files:
        print(f"Error: No .tif files found in {args.input_dir}")
        return 1
    
    if args.verbose:
        print(f"Found {len(tif_files)} .tif files:")
        for f in tif_files[:10]:  # Show first 10
            print(f"  {f}")
        if len(tif_files) > 10:
            print(f"  ... and {len(tif_files) - 10} more")
        print()
    
    try:
        # Get resampling method
        resampling_method = get_resampling_method(args.resampling)
        
        # Merge rasters
        if args.verbose:
            print("Merging rasters...")
        
        merged_data, transform, crs = merge_rasters(
            tif_files, 
            args.target_crs, 
            args.resolution,
            resampling_method,
            args.merge_method,
            args.verbose
        )
        
        if args.verbose:
            print(f"Merged raster shape: {merged_data.shape}")
            print(f"Data type: {merged_data.dtype}")
            print(f"Value range: {merged_data.min()} - {merged_data.max()}")
            print(f"Non-zero pixels: {np.count_nonzero(merged_data)}")
            print()
        
        # Save result
        if args.verbose:
            print(f"Saving merged raster to {args.output}...")
        
        save_merged_raster(
            merged_data, 
            transform, 
            crs, 
            args.output, 
            args.compress, 
            args.tiled, 
            args.tile_size
        )
        
        if args.verbose:
            print("Done!")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())