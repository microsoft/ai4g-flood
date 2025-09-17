#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import rasterio
import rasterio.io
from rasterio.io import MemoryFile
from rasterio.merge import merge as rio_merge
from rasterio.warp import Resampling
from tqdm import tqdm


def find_geotiffs(root: Path) -> List[Path]:
    # Expected layout: results/YYYY/MM/DD/*.tif
    # But we'll just recurse for robustness.
    return sorted(root.rglob("*.tif"))


def to_binary_memfile(src_path: Path) -> MemoryFile:
    """Open a source raster and convert its band 1 to a binary uint8 mask.

    Converts:
    - 255 -> 1
    - NaN/other -> 0

    Returns a MemoryFile with same geotransform/CRS.
    """
    with rasterio.open(src_path) as src:
        data = src.read(1, masked=True)  # masked array if nodata present

        # Build binary mask: 1 where value == 255, else 0
        # Works for float/uint inputs; NaNs won't equal 255 and thus map to 0.
        bin_arr = (data.filled(np.nan) == 255).astype(np.uint8)

        # Create an in-memory dataset with same transform/CRS/shape
        profile = src.profile.copy()
        profile.update(
            dtype=rasterio.uint8,
            count=1,
            nodata=0,  # we'll use 0 as nodata/"no flood"
            compress="DEFLATE",  # internal for the temp MemoryFile; final output will be ZSTD
            tiled=False,
        )

        memfile = MemoryFile()
        with memfile.open(**profile) as dst:
            dst.write(bin_arr, 1)
    return memfile


def merge_to_3857(mem_datasets: List[rasterio.io.DatasetReader]) -> Tuple[np.ndarray, rasterio.Affine]:
    """Merge a list of binary uint8 rasters to EPSG:3857 using nearest, preferring 1 over 0 (np.maximum)."""
    mosaic, out_transform = rio_merge(
        sources=mem_datasets,
        dst_crs="EPSG:3857",
        res=None,
        method=np.maximum,  # Prefer 1 where any overlap
        dst_nodata=0,
        resampling=Resampling.nearest,
    )
    # mosaic shape is (1, H, W) since count=1
    # Ensure dtype is uint8
    mosaic = mosaic.astype(np.uint8, copy=False)
    return mosaic, out_transform


def write_output(
    out_path: Path,
    mosaic: np.ndarray,
    transform: rasterio.Affine,
    crs: str = "EPSG:3857",
    blocksize: int = 512,
):
    height, width = mosaic.shape[1], mosaic.shape[2]
    profile = {
        "driver": "GTiff",
        "dtype": rasterio.uint8,
        "count": 1,
        "width": width,
        "height": height,
        "crs": crs,
        "transform": transform,
        "nodata": 0,
        "compress": "ZSTD",
        "zstd_level": 9,
        "tiled": True,
        "blockxsize": blocksize,
        "blockysize": blocksize,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mosaic[0], 1)


def main():
    parser = argparse.ArgumentParser(
        description="Merge flood mask GeoTIFFs."
    )
    parser.add_argument(
        "--input_root",
        type=str,
        required=True,
        help="Root directory containing results/YYYY/MM/DD/*.tif",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output GeoTIFF filename (e.g., merged_flood_3857.tif)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, limit to the first N input rasters (for testing).",
    )
    args = parser.parse_args()

    root = Path(args.input_root)
    out_path = Path(args.output)

    tifs = find_geotiffs(root)
    if args.limit and args.limit > 0:
        tifs = tifs[: args.limit]

    if not tifs:
        raise SystemExit(f"No GeoTIFFs found under {root}")

    print(f"Found {len(tifs)} GeoTIFF(s). Converting to binary masks...")
    memfiles: List[MemoryFile] = []
    datasets: List[rasterio.io.DatasetReader] = []
    try:
        for p in tqdm(tifs, unit="img"):
            mf = to_binary_memfile(p)
            memfiles.append(mf)
            datasets.append(mf.open())

        print("Merging to EPSG:3857...")
        mosaic, out_transform = merge_to_3857(datasets)

        print("Writing output (ZSTD, tiled)...")
        write_output(out_path, mosaic, out_transform, crs="EPSG:3857")

        print(f"Done. Wrote: {out_path}")
    finally:
        # Clean up in-memory datasets
        for ds in datasets:
            try:
                ds.close()
            except Exception:
                pass
        for mf in memfiles:
            try:
                mf.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
