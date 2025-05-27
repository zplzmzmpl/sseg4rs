import rasterio
from rasterio.windows import Window
import numpy as np
import os
import csv
from pathlib import Path
import multiprocessing as mp
from functools import partial
import gc
import time
import warnings
import argparse
from tqdm import tqdm

def parse_nodata(nodata_str, band_count):
    """
    Parse a NoData string into a tuple matching the raster's band count.
    
    Args:
        nodata_str (str): String of NoData values (e.g., '256' or '256,256,256,256')
        band_count (int): Number of bands in the raster
    
    Returns:
        tuple: Parsed NoData values
    
    Raises:
        ValueError: If the number of values doesn't match band_count or is invalid
    """
    if nodata_str is None:
        return None
    try:
        nodata_vals = [float(val) if '.' in val else int(val) for val in nodata_str.split(',')]
        if len(nodata_vals) == 1:
            return tuple(nodata_vals * band_count)
        elif len(nodata_vals) == band_count:
            return tuple(nodata_vals)
        else:
            raise ValueError(f"NoData values must be a single value or match the number of bands ({band_count})")
    except ValueError as e:
        raise ValueError(f"Invalid NoData value: {e}")

def pad_to_size(data, target_height, target_width):
    """
    Pad data array to the target size by repeating edge values.
    
    Args:
        data (numpy.ndarray): Input data with shape (bands, height, width)
        target_height (int): Target height
        target_width (int): Target width
    
    Returns:
        numpy.ndarray: Padded data array with shape (bands, target_height, target_width)
    """
    bands, current_height, current_width = data.shape
    
    # If already the correct size, return as is
    if current_height == target_height and current_width == target_width:
        return data
    
    # Create output array
    padded = np.zeros((bands, target_height, target_width), dtype=data.dtype)
    
    # Copy existing data
    padded[:, :current_height, :current_width] = data
    
    # Pad height if needed
    if current_height < target_height:
        # Repeat the last row for remaining rows
        for i in range(current_height, target_height):
            padded[:, i, :current_width] = data[:, current_height-1, :current_width]
    
    # Pad width if needed
    if current_width < target_width:
        # Repeat the last column for remaining columns
        for i in range(current_width, target_width):
            padded[:, :target_height, i] = padded[:, :target_height, current_width-1]
    
    return padded

def process_dual_tile(task, label_nodata=None, img_nodata=None, suffix=None):
    """
    Process a single tile for dual-mode cropping - synchronized label and image tiles
    
    Args:
        task (tuple): Tile processing parameters
        label_nodata (tuple, optional): Custom NoData values for label
        img_nodata (tuple, optional): Custom NoData values for image
    
    Returns:
        dict: Tile metadata or None if the tile is invalid
    """
    x, y, tile_size, step, label_src_path, img_src_path, output_dir = task
    
    try:
        # Create directories for HR images and LR labels
        hr_img_dir = os.path.join(output_dir, "HR_img")
        lr_label_dir = os.path.join(output_dir, "LR_label")
        Path(hr_img_dir).mkdir(exist_ok=True, parents=True)
        Path(lr_label_dir).mkdir(exist_ok=True, parents=True)
        
        # Process label tile
        with rasterio.open(label_src_path) as label_src:
            height, width = label_src.height, label_src.width
            
            # Adjust label window at edges
            label_window = Window(x, y, tile_size, tile_size)
            actual_width = min(tile_size, width - x)
            actual_height = min(tile_size, height - y)
            label_window = Window(x, y, actual_width, actual_height)
            
            # Skip if tile is too small
            if label_window.width < tile_size * 0.9 or label_window.height < tile_size * 0.9:
                return None
            
            # Read label tile
            label_tile = label_src.read(window=label_window)
            
            # Check for NoData - use custom or raster's nodata
            if label_nodata is not None:
                nodata_val = label_nodata[0]
            else:
                nodata_val = label_src.nodata
            
            # Reject tile with ANY nodata values
            if nodata_val is not None and np.any(label_tile == nodata_val):
                return None
            
            # Pad label tile to ensure it meets the target size
            label_tile = pad_to_size(label_tile, tile_size, tile_size)
            
            # Calculate geographic bounds of the original (unpadded) label tile
            # We use the original window bounds for finding the corresponding image area
            label_bounds = label_src.window_bounds(label_window)
        
        # Process corresponding image tile
        with rasterio.open(img_src_path) as img_src:
            # Calculate the image window that matches the label tile's geographic bounds
            img_window = rasterio.windows.from_bounds(
                *label_bounds, 
                transform=img_src.transform
            )
            
            # Ensure the window intersects with the image raster
            img_window = rasterio.windows.intersection(
                img_window, 
                Window(0, 0, img_src.width, img_src.height)
            )
            
            # Skip if there is no overlap (width or height <= 0)
            if img_window.width <= 0 or img_window.height <= 0:
                return None
            
            # Read image tile
            img_tile = img_src.read(window=img_window)
            
            # Check image NoData if specified
            if img_nodata is not None:
                img_nodata_mask = np.all(img_tile == np.array(img_nodata)[:, None, None], axis=0)
                if np.any(img_nodata_mask):
                    return None

            # Pad image tile to maintain aspect ratio and resolution
            target_img_height = int(tile_size * (img_window.height / label_window.height))
            target_img_width = int(tile_size * (img_window.width / label_window.width))
            
            # If the image tile is smaller than expected, pad it
            if img_tile.shape[1] < target_img_height or img_tile.shape[2] < target_img_width:
                img_tile = pad_to_size(img_tile, target_img_height, target_img_width)
        
        # Define tile ID based on label position
        tile_id = f"tile_{y//step}_{x//step}"
        
        # Define output file paths
        label_out = os.path.join(lr_label_dir, f"{suffix}{tile_id}_label.tif")
        img_out = os.path.join(hr_img_dir, f"{suffix}{tile_id}_image.tif")
        
        # Write label tile
        with rasterio.open(label_src_path) as label_src:
            label_profile = label_src.profile.copy()
            original_transform = label_src.window_transform(label_window)
            
            # Update profile for the padded tile
            label_profile.update({
                "height": tile_size,
                "width": tile_size,
                "transform": original_transform,
                "compress": "lzw",
                "tiled": True,
                "blockxsize": 256,
                "blockysize": 256
            })
            with rasterio.open(label_out, 'w', **label_profile) as dst:
                dst.write(label_tile)
        
        # Write image tile
        with rasterio.open(img_src_path) as img_src:
            img_profile = img_src.profile.copy()
            original_img_transform = img_src.window_transform(img_window)
            
            # Update profile for the padded image tile
            img_profile.update({
                "height": img_tile.shape[1],
                "width": img_tile.shape[2],
                "transform": original_img_transform,
                "compress": "lzw",
                "tiled": True,
                "blockxsize": 256,
                "blockysize": 256
            })
            with rasterio.open(img_out, 'w', **img_profile) as dst:
                dst.write(img_tile)
        
        # Clear memory
        gc.collect()
        
        # Return the file paths for CSV generation
        return {
            "image_fn": os.path.join("./HR_img" + "/" + f"{suffix}{tile_id}_image.tif"),
            "label_fn": os.path.join("./LR_label" + "/" + f"{suffix}{tile_id}_label.tif")
        }
    
    except Exception as e:
        print(f"Error processing tile at {x},{y}: {str(e)}")
        return None

def process_single_tile(task, user_nodata=None,suffix=None):
    """
    Process a single tile in single mode
    
    Args:
        task (tuple): Tile processing parameters
        user_nodata (tuple, optional): Custom NoData values
    
    Returns:
        dict: Tile metadata or None if the tile is invalid
    """
    x, y, tile_size, step, src_path, output_dir = task
    
    try:
        # Create output directory
        output_subdir = os.path.join(output_dir, "images")
        Path(output_subdir).mkdir(exist_ok=True, parents=True)
        
        # Define the tile window
        with rasterio.open(src_path) as src:
            height, width = src.height, src.width
            
            # Calculate actual window size (might be smaller at edges)
            actual_width = min(tile_size, width - x)
            actual_height = min(tile_size, height - y)
            window = Window(x, y, actual_width, actual_height)
            
            # Skip if tile is too small
            if window.width < tile_size * 0.9 or window.height < tile_size * 0.9:
                return None
            
            # Read tile data
            tile_data = src.read(window=window)
            
            # Use custom NoData if provided, else use raster's NoData
            if user_nodata is not None:
                nodatavals = user_nodata
            else:
                nodatavals = src.nodatavals if src.nodatavals and all(v is not None for v in src.nodatavals) else None
            
            # Check for NoData
            if nodatavals:
                nodata_array = np.array(nodatavals)[:, None, None]
                nodata_mask = np.all(tile_data == nodata_array, axis=0)
                if np.any(nodata_mask):
                    return None
            
            # Pad tile to exact size if needed
            if tile_data.shape[1] < tile_size or tile_data.shape[2] < tile_size:
                tile_data = pad_to_size(tile_data, tile_size, tile_size)
            
            # Get original window bounds
            original_transform = src.window_transform(window)
            
        # Generate tile ID
        tile_id = f"tile_{y//step}_{x//step}"
        
        # Output path
        output_path = os.path.join(output_subdir, f"{suffix}{tile_id}_label.tif")
        
        # Write tile
        with rasterio.open(src_path) as src:
            profile = src.profile.copy()
            profile.update({
                "height": tile_size,
                "width": tile_size,
                "transform": original_transform,
                "compress": "lzw",
                "tiled": True,
                "blockxsize": 256,
                "blockysize": 256
            })
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(tile_data)
        
        gc.collect()
        return {"image_fn": os.path.join("./images"  + "/" +  f"{suffix}{tile_id}_label.tif")}
    
    except Exception as e:
        print(f"Error processing tile at {x},{y}: {str(e)}")
        return None

def sync_crop(src_path, output_dir, tile_size=2000, overlap=0.1, 
             num_workers=None, suffix=None, single_mode=False, label_path=None, 
             label_nodata=None, img_nodata=None):
    """
    Crop raster images with flexible processing modes
    
    Args:
        src_path (str): Path to the source image
        output_dir (str): Directory for output tiles
        tile_size (int): Size of each tile
        overlap (float): Overlap fraction between tiles
        num_workers (int, optional): Number of worker processes
        single_mode (bool): Process only the source image
        label_path (str, optional): Path to the label image (dual mode)
        label_nodata (str, optional): Custom NoData value for label
        img_nodata (str, optional): Custom NoData value for image
    """
    start_time = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"There are {mp.cpu_count()} CPUs available")
    
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)
    
    print(f"Starting tile generation with {num_workers} workers")
    print(f"Processing mode: {'Single' if single_mode else 'Dual'}")
    print(f"Target tile size: {tile_size}x{tile_size} (will pad if necessary)")
    
    warnings.filterwarnings("ignore", category=UserWarning)
    
    with rasterio.Env(GDAL_CACHEMAX=512):
        # Single image mode
        if single_mode:
            with rasterio.open(src_path) as src:
                band_count = src.count
                # Parse custom NoData if provided
                if img_nodata is not None:
                    img_nodata = parse_nodata(img_nodata, band_count)
                
                step = int(tile_size * (1 - overlap))
                height, width = src.height, src.width
                nodata = src.nodata
                
                print(f"Image dimensions: {width}x{height}")
                print(f"NoData value: {img_nodata if img_nodata is not None else nodata}")
                
                tasks = []
                for y in range(0, height, step):
                    for x in range(0, width, step):
                        tasks.append((x, y, tile_size, step, src_path, output_dir))
            
            print(f"Generated {len(tasks)} potential tile locations")
            
            # Process tiles in parallel
            with mp.Pool(processes=num_workers) as pool:
                process_func = partial(process_single_tile, user_nodata=img_nodata, suffix=suffix)
                results = list(tqdm(pool.imap_unordered(process_func, tasks), total=len(tasks)))
        
        # Dual image mode (synchronized cropping)
        else:
            if not label_path:
                raise ValueError("Label path is required in dual processing mode")
            
            # Parse NoData values if provided
            with rasterio.open(label_path) as label_src:
                label_band_count = label_src.count
                label_nodata_val = parse_nodata(label_nodata, label_band_count) if label_nodata is not None else (label_src.nodata,) * label_band_count if label_src.nodata is not None else None
            with rasterio.open(src_path) as img_src:
                img_band_count = img_src.count
                img_nodata_val = parse_nodata(img_nodata, img_band_count) if img_nodata is not None else (img_src.nodata,) * img_band_count if img_src.nodata is not None else None
                # Validate coordinate system and transformation
                with rasterio.open(label_path) as label_src:
                    if label_src.crs != img_src.crs:
                        raise ValueError("Coordinate system does not match")
                    
                    if not np.allclose([label_src.transform.a, label_src.transform.e],
                                       [img_src.transform.a, img_src.transform.e],
                                       rtol=0.001):
                        raise ValueError("Spatial transformation parameters inconsistent")
                
                step = int(tile_size * (1 - overlap))
                height, width = label_src.height, label_src.width
                
                print(f"Image dimensions: {width}x{height}")
                print(f"Label NoData: {label_nodata_val}")
                print(f"Image NoData: {img_nodata_val}")
                
                tasks = []
                for y in range(0, height, step):
                    for x in range(0, width, step):
                        tasks.append((x, y, tile_size, step, label_path, src_path, output_dir))
            
            print(f"Generated {len(tasks)} potential tile locations")
            
            # Process tiles in parallel
            with mp.Pool(processes=num_workers) as pool:
                process_func = partial(process_dual_tile, 
                                      label_nodata=label_nodata_val, 
                                      img_nodata=img_nodata_val,
                                      suffix=suffix)
                results = list(tqdm(pool.imap_unordered(process_func, tasks), total=len(tasks)))
        
        # Filter valid results
        valid_tiles = [r for r in results if r is not None]
        count = len(valid_tiles)
        
        # Generate CSV
        csv_path = os.path.join(output_dir, "tile_records.csv")
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['image_fn'] if single_mode else ['image_fn', 'label_fn']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for tile in valid_tiles:
                writer.writerow(tile)
        
        elapsed_time = time.time() - start_time
        print(f"Generated {count} valid tiles in {elapsed_time:.2f} seconds")
        print(f"CSV file created at: {csv_path}")
        return valid_tiles

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced raster tile cropping tool")
    parser.add_argument("--src_path", required=True, help="Path to source raster file")
    parser.add_argument("--label_path", help="Path to label raster file (for dual mode)")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--tile_size", type=int, default=2000, help="Size of each tile")
    parser.add_argument("--overlap", type=float, default=0.1, help="Overlap fraction")
    parser.add_argument("--workers", type=int, default=None, help="Number of workers")
    parser.add_argument("--suffix", type=str, default=None, help="Suffix for output files")
    parser.add_argument("--single_mode", action="store_true", 
                       help="Enable single image processing mode")
    parser.add_argument("--label_nodata", type=str, default=None, 
                       help="Custom NoData value(s) for label (e.g., '256' or '256,256,256,256')")
    parser.add_argument("--img_nodata", type=str, default=None, 
                       help="Custom NoData value(s) for image (e.g., '0' or '0,0,0,0')")
    
    args = parser.parse_args()
    
    if not args.single_mode and not args.label_path:
        raise ValueError("Label path is required in dual processing mode")
    
    # Example usage:
    # 1. Single mode (single image):
    # python scripts\clips_v7.py --src_path BigBayDataset\landsat8\ld2015_rgb.tif --output_dir BigBayDataset\landsat8\clips --single_mode --img_nodata 0,0,0 --tile_size 1024
    
    # 2. Dual mode (label and image):
    # python scripts\clips_v7.py --label_path BigBayDataset\lucc\ld2010.tif --src_path BigBayDataset\landsat8\BigBay_L5T1L2_RGB_2010_prj.tif --output_dir E:\2025\test\2010 --tile_size 1024 --suffix 2010
    
    sync_crop(
        src_path=args.src_path,
        output_dir=args.output_dir,
        tile_size=args.tile_size,
        overlap=args.overlap,
        num_workers=args.workers,
        suffix=args.suffix,
        single_mode=args.single_mode,
        label_path=args.label_path if not args.single_mode else None,
        label_nodata=args.label_nodata,
        img_nodata=args.img_nodata
    )