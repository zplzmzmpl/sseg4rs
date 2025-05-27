import rasterio
from rasterio.windows import from_bounds
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
import argparse

def mosaic_tiles(tile_dir, output_path, overlap_behavior='max'):
    """
    Mosaic raster tiles into a single output raster, handling overlaps based on the specified behavior.
    Works with both single-band and multi-band images.

    Parameters:
    - tile_dir (str): Directory containing the input tile files (*.tif).
    - output_path (str): Path for the output mosaic raster file.
    - overlap_behavior (str): Strategy for handling overlaps ('max', 'last', or 'mean').
    """
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # Collect all tile files
    tile_files = list(Path(tile_dir).glob("*.tif"))
    if not tile_files:
        raise ValueError("No tiles found in the specified directory")

    # Stage 1: Collect metadata and verify consistency across tiles
    with rasterio.open(tile_files[0]) as ref:
        base_profile = ref.profile
        crs = ref.crs
        transform = ref.transform
        dtypes = ref.dtypes
        nodata = ref.nodata
        count = ref.count  # Number of bands
        pixel_size = (transform.a, abs(transform.e))  # (width, height) in geographic units

    print(f"Found {count} bands in the reference tile")
    print("Verifying tile consistency...")
    for tf in tqdm(tile_files, desc="Checking tiles"):
        with rasterio.open(tf) as src:
            if not (src.crs == crs and 
                    np.allclose([src.transform.a, src.transform.e], 
                               [transform.a, transform.e], atol=1e-9) and
                    src.count == count):
                raise ValueError(f"Metadata inconsistency in tile: {tf.name}")

    # Stage 2: Calculate the overall geographic extent of the mosaic
    print("Calculating mosaic range...")
    global_bounds = []
    for tf in tqdm(tile_files, desc="Collecting range"):
        with rasterio.open(tf) as src:
            global_bounds.append(src.bounds)
    
    all_lefts = [b[0] for b in global_bounds]
    all_tops = [b[3] for b in global_bounds]
    all_rights = [b[2] for b in global_bounds]
    all_bottoms = [b[1] for b in global_bounds]
    
    min_x, max_x = min(all_lefts), max(all_rights)
    min_y, max_y = min(all_bottoms), max(all_tops)

    # Calculate output dimensions in pixels
    output_width = int(np.ceil((max_x - min_x) / pixel_size[0]))
    output_height = int(np.ceil((max_y - min_y) / pixel_size[1]))

    # Define the output transform (top-left corner at (min_x, max_y))
    output_transform = rasterio.Affine(
        pixel_size[0], 0, min_x,
        0, -pixel_size[1], max_y
    )
    
    # Update the profile for the output raster
    profile = base_profile.copy()
    profile.update({
        'width': output_width,
        'height': output_height,
        'transform': output_transform,
        'count': count  # Set the band count
    })

    # Stage 3: Process tiles and write the mosaic
    print("Starting mosaic processing...")
    with rasterio.open(output_path, 'w', **profile) as dst:
        # Process each band
        for band_idx in range(count):
            # Initialize the output array - FIX: Handle None nodata values
            if nodata is None:
                # If nodata is None, initialize with zeros
                output_data = np.zeros((output_height, output_width), dtype=dtypes[band_idx])
                # Create a mask to track which pixels have been written
                coverage_mask = np.zeros((output_height, output_width), dtype=bool)
            else:
                # Initialize with nodata value
                output_data = np.full((output_height, output_width), nodata, dtype=dtypes[band_idx])
                coverage_mask = np.full((output_height, output_width), False, dtype=bool)
            
            print(f"Processing band {band_idx + 1}/{count}...")
            # Process each tile
            for tf in tqdm(tile_files, desc=f"Processing tiles for band {band_idx + 1}"):
                with rasterio.open(tf) as src:
                    # Calculate the overlapping geographic bounds
                    overlap_left = max(src.bounds[0], min_x)
                    overlap_right = min(src.bounds[2], min_x + output_width * pixel_size[0])
                    overlap_bottom = max(src.bounds[1], max_y - output_height * pixel_size[1])
                    overlap_top = min(src.bounds[3], max_y)
                    
                    # Check if there is an overlap
                    if overlap_left < overlap_right and overlap_bottom < overlap_top:
                        # Calculate windows for the tile and output based on overlapping bounds
                        tile_window = from_bounds(overlap_left, overlap_bottom, overlap_right, overlap_top, 
                                                transform=src.transform)
                        tile_window = tile_window.round_offsets().round_lengths()
                        
                        output_window = from_bounds(overlap_left, overlap_bottom, overlap_right, overlap_top, 
                                                  transform=output_transform)
                        output_window = output_window.round_offsets().round_lengths()
                        
                        # Read data from the tile (band_idx + 1 because rasterio bands are 1-indexed)
                        tile_data = src.read(band_idx + 1, window=tile_window)
                        
                        # Extract the corresponding region from the output array
                        current_data = output_data[output_window.row_off:output_window.row_off + output_window.height,
                                                 output_window.col_off:output_window.col_off + output_window.width]
                        
                        current_mask = coverage_mask[output_window.row_off:output_window.row_off + output_window.height,
                                                   output_window.col_off:output_window.col_off + output_window.width]
                        
                        # Verify that the shapes match (they should, given aligned rasters)
                        if tile_data.shape != current_data.shape:
                            print(f"Warning: Shape mismatch for tile {tf.name}: {tile_data.shape} vs {current_data.shape}")
                            # Adjust to the smaller size to prevent broadcasting errors
                            min_height = min(tile_data.shape[0], current_data.shape[0])
                            min_width = min(tile_data.shape[1], current_data.shape[1])
                            tile_data = tile_data[:min_height, :min_width]
                            current_data = current_data[:min_height, :min_width]
                            current_mask = current_mask[:min_height, :min_width]
                        
                        # Apply overlap handling logic
                        if nodata is None:
                            # When nodata is None, we use the coverage mask
                            tile_mask = np.ones_like(tile_data, dtype=bool)  # All pixels are valid
                            
                            if overlap_behavior == 'max':
                                # For previously uncovered areas, always use the tile data
                                # For covered areas, use the maximum value
                                update_mask = (~current_mask) | (tile_data > current_data)
                            elif overlap_behavior == 'mean' and np.any(current_mask):
                                # Mean strategy - only applicable for overlaps
                                overlap_mask = current_mask
                                # Calculate mean for overlapping areas
                                temp_data = current_data.copy()
                                # Average the values (current + new) / 2
                                temp_data[overlap_mask] = (current_data[overlap_mask] + tile_data[overlap_mask]) / 2
                                # Use tile data for non-overlapping areas
                                temp_data[~overlap_mask] = tile_data[~overlap_mask]
                                current_data = temp_data
                                update_mask = np.ones_like(tile_data, dtype=bool)  # Update all
                            elif overlap_behavior == 'last':
                                # Last strategy - always use the new tile
                                update_mask = np.ones_like(tile_data, dtype=bool)  # Update all
                            else:
                                # Default to 'max' behavior
                                update_mask = (~current_mask) | (tile_data > current_data)
                        else:
                            # Traditional approach with nodata values
                            tile_mask = tile_data != nodata
                            
                            if overlap_behavior == 'max':
                                update_mask = tile_mask & (
                                    (tile_data > current_data) | 
                                    (current_data == nodata)
                                )
                            elif overlap_behavior == 'mean' and np.any(current_data != nodata):
                                overlap_mask = (current_data != nodata) & tile_mask
                                temp_data = current_data.copy()
                                # Average the values where both the current and tile data are valid
                                temp_data[overlap_mask] = (current_data[overlap_mask] + tile_data[overlap_mask]) / 2
                                # Use tile data where current is nodata
                                nodata_mask = (current_data == nodata) & tile_mask
                                temp_data[nodata_mask] = tile_data[nodata_mask]
                                current_data = temp_data
                                update_mask = overlap_mask | nodata_mask
                            elif overlap_behavior == 'last':
                                update_mask = tile_mask
                            else:
                                update_mask = tile_mask & ((tile_data > current_data) | (current_data == nodata))
                        
                        # Merge the tile data into the output array
                        merged = np.where(update_mask, tile_data, current_data)
                        output_data[output_window.row_off:output_window.row_off + output_window.height,
                                  output_window.col_off:output_window.col_off + output_window.width] = merged
                        
                        # Update coverage mask
                        if nodata is None:
                            coverage_mask[output_window.row_off:output_window.row_off + output_window.height,
                                        output_window.col_off:output_window.col_off + output_window.width] = True
            
            # Write the band data to the output file
            dst.write(output_data.astype(dtypes[band_idx]), band_idx + 1)

    print(f"Mosaic completed! Output file saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="tile mosaic tool")
    parser.add_argument("--tile_dir", required=True, help="Directory containing tile files")
    parser.add_argument("--output_path", required=True, help="Path for the output mosaic file")
    parser.add_argument("--overlap", choices=['max', 'last', 'mean'], default='max',
                       help="Overlap processing strategy: 'max' takes maximum value, 'last' uses last tile, 'mean' averages values")
    
    args = parser.parse_args()

    # python scripts\mosaic-tiles_v2.py --tile_dir E:\2025\code\Paraformer\save\prediction\01ov_20cls_light_2020 --output_path D:\FinalDesignData\predict\01ov_20class_light_2020\01ov_20class_light_2020.tif --overlap last
    
    mosaic_tiles(
        tile_dir=args.tile_dir,
        output_path=args.output_path,
        overlap_behavior=args.overlap
    )