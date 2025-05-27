import argparse
import numpy as np
import rasterio
from pathlib import Path

def get_unique_values_from_tifs(folder_path):
    # Use a set for efficiently storing unique values
    all_unique_values = set()
    
    # Get all tif files in the folder
    tif_files = list(Path(folder_path).glob('*.tif'))
    print(f"Found {len(tif_files)} TIF files in the folder")
    
    # Process each file
    for i, tif_path in enumerate(tif_files):
        print(f"Processing file {i+1}/{len(tif_files)}: {tif_path.name}")
        
        try:
            with rasterio.open(tif_path) as src:
                # Read band 1 data in blocks to manage memory
                # Get the height and width of the raster
                height = src.height
                width = src.width
                
                # Define block size for reading (adjust based on memory constraints)
                block_size = 1024  # Rows to read at once
                
                # Process the raster in blocks
                for y in range(0, height, block_size):
                    # Adjust block height for the last block
                    actual_block_size = min(block_size, height - y)
                    
                    # Read a block of data
                    block_data = src.read(1, window=((y, y + actual_block_size), (0, width)))
                    
                    # Update set with unique values from this block
                    # Convert block to 1D array for efficiency
                    unique_in_block = np.unique(block_data)
                    all_unique_values.update(unique_in_block)
                    
                    # Optional: Print progress for large files
                    if height > block_size * 10 and (y + block_size) % (block_size * 10) == 0:
                        print(f"  Progress: {y + block_size}/{height} rows processed")
                
        except Exception as e:
            print(f"Error processing {tif_path}: {e}")
    
    # Convert set to sorted list for output
    unique_values_list = sorted(list(all_unique_values))
    
    return unique_values_list

# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get unique values from all TIF files in a folder")
    # Specify the folder containing the TIF files
    parser.add_argument("--folder_path", type=str, help="Path to the folder containing TIF files")
    args = parser.parse_args()
    
    # Get unique values
    unique_values = get_unique_values_from_tifs(args.folder_path)
    # python scripts/uniqueValue.py --folder_path E:\2025\test\2020\LR_label
    # 1990: [11, 12, 21, 22, 23, 24, 31, 32, 33, 41, 42, 43, 45, 46, 51, 52, 53, 61, 64, 65]
    # 2000: [11, 12, 21, 22, 23, 24, 31, 32, 33, 41, 42, 43, 45, 46, 51, 52, 53, 61, 64, 65]
    # 2010: [11, 12, 21, 22, 23, 24, 31, 32, 33, 41, 42, 43, 45, 46, 51, 52, 53, 61, 64, 65]
    # 2015: [11, 12, 21, 22, 23, 24, 31, 32, 33, 41, 42, 43, 45, 46, 51, 52, 53, 61, 64, 65]
    # 2018: [11, 12, 21, 22, 23, 24, 31, 32, 33, 41, 42, 43, 45, 46, 51, 52, 53, 61, 64, 65]
    # 2020: [11, 12, 21, 22, 23, 24, 31, 32, 33, 41, 42, 43, 45, 46, 51, 52, 53, 61, 64, 65]
    print("\nResults:")
    print(f"Number of unique values across all rasters: {len(unique_values)}")
    print(f"Unique values: {unique_values}")