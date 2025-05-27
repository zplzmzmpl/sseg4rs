import argparse
import numpy as np
import rasterio
from pathlib import Path

def calculate_image_stats(folder_path):
    # Lists to store means and stds for each image
    all_means = []
    all_stds = []
    
    # Get all tif files in the folder
    tif_files = list(Path(folder_path).glob('*.tif'))
    print(f"Found {len(tif_files)} TIF files in the folder")
    
    # Process each file
    for i, tif_path in enumerate(tif_files):
        print(f"Processing file {i+1}/{len(tif_files)}: {tif_path.name}")
        
        try:
            with rasterio.open(tif_path) as src:
                # Check if the image has 4 bands
                if src.count != 4:
                    print(f"  Warning: {tif_path.name} does not have 4 bands. Skipping.")
                    continue
                
                # Initialize arrays to store band stats
                image_means = []
                image_stds = []
                
                # Process each band
                for band in range(1, 5):  # Bands are 1-indexed in rasterio
                    # Define block size for reading (adjust based on memory constraints)
                    block_size = 1024  # Rows to read at once
                    height = src.height
                    width = src.width
                    
                    # Use Welford's online algorithm for calculating mean and variance
                    # This allows processing large rasters without loading entire band into memory
                    n = 0
                    mean = 0.0
                    M2 = 0.0  # Sum of squared differences from the mean
                    
                    # Process the raster in blocks
                    for y in range(0, height, block_size):
                        # Adjust block height for the last block
                        actual_block_size = min(block_size, height - y)
                        
                        # Read a block of data
                        block_data = src.read(band, window=((y, y + actual_block_size), (0, width)))
                        
                        # Flatten the block to process all pixels
                        pixels = block_data.flatten()
                        
                        # Skip nodata values if present
                        if src.nodata is not None:
                            pixels = pixels[pixels != src.nodata]
                        
                        # Update online statistics
                        for pixel in pixels:
                            n += 1
                            delta = pixel - mean
                            mean += delta / n
                            delta2 = pixel - mean
                            M2 += delta * delta2
                    
                    # Calculate final statistics
                    if n > 1:
                        variance = M2 / n
                        std_dev = np.sqrt(variance)
                    else:
                        mean = np.nan
                        std_dev = np.nan
                    
                    image_means.append(mean)
                    image_stds.append(std_dev)
                
                # Add this image's stats to the overall lists
                all_means.append(image_means)
                all_stds.append(image_stds)
                
                print(f"  Image stats - Means: {[round(m, 2) for m in image_means]}, STDs: {[round(s, 2) for s in image_stds]}")
                
        except Exception as e:
            print(f"Error processing {tif_path}: {e}")
    
    # Convert to numpy arrays for easier calculation
    all_means = np.array(all_means)
    all_stds = np.array(all_stds)
    
    # Calculate average means and stds across all images
    if len(all_means) > 0:
        avg_means = np.mean(all_means, axis=0)
        avg_stds = np.mean(all_stds, axis=0)
        
        # Round to 2 decimal places for cleaner output
        avg_means = np.round(avg_means, 2)
        avg_stds = np.round(avg_stds, 2)
    else:
        avg_means = np.array([])
        avg_stds = np.array([])
    
    return avg_means, avg_stds

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate average means and standard deviations of 4-band TIF images")
    parser.add_argument("--folder_path", type=str, help="Path to the folder containing 4-band TIF images")
    arg = parser.parse_args()
    # Specify the folder containing the TIF files
    # folder_path = r"D:\FinalDesignData\BigBayDataset\clips\HR_img"
    # python scripts\means_stds.py --folder_path E:\2025\code\Paraformer\dataset\2018clips_01_overlap_6class\HR_img
    
    # Calculate statistics
    avg_means, avg_stds = calculate_image_stats(arg.folder_path)
    
    # Print results
    if len(avg_means) > 0:
        print("\nResults:")
        print(f"IMAGE_MEANS : {avg_means.tolist()}")
        print(f"IMAGE_STDS : {avg_stds.tolist()}")
    else:
        print("\nNo valid 4-band TIF images were processed.")