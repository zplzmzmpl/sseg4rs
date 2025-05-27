import numpy as np
import rasterio
import argparse
import os
import glob
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Note: 'tqdm' package not found. Install with 'pip install tqdm' for progress bar support.")

# Define the label classes and their corresponding RGB colors
LABEL_CLASSES = [11, 12, 21, 22, 23, 24, 31, 32, 33, 41, 42, 43, 45, 46, 51, 52, 53, 61, 64, 65]
LABEL_CLASS_COLORMAP = {
    11: (0, 255, 0),      # Bright green
    12: (124, 252, 0),    # Lawn green
    21: (34, 139, 34),    # Forest green
    22: (50, 205, 50),    # Lime green
    23: (154, 205, 50),   # Yellow green
    24: (144, 238, 144),  # Light green
    31: (255, 255, 0),    # Yellow
    32: (255, 215, 0),    # Gold
    33: (255, 250, 205),  # Light yellow
    41: (0, 0, 255),      # Blue
    42: (135, 206, 235),  # Sky blue
    43: (173, 216, 230),  # Light blue
    45: (255, 165, 0),    # Orange
    46: (255, 140, 0),    # Dark orange
    51: (255, 0, 0),      # Red
    52: (255, 105, 180),  # Hot pink
    53: (255, 160, 122),  # Light salmon
    61: (255, 228, 196),  # Bisque
    64: (46, 139, 87),    # Sea green
    65: (139, 69, 19)     # Saddle brown
}
# LABEL_CLASS_COLORMAP = {
#     11: (0, 255, 0),
#     12: (0, 255, 0),
#     21: (34, 139, 34),
#     22: (34, 139, 34),
#     23: (34, 139, 34),
#     24: (34, 139, 34),
#     31: (255, 255, 0),
#     32: (255, 255, 0),
#     33: (255, 255, 0),
#     41: (0, 0, 255),
#     42: (0, 0, 255),
#     43: (0, 0, 255),
#     45: (0, 0, 255),
#     46: (0, 0, 255),
#     51: (255, 0, 0),
#     52: (255, 0, 0),
#     53: (255, 0, 0),
#     61: (255, 228, 196),
#     64: (255, 228, 196),
#     65: (255, 228, 196)
# }

def convert_classified_tif_to_rgb(input_tif_path, output_rgb_path, print_statistics=False, nodata_color=(0, 0, 0), unknown_color=(128, 128, 128)):
    """
    Convert a single-band classified TIF image to a 3-band RGB image based on a color mapping.
    
    Parameters:
    input_tif_path (str): Path to the input classified TIF file
    output_rgb_path (str): Path where the output RGB TIF will be saved
    print_statistics (bool): Whether to print statistics about the classified image
    nodata_color (tuple): RGB color to use for nodata values (default: black)
    unknown_color (tuple): RGB color to use for values not in LABEL_CLASS_COLORMAP (default: gray)
    """
    # Open the classified TIF file
    with rasterio.open(input_tif_path) as src:
        # Read the classified data
        classified_data = src.read(1)  # Assuming it's a single band
        
        # Get the nodata value from the source file
        nodata_value = src.nodata
        
        # Get the metadata to preserve projection and other information
        metadata = src.meta.copy()
        
        # Update metadata for RGB output (3 bands)
        metadata.update({
            'count': 3,
            'dtype': 'uint8',  # RGB values are 0-255
            'nodata': None     # RGB images typically don't use nodata value
        })
        
        # Initialize RGB array with the unknown_color (for pixels not in the colormap)
        # This sets a default color for values not in our label classes
        height, width = classified_data.shape
        rgb_data = np.ones((3, height, width), dtype=np.uint8)
        for i in range(3):
            rgb_data[i, :, :] = unknown_color[i]
        
        # First handle nodata values if present
        if nodata_value is not None:
            nodata_mask = (classified_data == nodata_value)
            for i in range(3):
                rgb_data[i][nodata_mask] = nodata_color[i]
        
        # Then map known classes to RGB colors
        for label_class, color in LABEL_CLASS_COLORMAP.items():
            # Create a mask for the current class
            mask = (classified_data == label_class)
            
            # Assign RGB values
            for i in range(3):  # R, G, B channels
                rgb_data[i][mask] = color[i]
        
        # Write the RGB image
        with rasterio.open(output_rgb_path, 'w', **metadata) as dst:
            dst.write(rgb_data)
        
        print(f"RGB image created and saved to {output_rgb_path}")
        
        # Print statistics if requested
        if print_statistics:
            unique_values = np.unique(classified_data)
            unmapped_values = [val for val in unique_values if val != nodata_value and val not in LABEL_CLASS_COLORMAP]
            
            print(f"Total unique values in classified image: {len(unique_values)}")
            print(f"Values in colormap: {len(LABEL_CLASS_COLORMAP)}")
            if nodata_value is not None:
                print(f"Nodata value: {nodata_value}")
            print(f"Unmapped values (colored as gray): {unmapped_values}")

def process_directory(input_dir, output_dir, print_statistics=False):
    """
    Process all TIF files in a directory and convert them to RGB.
    
    Parameters:
    input_dir (str): Path to the directory containing classified TIF files
    output_dir (str): Path to the directory where output RGB TIF files will be saved
    print_statistics (bool): Whether to print statistics about each classified image
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all TIF files in the input directory
    tif_files = glob.glob(os.path.join(input_dir, "*.tif")) + glob.glob(os.path.join(input_dir, "*.TIF"))
    
    if not tif_files:
        print(f"No TIF files found in {input_dir}")
        return
    
    print(f"Found {len(tif_files)} TIF files to process")
    
    # Process each TIF file with progress bar if tqdm is available
    if TQDM_AVAILABLE:
        for tif_file in tqdm(tif_files, desc="Processing TIF files", unit="file"):
            # Get the filename without the directory path
            input_filename = os.path.basename(tif_file)
            
            # Create the output path with the same filename
            output_path = os.path.join(output_dir, input_filename)
            
            # Only print filename if statistics are enabled, otherwise it's redundant with the progress bar
            if print_statistics:
                print(f"\nProcessing: {input_filename}")
            
            # Convert the file
            convert_classified_tif_to_rgb(tif_file, output_path, print_statistics)
    else:
        # Fallback to regular processing without progress bar
        for i, tif_file in enumerate(tif_files):
            # Get the filename without the directory path
            input_filename = os.path.basename(tif_file)
            
            # Create the output path with the same filename
            output_path = os.path.join(output_dir, input_filename)
            
            # print(f"Processing: {input_filename} ({i+1}/{len(tif_files)})")
            
            # Convert the file
            convert_classified_tif_to_rgb(tif_file, output_path, print_statistics)
    
    print(f"All TIF files processed. RGB results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert classified TIF to RGB TIF.")
    parser.add_argument("--batch", choices=["yes", "no"], default="no",
                       help="Process in batch mode (yes) or single file mode (no, default)")
    parser.add_argument("--input_dir", help="Path to the directory containing input classified TIF files")
    parser.add_argument("--output_dir", help="Path to the directory where output RGB TIF files will be saved")
    parser.add_argument("--input_tif", help="Path to a single input classified TIF file (for single file mode)")
    parser.add_argument("--output_rgb", help="Path to a single output RGB TIF file (for single file mode)")
    parser.add_argument("--print_statistics", choices=["yes", "no"], default="no",
                       help="Whether to print statistics about the classified images (default: no)")
    
    args = parser.parse_args()
    
    # Convert string arguments to boolean
    print_stats = args.print_statistics.lower() == "yes"
    batch_mode = args.batch.lower() == "yes"

    # python scripts/cls2rgb_v2.py --batch yes --input_dir BigBayDataset\landsat8\clips\20cls_gray_label --output_dir BigBayDataset\landsat8\clips\20cls_rgb_label --print_statistics no
    # python scripts/cls2rgb_v2.py --batch no --input_tif BigBayDataset\lucc\ld1980.tif --output_rgb E:\2025\test\ld1980_rgb.tif --print_statistics yes

    if batch_mode:
        # Batch processing mode
        if not (args.input_dir and args.output_dir):
            parser.error("In batch mode, both --input_dir and --output_dir are required.")
        process_directory(args.input_dir, args.output_dir, print_stats)
    else:
        # Single file mode
        if not (args.input_tif and args.output_rgb):
            parser.error("In single file mode, both --input_tif and --output_rgb are required.")
        convert_classified_tif_to_rgb(args.input_tif, args.output_rgb, print_stats)