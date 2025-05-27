import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import rasterio
import datetime
import time
from tqdm import tqdm
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score, classification_report, f1_score
from matplotlib.colors import LinearSegmentedColormap
# from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
import psutil
import warnings
import argparse


class ProgressMonitor:
    """
    A class to monitor and display progress during processing.
    """
    def __init__(self, total_steps, description="Processing"):
        """
        Initialize the progress monitor.
        
        Parameters:
        -----------
        total_steps : int
            Total number of steps in the process.
        description : str
            Description of the process being monitored.
        """
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = time.time()
        self.pbar = tqdm(total=total_steps, desc=description)
        
    def update(self, step_increment=1):
        """
        Update the progress by the specified increment.
        
        Parameters:
        -----------
        step_increment : int
            Number of steps to increment by.
        """
        self.current_step += step_increment
        self.pbar.update(step_increment)
        
    def set_description(self, description):
        """
        Update the progress description.
        
        Parameters:
        -----------
        description : str
            New description to display.
        """
        self.pbar.set_description(description)
        
    def add_info(self, info):
        """
        Add information to the progress bar.
        
        Parameters:
        -----------
        info : str
            Information to display.
        """
        self.pbar.set_postfix_str(info)
        
    def close(self):
        """Close the progress bar."""
        self.pbar.close()
        
    def get_elapsed_time(self):
        """Return elapsed time in seconds."""
        return time.time() - self.start_time


def load_raster_optimized(file_path, progress_monitor=None):
    """
    Load raster data with optimized memory usage.
    
    Parameters:
    -----------
    file_path : str
        Path to the raster file.
    progress_monitor : ProgressMonitor, optional
        Monitor to track progress.
        
    Returns:
    --------
    tuple
        Tuple containing (data_array, metadata).
    """
    if progress_monitor:
        progress_monitor.set_description(f"Loading raster {os.path.basename(file_path)}")
    
    start_time = time.time()
    
    with rasterio.Env(GDAL_CACHEMAX=512):  # Optimize GDAL cache size
        with rasterio.open(file_path) as src:
            # Get memory info to decide reading strategy
            mem_info = psutil.virtual_memory()
            available_mem = mem_info.available / (1024 * 1024 * 1024)  # Convert to GB
            
            # Calculate required memory (rough estimate)
            width, height = src.width, src.height
            pixel_size = np.dtype(src.dtypes[0]).itemsize
            required_mem = (width * height * pixel_size) / (1024 * 1024 * 1024)  # GB
            
            if progress_monitor:
                progress_monitor.add_info(f"Mem required: {required_mem:.2f} GB, Available: {available_mem:.2f} GB")
            
            # If we have enough memory, read directly
            if required_mem < available_mem * 0.7:  # Use up to 70% of available memory
                data = src.read(1)
            else:
                # Read in blocks/windows if the file is too large
                data = np.zeros((height, width), dtype=src.dtypes[0])
                block_size = min(1024, height, width)  # Adjust block size based on memory
                
                total_blocks = int(np.ceil(width / block_size) * np.ceil(height / block_size))
                if progress_monitor:
                    block_progress = tqdm(total=total_blocks, desc="Reading blocks")
                
                for y in range(0, height, block_size):
                    y_block_size = min(block_size, height - y)
                    for x in range(0, width, block_size):
                        x_block_size = min(block_size, width - x)
                        window = rasterio.windows.Window(x, y, x_block_size, y_block_size)
                        data[y:y+y_block_size, x:x+x_block_size] = src.read(1, window=window)
                        if progress_monitor:
                            block_progress.update(1)
                
                if progress_monitor:
                    block_progress.close()
            
            meta = src.meta
    
    if progress_monitor:
        elapsed = time.time() - start_time
        progress_monitor.add_info(f"Loaded in {elapsed:.2f}s")
        progress_monitor.update()
    
    return data, meta


def evaluate_classification_accuracy(ground_truth_path, classified_path, gt_classnames=None, 
                                     result_classnames=None, output_dir=None, resample_method='crop',
                                     progress_callback=None, n_jobs=-1):
    """
    Evaluate the accuracy of remote sensing classification results with improved performance.
    
    Parameters:
    -----------
    ground_truth_path : str
        Path to the ground truth GeoTIFF file.
    classified_path : str
        Path to the classified result GeoTIFF file.
    gt_classnames : dict, optional
        Dictionary mapping ground truth class values to class names.
    result_classnames : dict, optional
        Dictionary mapping classified result class values to class names.
    output_dir : str, optional
        Directory to save output figures and reports.
    resample_method : str, optional
        Method to use for handling dimension mismatch ('crop' or 'resample').
    progress_callback : function, optional
        Function to call for progress updates.
    n_jobs : int, optional
        Number of parallel jobs to use (-1 for all cores).
        
    Returns:
    --------
    dict
        Dictionary containing accuracy metrics, confusion matrix, and class mapping.
    """
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    # Determine number of steps for progress monitoring
    total_steps = 6  # Load GT + Load Classified + Resampling + Metrics + Plot CM + Plot Confused Classes
    monitor = ProgressMonitor(total_steps, "Classification Accuracy Assessment")
    
    # Parallel processing setup
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
        print(f"Using all available CPU cores: {n_jobs}")
    
    # Read raster data with optimized loading
    ground_truth, gt_meta = load_raster_optimized(ground_truth_path, monitor)
    classified, class_meta = load_raster_optimized(classified_path, monitor)
    
    # Check if the two rasters have the same dimensions
    if ground_truth.shape != classified.shape:
        monitor.set_description("Handling dimension mismatch")
        print(f"WARNING: Dimension mismatch detected. Ground Truth Shape: {ground_truth.shape}, Classified Shape: {classified.shape}")
        
        if resample_method == 'crop':
            # Optimize by only cropping once instead of multiple operations
            min_rows = min(ground_truth.shape[0], classified.shape[0])
            min_cols = min(ground_truth.shape[1], classified.shape[1])
            
            ground_truth = ground_truth[:min_rows, :min_cols]
            classified = classified[:min_rows, :min_cols]
            
            print(f"Using common dimensions: {ground_truth.shape}")
            
        elif resample_method == 'resample':
            print("Resampling classified image to match ground truth dimensions...")
            
            # Vectorized resampling approach for better performance
            resampled = np.zeros(ground_truth.shape, dtype=classified.dtype)
            
            # Create transforms
            src_transform = from_bounds(
                0, 0, classified.shape[1], classified.shape[0], 
                classified.shape[1], classified.shape[0]
            )
            dst_transform = from_bounds(
                0, 0, ground_truth.shape[1], ground_truth.shape[0], 
                ground_truth.shape[1], ground_truth.shape[0]
            )
            
            # Block-based resampling for large images
            if max(ground_truth.shape) > 5000:  # Large image threshold
                block_size = 2000  # Adjust based on available memory
                for y in range(0, ground_truth.shape[0], block_size):
                    y_block_size = min(block_size, ground_truth.shape[0] - y)
                    for x in range(0, ground_truth.shape[1], block_size):
                        x_block_size = min(block_size, ground_truth.shape[1] - x)
                        dst_window = np.zeros((y_block_size, x_block_size), dtype=classified.dtype)
                        
                        # Calculate source window coordinates
                        src_x = int(x * classified.shape[1] / ground_truth.shape[1])
                        src_y = int(y * classified.shape[0] / ground_truth.shape[0])
                        src_width = int(x_block_size * classified.shape[1] / ground_truth.shape[1])
                        src_height = int(y_block_size * classified.shape[0] / ground_truth.shape[0])
                        
                        src_window = classified[src_y:src_y+src_height, src_x:src_x+src_width]
                        
                        # Resize this block
                        reproject(
                            source=src_window.reshape((1, src_window.shape[0], src_window.shape[1])),
                            destination=dst_window.reshape((1, dst_window.shape[0], dst_window.shape[1])),
                            src_transform=src_transform,
                            src_crs=class_meta.get('crs'),
                            dst_transform=dst_transform,
                            dst_crs=gt_meta.get('crs'),
                            resampling=Resampling.nearest
                        )
                        
                        resampled[y:y+y_block_size, x:x+x_block_size] = dst_window
            else:
                # For smaller images, do it all at once
                reproject(
                    source=classified.reshape((1, classified.shape[0], classified.shape[1])),
                    destination=resampled.reshape((1, resampled.shape[0], resampled.shape[1])),
                    src_transform=src_transform,
                    src_crs=class_meta.get('crs'),
                    dst_transform=dst_transform,
                    dst_crs=gt_meta.get('crs'),
                    resampling=Resampling.nearest
                )
            
            classified = resampled
            print(f"Resampled classified image to: {classified.shape}")
        else:
            raise ValueError(f"Unknown resample method: {resample_method}. Use 'crop' or 'resample'.")
    
    monitor.update()
    monitor.set_description("Computing metrics")
    
    # Get unique class values efficiently
    # Use np.unique with return_counts=True to avoid multiple passes
    gt_values, gt_counts = np.unique(ground_truth[ground_truth != gt_meta.get('nodata', 0)], return_counts=True)
    class_values, class_counts = np.unique(classified[classified != class_meta.get('nodata', 0)], return_counts=True)
    
    # Create class mappings if not provided
    if gt_classnames is None:
        gt_classnames = {val: f"Class_{val}" for val in gt_values}
    
    if result_classnames is None:
        result_classnames = {val: f"Class_{val}" for val in class_values}
    
    # Create a cross-reference mapping
    class_mapping = {}
    for gt_val, gt_name in gt_classnames.items():
        for class_val, class_name in result_classnames.items():
            if gt_name == class_name:
                class_mapping[gt_val] = class_val
    
    # Optimize mask creation by combining operations
    mask = np.logical_and(
        ground_truth != gt_meta.get('nodata', 0),
        classified != class_meta.get('nodata', 0)
    )
    
    # Optimize memory usage by compressing data types if possible
    y_true = ground_truth[mask]
    y_pred = classified[mask]
    
    # Free memory of original arrays if they're large
    if ground_truth.size > 100_000_000:  # About 100 million elements
        del ground_truth
        # Explicitly trigger garbage collection if needed
        import gc
        gc.collect()
    
    # Map prediction values using vectorized numpy operations
    reverse_mapping = {v: k for k, v in class_mapping.items()}
    y_pred_mapped = np.copy(y_pred)
    
    # Vectorized approach to mapping values
    for pred_val, gt_val in reverse_mapping.items():
        y_pred_mapped = np.where(y_pred == pred_val, gt_val, y_pred_mapped)
    
    # Calculate accuracy metrics
    metrics = {}
    metrics['overall_accuracy'] = accuracy_score(y_true, y_pred_mapped)
    metrics['kappa'] = cohen_kappa_score(y_true, y_pred_mapped)
    
    # Generate class labels
    gt_labels = [gt_classnames[val] for val in sorted(gt_classnames.keys())]
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred_mapped, labels=sorted(gt_classnames.keys()))
    
    # Calculate precision, recall, and F1-score for each class
    class_report = classification_report(y_true, y_pred_mapped, 
                                         labels=sorted(gt_classnames.keys()),
                                         target_names=gt_labels,
                                         output_dict=True)
    metrics['classification_report'] = class_report
    
    # Calculate overall F1 score (weighted)
    metrics['f1_weighted'] = f1_score(y_true, y_pred_mapped, average='weighted')
    
    # Calculate user's accuracy (precision) and producer's accuracy (recall)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # Ignore division by zero
        metrics['users_accuracy'] = np.diag(cm) / np.maximum(cm.sum(axis=0), 1)  # Avoid div by zero
        metrics['producers_accuracy'] = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)  # Avoid div by zero
    
    monitor.update()
    monitor.set_description("Creating visualizations")
    
    # Visualize confusion matrix
    plt.figure(figsize=(12, 10))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # Ignore division by zero
        cm_normalized = cm.astype('float') / np.maximum(cm.sum(axis=1)[:, np.newaxis], 1)
    
    # Custom colormap for better visualization
    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#f7fbff", "#2171b5"], N=256)
    
    sns.heatmap(cm_normalized, annot=cm, fmt='d', cmap=cmap,
                xticklabels=gt_labels, yticklabels=gt_labels)
    plt.title('Confusion Matrix')
    plt.ylabel('Ground Truth')
    plt.xlabel('Classified Result')
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300)
    
    monitor.update()
    
    # Create a figure showing the most confused classes
    conf_values = []
    for i in range(len(gt_labels)):
        for j in range(len(gt_labels)):
            if i != j and cm[i, j] > 0:
                total_class_pixels = cm.sum(axis=1)[i]
                if total_class_pixels > 0:  # Avoid division by zero
                    conf_values.append((gt_labels[i], gt_labels[j], cm[i, j], cm[i, j]/total_class_pixels))
    
    # Sort by confusion percentage
    conf_values.sort(key=lambda x: x[3], reverse=True)
    
    # Plot top confused classes
    plt.figure(figsize=(12, 8))
    confusion_data = [(f"{true} -> {pred}", count) for true, pred, count, _ in conf_values[:10]]
    
    if confusion_data:  # Only if there are confused classes
        classes, values = zip(*confusion_data)
        plt.barh(classes, values)
        plt.xlabel('Number of Pixels')
        plt.title('Top Confused Classes')
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'confused_classes.png'), dpi=300)
    
    monitor.update()
    
    # Print results
    print(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}")
    print(f"Kappa Coefficient: {metrics['kappa']:.4f}")
    print(f"Weighted F1 Score: {metrics['f1_weighted']:.4f}")
    print("\nClass Mapping:")
    for gt_val, class_val in class_mapping.items():
        print(f"  Ground Truth {gt_val} ({gt_classnames[gt_val]}) -> Classified {class_val} ({result_classnames[class_val]})")
    
    # Close the plots
    plt.close('all')
    
    monitor.update()
    monitor.close()
    
    # Return the metrics and confusion matrix
    results = {
        'metrics': metrics,
        'confusion_matrix': cm,
        'class_mapping': class_mapping,
        'ground_truth_classes': gt_classnames,
        'classified_classes': result_classnames
    }
    
    return results


def save_classification_report(results, output_path, include_timestamp=True):
    """
    Save the classification accuracy assessment report to a text file.
    
    Parameters:
    -----------
    results : dict
        Results dictionary from evaluate_classification_accuracy.
    output_path : str
        Path to save the report.
    include_timestamp : bool, optional
        Whether to include a timestamp in the report.
    """
    metrics = results['metrics']
    cm = results['confusion_matrix']
    mapping = results['class_mapping']
    gt_classes = results['ground_truth_classes']
    
    with open(output_path, 'w') as f:
        f.write("REMOTE SENSING CLASSIFICATION ACCURACY ASSESSMENT\n")
        f.write("="*50 + "\n")
        
        if include_timestamp:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"Generated on: {timestamp}\n\n")
        
        f.write("OVERALL METRICS\n")
        f.write("-"*50 + "\n")
        f.write(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}\n")
        f.write(f"Kappa Coefficient: {metrics['kappa']:.4f}\n")
        f.write(f"Weighted F1 Score: {metrics['f1_weighted']:.4f}\n\n")
        
        f.write("CLASS MAPPING\n")
        f.write("-"*50 + "\n")
        for gt_val, class_val in mapping.items():
            f.write(f"Ground Truth {gt_val} ({gt_classes[gt_val]}) -> Classified {class_val}\n")
        f.write("\n")
        
        f.write("CLASS-WISE METRICS\n")
        f.write("-"*50 + "\n")
        for class_name, metrics_dict in metrics['classification_report'].items():
            if class_name in ['accuracy', 'macro avg', 'weighted avg']:
                continue
            f.write(f"Class: {class_name}\n")
            f.write(f"  Precision: {metrics_dict['precision']:.4f}\n")
            f.write(f"  Recall: {metrics_dict['recall']:.4f}\n")
            f.write(f"  F1-Score: {metrics_dict['f1-score']:.4f}\n")
            f.write(f"  Support: {metrics_dict['support']}\n\n")
        
        f.write("CONFUSION MATRIX\n")
        f.write("-"*50 + "\n")
        f.write("Rows: Ground Truth, Columns: Classified Result\n")
        f.write(str(cm) + "\n")

    print(f"Classification report saved to {output_path}")


def generate_error_map(ground_truth_path, classified_path, output_path, resample_method='crop', 
                       chunk_size=1000, class_mapping=None):
    """
    Generate a spatial error map showing where classification errors occur.
    Uses chunked processing for large datasets.
    
    Parameters:
    -----------
    ground_truth_path : str
        Path to the ground truth GeoTIFF file.
    classified_path : str
        Path to the classified result GeoTIFF file.
    output_path : str
        Path to save the error map GeoTIFF.
    resample_method : str, optional
        Method to use for handling dimension mismatch ('crop' or 'resample').
    chunk_size : int, optional
        Size of chunks for processing large datasets.
    class_mapping : dict, optional
        Mapping from classified values to ground truth values.
    
    Returns:
    --------
    None
    """
    print(f"Generating error map: {os.path.basename(output_path)}")
    monitor = ProgressMonitor(3, "Generating Error Map")
    
    # Read raster data using optimized loader
    ground_truth, gt_meta = load_raster_optimized(ground_truth_path, monitor)
    classified, class_meta = load_raster_optimized(classified_path, monitor)
    
    # Handle dimension mismatch if necessary
    if ground_truth.shape != classified.shape:
        print(f"WARNING: Dimension mismatch detected in error map generation.")
        print(f"Ground Truth Shape: {ground_truth.shape}, Classified Shape: {classified.shape}")
        
        if resample_method == 'crop':
            # Determine the minimum dimensions
            min_rows = min(ground_truth.shape[0], classified.shape[0])
            min_cols = min(ground_truth.shape[1], classified.shape[1])
            
            # Crop both images
            ground_truth = ground_truth[:min_rows, :min_cols]
            classified = classified[:min_rows, :min_cols]
            print(f"Cropped to common dimensions: {ground_truth.shape}")
        else:
            # For simplicity, we'll just crop. Implement resampling if needed.
            min_rows = min(ground_truth.shape[0], classified.shape[0])
            min_cols = min(ground_truth.shape[1], classified.shape[1])
            ground_truth = ground_truth[:min_rows, :min_cols]
            classified = classified[:min_rows, :min_cols]
    
    monitor.set_description("Processing error map")
    
    # Get mask of valid pixels
    mask = np.logical_and(
        ground_truth != gt_meta.get('nodata', 0), 
        classified != class_meta.get('nodata', 0)
    )
    
    # Use the provided class mapping or create a basic one
    if class_mapping is None:
        # If no mapping provided, try to create one based on value equality
        unique_gt = np.unique(ground_truth[mask])
        unique_cls = np.unique(classified[mask])
        
        class_mapping = {}
        for gt_val in unique_gt:
            for cls_val in unique_cls:
                if gt_val == cls_val:
                    class_mapping[cls_val] = gt_val
        
        # If still no mapping (different class ID systems), use example mapping
        if len(class_mapping) == 0:
            class_mapping = {9: 0, 1: 1, 2: 2, 16: 3, 11: 4, 15: 5, 14: 6}
    
    # Prepare output array
    error_map = np.zeros_like(ground_truth, dtype=np.uint8)
    
    # For very large datasets, process in chunks
    height, width = ground_truth.shape
    is_large = height * width > 100_000_000  # About 100 million pixels
    
    if is_large:
        # Process chunks sequentially to save memory
        chunks_y = range(0, height, chunk_size)
        chunks_x = range(0, width, chunk_size)
        
        total_chunks = len(list(chunks_y)) * len(list(chunks_x))
        with tqdm(total=total_chunks, desc="Processing chunks") as pbar:
            for y in chunks_y:
                chunk_height = min(chunk_size, height - y)
                for x in chunks_x:
                    chunk_width = min(chunk_size, width - x)
                    
                    # Get chunks
                    gt_chunk = ground_truth[y:y+chunk_height, x:x+chunk_width]
                    cls_chunk = classified[y:y+chunk_height, x:x+chunk_width]
                    mask_chunk = mask[y:y+chunk_height, x:x+chunk_width]
                    
                    # Create a mapped version of the classified chunk
                    cls_mapped = np.copy(cls_chunk)
                    for cls_val, gt_val in class_mapping.items():
                        cls_mapped[cls_chunk == cls_val] = gt_val
                    
                    # Compare and mark errors
                    chunk_error = np.zeros_like(gt_chunk, dtype=np.uint8)
                    chunk_error[mask_chunk] = (gt_chunk[mask_chunk] != cls_mapped[mask_chunk]).astype(np.uint8)
                    
                    # Write to output
                    error_map[y:y+chunk_height, x:x+chunk_width] = chunk_error
                    
                    pbar.update(1)
    else:
        # For smaller datasets, process all at once
        # Map classified values to ground truth values
        classified_mapped = np.copy(classified)
        for cls_val, gt_val in class_mapping.items():
            classified_mapped[classified == cls_val] = gt_val
        
        # Mark errors (only where mask is True)
        error_map[mask] = (ground_truth[mask] != classified_mapped[mask]).astype(np.uint8)
    
    monitor.update()
    monitor.set_description("Writing error map to disk")
    
    # Write error map to file
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=error_map.shape[0],
        width=error_map.shape[1],
        count=1,
        dtype=np.uint8,
        crs=gt_meta['crs'],
        transform=gt_meta['transform'],
        compress='lzw',  # Add compression for smaller file size
    ) as dst:
        dst.write(error_map, 1)
        dst.write_colormap(
            1, {
                0: (0, 255, 0, 255),  # Correct (green)
                1: (255, 0, 0, 255),  # Error (red)
            }
        )
    
    monitor.update()
    monitor.close()
    print(f"Error map saved to {output_path}")


def run_accuracy_assessment(gt_path, cls_res_path, output_dir, gt_classnames, result_classnames, 
                           resample_method='crop'):
    """
    Run the complete accuracy assessment workflow with progress monitoring.
    
    Parameters:
    -----------
    gt_path : str
        Path to the ground truth GeoTIFF file.
    cls_res_path : str
        Path to the classified result GeoTIFF file.
    output_dir : str
        Directory to save output files.
    gt_classnames : dict
        Dictionary mapping ground truth class values to class names.
    result_classnames : dict
        Dictionary mapping classified result class values to class names.
    resample_method : str, optional
        Method to use for handling dimension mismatch ('crop' or 'resample').
    """
    start_time = time.time()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*50}")
    print(f"REMOTE SENSING CLASSIFICATION ACCURACY ASSESSMENT")
    print(f"{'='*50}")
    print(f"Ground Truth: {os.path.basename(gt_path)}")
    print(f"Classification: {os.path.basename(cls_res_path)}")
    print(f"Output Directory: {output_dir}")
    print(f"{'='*50}\n")
    
    # Set up output paths
    output_report = os.path.join(output_dir, "classification_report.txt")
    output_errormap = os.path.join(output_dir, "error_map.tif")
    
    # Step 1: Evaluate classification accuracy
    print("Step 1: Evaluating classification accuracy...")
    results = evaluate_classification_accuracy(
        gt_path, 
        cls_res_path,
        gt_classnames=gt_classnames,
        result_classnames=result_classnames,
        output_dir=output_dir,
        resample_method=resample_method
    )
    
    # Step 2: Save classification report
    print("\nStep 2: Saving classification report...")
    save_classification_report(results, output_report)
    
    # Step 3: Generate error map
    print("\nStep 3: Generating error map...")
    generate_error_map(
        gt_path, 
        cls_res_path, 
        output_errormap, 
        resample_method=resample_method,
        class_mapping={v: k for k, v in results['class_mapping'].items()}
    )
    
    # Report total execution time
    elapsed_time = time.time() - start_time
    print(f"\nAssessment completed in {elapsed_time:.2f} seconds.")
    print(f"Results saved to {output_dir}")
    print(f"Classification report: {output_report}")
    print(f"Error map: {output_errormap}")

def create_arg_parser():
    """
    Create an argument parser for command line execution.
    
    Returns:
    --------
    argparse.ArgumentParser
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Remote Sensing Classification Accuracy Assessment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--gt_path", "-g", type=str, 
                        help="Path to the ground truth GeoTIFF file")
    parser.add_argument("--cls_res_path", "-c", type=str, 
                        help="Path to the classified result GeoTIFF file")
    parser.add_argument("--output_dir","-o", type=str, 
                        help="Directory to save output files")
    parser.add_argument("--gt_classnames", type=str, default=None, 
                        help="dictionary mapping ground truth class values to class names (e.g., '1:Class1,2:Class2')")
    parser.add_argument("--result_classnames", type=str, default=None, 
                        help="dictionary mapping classified result class values to class names (e.g., '1:Class1,2:Class2')")
    parser.add_argument("--resample_method", type=str, choices=['crop', 'resample'], default='crop', 
                        help="Method to handle dimension mismatch ('crop' or 'resample'): crop for faster processing, resample for better accuracy")
    
    return parser

def parse_class_mapping(mapping_str):
    """
    Parse a class mapping string into a dictionary.
    
    Parameters:
    -----------
    mapping_str : str
        String representation of the mapping (e.g., "1:2,3:4").
    
    Returns:
    --------
    dict
        Parsed class mapping dictionary.
    """
    mapping = {}
    pairs = mapping_str.split(',')
    for pair in pairs:
        k, v = map(int, pair.split(':'))
        mapping[k] = v
    return mapping if mapping else None

def main():
    parser = create_arg_parser()
    
    args = parser.parse_args()

    if not os.path.exists(args.gt_path):
        raise FileNotFoundError(f"Ground truth file not found: {args.gt_path}")
    if not os.path.exists(args.cls_res_path):
        raise FileNotFoundError(f"Classified result file not found: {args.cls_res_path}")
    
    gt_classnames = parse_class_mapping(args.gt_classnames) if args.gt_classnames else None
    result_classnames = parse_class_mapping(args.result_classnames) if args.result_classnames else None

    if gt_classnames is None:
        print("No ground truth class names provided. Using default mapping.")
        gt_classnames = {
          0: "canals",
          1: "cultivation",
          2: "forest",
          3: "other_builds",
          4: "reservoir_pits",
          5: "rural",
          6: "urban"}
        # gt_classnames = {
        #     2: "forest",
        #     0: "water",
        #     4: "water",
        #     3: "builds",
        #     5: "builds",
        #     6: "builds"}
    if result_classnames is None:
        print("No classified result class names provided. Using default mapping.")
        # result_classnames = {
        #   9: "canals",
        #   1: "cultivation",
        #   2: "forest",
        #   16: "other_builds",
        #   11: "reservoir_pits",
        #   15: "rural",
        #   14: "urban"}
        # result_classnames = {
        #     1: "forest",
        #     3: "water",
        #     4: "builds"}
        result_classnames = {
            1:"forest",
            4:"canals",
            6:"reservoir_pits",
            9:"urban",
            10:"rural",
            11:"other_builds"}
    
    run_accuracy_assessment(
        args.gt_path, 
        args.cls_res_path, 
        args.output_dir, 
        gt_classnames=gt_classnames, 
        result_classnames=result_classnames,
        resample_method=args.resample_method
    )
    
if __name__ == "__main__":
    # python scripts\assessment_v1.py --gt_path F:\chrome\2024_S2_allbands\cls_results\classified_image.tif --cls_res_path predict\01ov_12class_light\s2_2024_mosaic_12class_light_model.tif --output_dir predict\01ov_12class_light\assessment_results
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        print(traceback.format_exc())
        import sys
        sys.exit(1)