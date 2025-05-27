"""
Random Forest Classification for Large Remote Sensing Images
- Handles large TIF images (20GB+) with memory-efficient processing
- Processes polygon training samples from SHP files with random sampling
- Evaluates classification accuracy with confusion matrix
- Includes visualization and reporting functions
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
# from rasterio.features import rasterize
from rasterio.windows import Window
# from rasterio.plot import show
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import mapping
import joblib
from tqdm import tqdm
import gc
import time
from matplotlib.colors import ListedColormap

# Memory usage monitoring function
def print_memory_usage(message="Current memory usage:"):
    """Print the current memory usage of the Python process"""
    import psutil
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_usage_mb = memory_info.rss / 1024 / 1024
    print(f"{message} {memory_usage_mb:.2f} MB")

class LargeRasterClassifier:
    def __init__(self, raster_path, shapefile_path, output_directory="output",
                  model_name="rf_model", classified_img_name="classified_image", chunk_size=1000, sample_rate=0.1, test_size=0.3, n_estimators=100, max_depth=None):
        """
        Initialize the classifier with file paths and parameters

        Parameters:
        -----------
        raster_path : str
            Path to the TIF image file
        shapefile_path : str
            Path to the SHP file with training polygons
        output_directory : str
            Directory to save outputs
        model_name : str
            Name of the model file to save/load
        classified_img_name : str
            Name of the classified image file to save
        chunk_size : int
            Size of image chunks to process at once
        sample_rate : float
            Rate of random sampling from polygons (0-1)
        test_size : float
            Fraction of data to use for validation (0-1)
        n_estimators : int
            Number of trees in random forest
        max_depth : int
            Maximum depth of trees in random forest
        """
        self.raster_path = raster_path
        self.shapefile_path = shapefile_path
        self.output_directory = output_directory
        self.model_name = model_name
        self.classified_img_name = classified_img_name
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.test_size = test_size
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.class_mapping = None
        self.rf_model = None
        self.trained = False
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
            
        # Get initial raster information
        with rasterio.open(self.raster_path) as src:
            self.meta = src.meta
            self.shape = src.shape
            self.num_bands = src.count
            
        print(f"Raster dimensions: {self.shape}, bands: {self.num_bands}")
        print_memory_usage("Initial memory usage:")
        
    def read_training_data(self):
        """Read training data from shapefile and establish class mapping"""
        # Read the shapefile containing training polygons
        self.gdf = gpd.read_file(self.shapefile_path)
        
        # Check for class column
        class_columns = [col for col in self.gdf.columns if "class" in col.lower() or "category" in col.lower()]
        if not class_columns:
            raise ValueError("No class or category column found in shapefile")
        
        self.class_column = class_columns[0]
        
        # Create a mapping from class names to numeric labels
        classes = sorted(self.gdf[self.class_column].unique())
        self.class_mapping = {cls: i for i, cls in enumerate(classes)}
        self.inv_class_mapping = {i: cls for i, cls in enumerate(classes)}
        
        print(f"Found {len(classes)} classes: {', '.join(classes)}")
        print_memory_usage()
        
    def sample_from_polygons(self):
        """Extract samples from the training polygons with random sampling"""
        print("Extracting samples from training polygons...")
        self.X_train = []
        self.y_train = []
        
        # Track samples per class to ensure balanced training
        samples_per_class = {}
        
        with rasterio.open(self.raster_path) as src:
            # Process each class
            for class_name, class_id in self.class_mapping.items():
                print(f"Processing class: {class_name}")
                # Filter polygons for current class
                class_polygons = self.gdf[self.gdf[self.class_column] == class_name]
                
                if class_polygons.empty:
                    print(f"Warning: No polygons found for class {class_name}")
                    samples_per_class[class_name] = 0
                    continue
                
                # For each polygon
                samples_per_polygon = []
                class_samples = []
                class_labels = []
                
                for idx, poly in tqdm(class_polygons.iterrows(), total=len(class_polygons)):
                    # Get mask and data for this polygon
                    try:
                        geometry = [mapping(poly.geometry)]
                        out_image, out_transform = mask(src, geometry, crop=True)
                        
                        # Check for completely masked or empty result
                        if out_image.size == 0 or (src.nodata is not None and np.all(out_image == src.nodata)):
                            print(f"Warning: Empty or completely masked polygon {idx}")
                            continue
                            
                        # Get valid pixels (non-masked)
                        valid_mask = np.all(out_image != src.nodata, axis=0) if src.nodata is not None else np.ones(out_image.shape[1:], dtype=bool)
                        
                        # Make sure there are valid pixels
                        if np.sum(valid_mask) == 0:
                            print(f"Warning: No valid pixels in polygon {idx}")
                            continue
                        
                        # If we need to sample (too many pixels)
                        if valid_mask.sum() > 1000:  # Arbitrary threshold
                            # Random sample based on sample_rate
                            indices = np.where(valid_mask)
                            num_samples = max(int(len(indices[0]) * self.sample_rate), 100)  # At least 100 samples
                            sample_indices = np.random.choice(len(indices[0]), size=num_samples, replace=False)
                            
                            # Get pixel values for these indices
                            samples = out_image[:, indices[0][sample_indices], indices[1][sample_indices]].T
                            samples_per_polygon.append(len(samples))
                            
                            if len(samples) > 0:
                                class_samples.extend(samples)
                                class_labels.extend([class_id] * len(samples))
                        else:
                            # Take all pixels if below threshold
                            samples = out_image[:, valid_mask].T
                            samples_per_polygon.append(len(samples))
                            
                            if len(samples) > 0:
                                class_samples.extend(samples)
                                class_labels.extend([class_id] * len(samples))
                    except Exception as e:
                        print(f"Error processing polygon {idx}: {e}")
                
                # Add all samples for this class
                samples_per_class[class_name] = len(class_samples)
                if len(class_samples) > 0:
                    self.X_train.extend(class_samples)
                    self.y_train.extend(class_labels)
                    
                    avg_samples = np.mean(samples_per_polygon) if samples_per_polygon else 0
                    print(f"Class {class_name}: {len(samples_per_polygon)} polygons, {len(class_samples)} total samples, avg {avg_samples:.1f} samples/polygon")
                else:
                    print(f"Warning: No valid samples collected for class {class_name}")
        
        # Print summary of samples per class
        print("\nSamples per class summary:")
        for class_name, count in samples_per_class.items():
            print(f"  {class_name}: {count} samples")
        
        # Check if any class has zero samples
        zero_sample_classes = [cls for cls, count in samples_per_class.items() if count == 0]
        if zero_sample_classes:
            print(f"\nWARNING: The following classes have zero samples: {', '.join(zero_sample_classes)}")
            print("These classes will not be represented in the classification model!")
        
        # Convert to numpy arrays
        self.X_train = np.array(self.X_train)
        self.y_train = np.array(self.y_train)
        
        print(f"Total samples: {len(self.X_train)}, shape: {self.X_train.shape}")
        print_memory_usage("After sampling:")
        
    def train_model(self):
        """Train the Random Forest classifier"""
        print("Training Random Forest model...")
        start_time = time.time()
        
        # Split training data
        X_train, X_val, y_train, y_val = train_test_split(
            self.X_train, self.y_train, test_size=self.test_size, random_state=42, stratify=self.y_train
        )
        
        # Initialize and train the Random Forest model
        self.rf_model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            n_jobs=-1,  # Use all available cores
            random_state=42,
            verbose=1
        )
        
        self.rf_model.fit(X_train, y_train)
        
        # Evaluate on validation set
        y_pred = self.rf_model.predict(X_val)
        
        # Calculate metrics
        self.accuracy = accuracy_score(y_val, y_pred)
        self.conf_matrix = confusion_matrix(y_val, y_pred)
        
        # Save validation data for later accuracy assessment
        self.X_val = X_val
        self.y_val = y_val
        
        elapsed_time = time.time() - start_time
        print(f"Training completed in {elapsed_time:.2f} seconds")
        print(f"Validation accuracy: {self.accuracy:.4f}")
        print_memory_usage("After training:")
        
        # Save the model
        self.save_model()
        self.trained = True
        
    def save_model(self, model_path=None):
        """Save the trained model to disk"""
        if model_path is None:
            model_path = os.path.join(self.output_directory, f"{self.model_name}.joblib")
            
        joblib.dump(self.rf_model, model_path)
        print(f"Model saved to {model_path}")
        
        # Save class mapping
        mapping_path = os.path.join(self.output_directory, "class_mapping.joblib")
        joblib.dump({
            'class_mapping': self.class_mapping, 
            'inv_class_mapping': self.inv_class_mapping
        }, mapping_path)
        
    def load_model(self, model_path=None, mapping_path=None):
        """Load a previously trained model"""
        if model_path is None:
            model_path = os.path.join(self.output_directory, f"{self.model_name}.joblib")
            print(f"Loading model from {model_path}")
        if mapping_path is None:
            mapping_path = os.path.join(self.output_directory, "class_mapping.joblib")
            print(f"Loading class mapping from {mapping_path}")
            
        self.rf_model = joblib.load(model_path)
        mappings = joblib.load(mapping_path)
        self.class_mapping = mappings['class_mapping']
        self.inv_class_mapping = mappings['inv_class_mapping']
        
        print(f"Model loaded from {model_path}")
        self.trained = True
        
    def classify_image(self, output_path=None):
        """Classify the entire image in chunks to manage memory"""
        if not self.trained:
            raise ValueError("Model must be trained or loaded before classification")
            
        if output_path is None:
            output_path = os.path.join(self.output_directory, f"{self.classified_img_name}.tif")
            
        print(f"Classifying image in chunks of size {self.chunk_size}...")
        start_time = time.time()
        
        with rasterio.open(self.raster_path) as src:
            # Create a new raster for classification results
            meta = src.meta.copy()
            meta.update({
                'count': 1,
                'dtype': 'uint8',
                'nodata': 255  # Use 255 as nodata value
            })
            
            with rasterio.open(output_path, 'w', **meta) as dst:
                # Process the image in chunks
                rows, cols = src.shape
                
                # Create progress bar
                total_chunks = ((rows - 1) // self.chunk_size + 1) * ((cols - 1) // self.chunk_size + 1)
                print(f"Total chunks to process: {total_chunks}")
                with tqdm(total=total_chunks) as pbar:
                    for row_start in range(0, rows, self.chunk_size):
                        row_end = min(row_start + self.chunk_size, rows)
                        
                        for col_start in range(0, cols, self.chunk_size):
                            col_end = min(col_start + self.chunk_size, cols)
                            
                            # Define the window for the current chunk
                            window = Window(col_start, row_start, col_end - col_start, row_end - row_start)
                            
                            # Read the data for this window
                            chunk = src.read(window=window)
                            
                            # Reshape the data for prediction (bands, height, width) -> (pixels, bands)
                            n_bands, n_rows, n_cols = chunk.shape
                            flat_chunk = chunk.reshape(n_bands, n_rows * n_cols).T
                            
                            # Find valid pixels (not nodata)
                            if src.nodata is not None:
                                valid_mask = ~np.any(flat_chunk == src.nodata, axis=1)
                                valid_indices = np.where(valid_mask)[0]
                            else:
                                valid_indices = np.arange(flat_chunk.shape[0])
                            
                            # Initialize result array for this chunk (all nodata value)
                            result_chunk = np.full((n_rows * n_cols), 255, dtype=np.uint8)
                            
                            if len(valid_indices) > 0:
                                # Predict only for valid pixels
                                valid_data = flat_chunk[valid_indices]
                                predictions = self.rf_model.predict(valid_data)
                                
                                # Place predictions back into the result array
                                result_chunk[valid_indices] = predictions
                            
                            # Reshape back to 2D and write to output
                            result_2d = result_chunk.reshape((n_rows, n_cols))
                            dst.write(result_2d, window=window, indexes=1)
                            
                            # Update progress bar
                            pbar.update(1)
                            
                            # Clean up memory
                            del chunk, flat_chunk, result_chunk, result_2d
                            gc.collect()
                            
        elapsed_time = time.time() - start_time
        print(f"Classification completed in {elapsed_time:.2f} seconds")
        print(f"Classified image saved to {output_path}")
        print_memory_usage("After classification:")
        
        return output_path
                
    def evaluate_accuracy(self):
        """Evaluate model accuracy and generate confusion matrix plot"""
        if not self.trained:
            raise ValueError("Model must be trained before evaluation")
            
        # Predict on validation data
        y_pred = self.rf_model.predict(self.X_val)
        
        # Calculate accuracy
        accuracy = accuracy_score(self.y_val, y_pred)
        
        # Identify the unique classes present in both actual and predicted values
        unique_classes = sorted(np.unique(np.concatenate([self.y_val, y_pred])))
        
        # Generate confusion matrix
        conf_matrix = confusion_matrix(self.y_val, y_pred, labels=unique_classes)
        
        # Generate classification report with only the classes that are present
        class_names = [self.inv_class_mapping[i] for i in unique_classes if i in self.inv_class_mapping]
        report = classification_report(self.y_val, y_pred, labels=unique_classes, target_names=class_names)
        
        print(f"Overall Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(report)
        
        # Plot and save confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        cm_path = os.path.join(self.output_directory, "confusion_matrix.png")
        plt.tight_layout()
        plt.savefig(cm_path)
        plt.close()
        
        # Save report to text file
        report_path = os.path.join(self.output_directory, "classification_report.txt")
        with open(report_path, 'w') as f:
            f.write(f"Overall Accuracy: {accuracy:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(report)
            
        print(f"Evaluation results saved to {self.output_directory}")
        
        # Return accuracy for convenience
        return accuracy
    
    def visualize_results(self, classified_path=None):
        """Create a colorful visualization of the classification results"""
        if classified_path is None:
            classified_path = os.path.join(self.output_directory, f"{self.classified_img_name}.tif")
            
        # Create a custom colormap for classes
        # First get the classes that are actually in the model
        if hasattr(self, 'rf_model') and self.rf_model is not None:
            used_classes = sorted(list(set(self.y_train)))
        else:
            used_classes = sorted(self.class_mapping.values())
            
        n_classes = len(used_classes)
        colors = plt.cm.rainbow(np.linspace(0, 1, n_classes))
        
        # Create a colormap that includes the nodata value (255)
        # Add a transparent color for nodata value
        cmap_colors = np.vstack([colors, [1, 1, 1, 0]])  # White transparent for nodata
        class_indices = np.append(used_classes, 255)  # Add nodata value
        class_cmap = ListedColormap(cmap_colors)
            
        try:
            # Read the classified image
            with rasterio.open(classified_path) as src:
                classified = src.read(1)
                transform = src.transform
                
                # Check if we have valid data
                if np.all(classified == 255):
                    print("Warning: Classified image contains only nodata values!")
                
                # Create figure
                fig, ax = plt.subplots(figsize=(12, 10))
                
                # Plot the image with colormap
                # Use a masked array to make nodata values transparent
                masked_data = np.ma.masked_where(classified == 255, classified)
                image = ax.imshow(masked_data, cmap=class_cmap, interpolation='nearest')
                
                # Add colorbar only for valid classes
                valid_ticks = np.array(used_classes)
                cbar = plt.colorbar(image, ax=ax, ticks=valid_ticks)
                
                # Get class names for the ticks
                class_names = [self.inv_class_mapping.get(i, f"Class {i}") for i in valid_ticks]
                cbar.set_ticklabels(class_names)
                
                # Add title and labels
                ax.set_title('Land Cover Classification Results')
                ax.set_axis_off()
                
                # Save figure
                viz_path = os.path.join(self.output_directory, "classification_visualization.png")
                plt.tight_layout()
                plt.savefig(viz_path, dpi=300)
                plt.close()
                
                print(f"Visualization saved to {viz_path}")
                
        except Exception as e:
            print(f"Error visualizing results: {e}")
            # Create a simpler visualization as fallback
            try:
                with rasterio.open(classified_path) as src:
                    classified = src.read(1)
                    
                    # Simple visualization without fancy colormap
                    plt.figure(figsize=(12, 10))
                    plt.imshow(classified, cmap='viridis')
                    plt.colorbar(label='Class')
                    plt.title('Land Cover Classification (Simple Visualization)')
                    
                    viz_path = os.path.join(self.output_directory, "classification_visualization_simple.png")
                    plt.savefig(viz_path, dpi=300)
                    plt.close()
                    
                    print(f"Simple visualization saved to {viz_path}")
            except Exception as e2:
                print(f"Could not create even simple visualization: {e2}")
            
            print(f"Visualization saved to {viz_path}")
            
    def feature_importance(self):
        """Analyze and visualize feature importance"""
        if not self.trained:
            raise ValueError("Model must be trained before feature importance analysis")
            
        # Get feature importances
        importances = self.rf_model.feature_importances_
        
        # Create band names
        band_names = [f"Band {i+1}" for i in range(len(importances))]
        
        # Sort importances
        indices = np.argsort(importances)[::-1]
        
        # Plot feature importances
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [band_names[i] for i in indices], rotation=45)
        plt.xlabel('Bands')
        plt.ylabel('Importance')
        plt.title('Feature Importance')
        
        # Save plot
        importance_path = os.path.join(self.output_directory, "feature_importance.png")
        plt.tight_layout()
        plt.savefig(importance_path)
        plt.close()
        
        # Save to CSV
        importance_df = pd.DataFrame({
            'Band': [band_names[i] for i in indices],
            'Importance': importances[indices]
        })
        csv_path = os.path.join(self.output_directory, "feature_importance.csv")
        importance_df.to_csv(csv_path, index=False)
        
        print(f"Feature importance saved to {importance_path} and {csv_path}")
        
    def create_summary_report(self, start_time=None, end_time=None):
        """Create a comprehensive summary report of the classification"""
        report_path = os.path.join(self.output_directory, "summary_report.html")
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Remote Sensing Classification Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #2c3e50; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .section {{ margin-bottom: 30px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                img {{ max-width: 100%; height: auto; }}
                .metrics {{ display: flex; justify-content: space-between; }}
                .metric-box {{ flex: 1; margin: 10px; padding: 15px; background-color: #f8f9fa; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Remote Sensing Classification Report</h1>
                
                <div class="section">
                    <h2>Classification Overview</h2>
                    <p>Input image: {os.path.basename(self.raster_path)}</p>
                    <p>Image dimensions: {self.shape[0]} x {self.shape[1]} pixels, {self.num_bands} bands</p>
                    <p>Training data: {os.path.basename(self.shapefile_path)}</p>
                    <p>Number of classes: {len(self.class_mapping)}</p>
                    <p>Classes: {', '.join(self.class_mapping.keys())}</p>
                </div>
                
                <div class="section">
                    <h2>Model Information</h2>
                    <p>Model type: Random Forest</p>
                    <p>Number of trees: {self.n_estimators}</p>
                    <p>Max depth: {self.max_depth if self.max_depth else 'None (unlimited)'}</p>
                    <p>Training samples: {len(self.X_train)} pixels</p>
                </div>
                
                <div class="section">
                    <h2>Accuracy Assessment</h2>
                    <div class="metrics">
                        <div class="metric-box">
                            <h3>Overall Accuracy</h3>
                            <p style="font-size: 24px; text-align: center;">{self.accuracy:.4f}</p>
                        </div>
                    </div>
                    
                    <h3>Confusion Matrix</h3>
                    <img src="confusion_matrix.png" alt="Confusion Matrix">
                    
                    <h3>Classification Report</h3>
                    <iframe src="classification_report.txt" style="width: 100%; height: 300px; border: 1px solid #ddd;"></iframe>
                </div>
                
                <div class="section">
                    <h2>Classification Results</h2>
                    <img src="classification_visualization.png" alt="Classification Results">
                </div>
                
                <div class="section">
                    <h2>Feature Importance</h2>
                    <img src="feature_importance.png" alt="Feature Importance">
                </div>
                
                <div class="section">
                    <h2>Processing Information</h2>
                    <p>Processing date: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p>Processing time: {time.strftime('%H:%M:%S', time.gmtime(end_time - start_time)) if start_time and end_time else 'N/A'}</p>
                    <p>Output directory: {os.path.abspath(self.output_directory)}</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        with open(report_path, 'w') as f:
            f.write(html_content)
            
        print(f"Summary report saved to {report_path}")

def main():
    """Main function to run the classification pipeline"""
    # Set these paths according to your data
    raster_path = "F:/chrome/2024_S2_allbands/subset_prj.tif"
    training_samples = "F:/chrome/2024_S2_allbands/validation/validation.shp"
    output_dir = "F:/chrome/2024_S2_allbands/cls_results"
    model_name = "rf_model"
    classified_img_name = "classified_image"
    
    # Set processing parameters
    chunk_size = 2000  # Size of chunks for processing large images
    sample_rate = 0.3  # Sample 10% of pixels from each polygon
    test_size = 0.3  # Fraction of data to use for validation
    n_trees = 100      # Number of trees in random forest
    
    # Initialize classifier
    classifier = LargeRasterClassifier(
        raster_path=raster_path,
        shapefile_path=training_samples,
        output_directory=output_dir,
        model_name=model_name,
        classified_img_name=classified_img_name,
        test_size=test_size,
        chunk_size=chunk_size,
        sample_rate=sample_rate,
        n_estimators=n_trees
    )
    
    # Run the classification pipeline
    print("Starting classification pipeline...")
    print_memory_usage("Before processing:")
    st = time.time()
    classifier.read_training_data()
    classifier.sample_from_polygons()
    classifier.train_model()
    classified_path = classifier.classify_image()
    classifier.evaluate_accuracy()
    classifier.visualize_results(classified_path)
    classifier.feature_importance()
    classifier.create_summary_report(st, time.time())
    print("Classification pipeline completed successfully!")

if __name__ == "__main__":
    main()