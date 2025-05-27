import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import matplotlib.pyplot as plt
import albumentations as albu
from PIL import Image
import random

# Define your classes and palette
CLASSES = ('paddy_field', 'dry_land', 'forest', 'shrubbery', 'sparse_forests', 
           'other_forest', 'high_mulched_grass', 'medium_mulched_grass', 'low_mulched_grass', 
           'canals', 'pools', 'reservoir_pits', 'tideland', 'beach', 'town', 'rural', 
           'other_builds', 'sandland', 'swamp', 'bare_land')

PALETTE = [
    [0, 255, 0],     # paddy_field
    [124, 252, 0],   # dry_land
    [34, 139, 34],   # forest
    [50, 205, 50],   # shrubbery
    [154, 205, 50],  # sparse_forests
    [144, 238, 144], # other_forest
    [255, 255, 0],   # high_mulched_grass
    [255, 215, 0],   # medium_mulched_grass
    [255, 250, 205], # low_mulched_grass
    [0, 0, 255],     # canals
    [135, 206, 235], # pools
    [173, 216, 230], # reservoir_pits
    [255, 165, 0],   # tideland
    [255, 255, 255], # beach
    [255, 0, 0],     # town
    [255, 105, 180], # rural
    [255, 160, 122], # other_builds
    [255, 228, 196], # sandland
    [46, 139, 87],   # swamp
    [139, 69, 19]    # bare_land
]

# CLASSES = ('cultivation', 'forest','grasses', 'water', 'buildings', 'unused_land')

# PALETTE = [
#     [0, 255, 0],
#     [34, 139, 34],
#     [255, 255, 0],
#     [0, 0, 255],
#     [255, 0, 0],
#     [255, 228, 196]
# ]

ORIGIN_IMG_SIZE = (1024, 1024)
INPUT_IMG_SIZE = (1024, 1024)
TEST_IMG_SIZE = (1024, 1024)

def get_training_transform():
    """Define transformations for training images"""
    train_transform = [
        albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.Normalize()
    ]
    return albu.Compose(train_transform)

def get_val_transform():
    """Define transformations for validation/testing images"""
    val_transform = [
        albu.Normalize()
    ]
    return albu.Compose(val_transform)

def train_aug(img, mask):
    """Apply augmentation to training images and masks"""
    # Convert PIL images to numpy arrays if they are not already
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    if not isinstance(mask, np.ndarray):
        mask = np.array(mask)
    
    # Apply augmentations
    aug = get_training_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    
    return img, mask

def val_aug(img, mask):
    """Apply augmentation to validation/testing images and masks"""
    # Convert PIL images to numpy arrays if they are not already
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    if not isinstance(mask, np.ndarray):
        mask = np.array(mask)
    
    # Apply normalization
    aug = get_val_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    
    return img, mask

def rgb_to_label(mask_rgb):
    """
    Convert RGB mask to class indices
    
    Args:
        mask_rgb: RGB mask image as numpy array [H, W, 3]
        
    Returns:
        mask_label: single channel mask with class indices [H, W]
    """
    mask_label = np.zeros(mask_rgb.shape[:2], dtype=np.uint8)
    
    # Convert mask_rgb to the same format as PALETTE for comparison
    if mask_rgb.dtype == np.float32:
        mask_rgb = (mask_rgb * 255).astype(np.uint8)
    
    for i, rgb in enumerate(PALETTE):
        # Create mask where current class color is present
        class_mask = np.all(mask_rgb == rgb, axis=-1)
        # Set the corresponding pixels in the label mask to the class index
        mask_label[class_mask] = i
    
    return mask_label

class LandUseDataset(Dataset):
    def __init__(self, data_root='data/landuse_dataset', mode='train', 
                 img_dir='image', mask_dir='label',
                 img_suffix='.tif', mask_suffix='.tif', 
                 transform=None, mosaic_ratio=0.0,
                 img_size=ORIGIN_IMG_SIZE):
        """
        Args:
            data_root: path to dataset directory
            mode: 'train', 'val', or 'test'
            img_dir: directory name containing images
            mask_dir: directory name containing masks
            img_suffix: file extension for images
            mask_suffix: file extension for masks
            transform: function to apply transformations
            mosaic_ratio: probability of using mosaic augmentation
            img_size: input image size (height, width)
        """
        self.data_root = data_root
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix
        self.mode = mode
        self.mosaic_ratio = mosaic_ratio
        self.img_size = img_size
        
        # Set default transform if not provided
        if transform is None:
            self.transform = train_aug if mode == 'train' else val_aug
        else:
            self.transform = transform
            
        # Get image IDs
        self.img_ids = self.get_img_ids(self.data_root, self.img_dir, self.mask_dir)
        
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, index):
        p_ratio = random.random()
        # Use mosaic augmentation with probability mosaic_ratio (only during training)
        if p_ratio > self.mosaic_ratio or self.mode != 'train':
            img, mask = self.load_img_and_mask(index)
            if self.transform:
                img, mask = self.transform(img, mask)
        else:
            img, mask = self.load_mosaic_img_and_mask(index)
            if self.transform:
                img, mask = self.transform(img, mask)
        
        # Convert to torch tensors
        img = torch.from_numpy(img).permute(2, 0, 1).float()  # [C, H, W]
        mask = torch.from_numpy(mask).long()  # [H, W]
        
        img_id = self.img_ids[index]
        results = dict(img_id=img_id, img=img, gt_semantic_seg=mask)
        return results
    
    def get_img_ids(self, data_root, img_dir, mask_dir):
        """Get list of image IDs from filenames"""
        img_path = osp.join(data_root, img_dir)
        img_filename_list = sorted([f for f in os.listdir(img_path) if f.endswith(self.img_suffix)])
        
        # Extract image IDs from filenames
        # Assuming the image filename format is something like 'tile_1_1_image.tif'
        # and mask filename is 'tile_1_1_label.tif'
        img_ids = []
        for filename in img_filename_list:
            # Extract base ID without suffix
            img_id = filename.replace('_image', '').replace(self.img_suffix, '')
            
            # Check if corresponding mask exists
            mask_filename = img_id + '_label' + self.mask_suffix
            mask_path = osp.join(data_root, mask_dir, mask_filename)
            
            if os.path.exists(mask_path):
                img_ids.append(img_id)
        
        return img_ids
    
    def load_img_and_mask(self, index):
        """Load image and mask by index"""
        img_id = self.img_ids[index]
        
        # Construct file paths
        img_filename = img_id + '_image' + self.img_suffix
        mask_filename = img_id + '_label' + self.mask_suffix
        
        img_path = osp.join(self.data_root, self.img_dir, img_filename)
        mask_path = osp.join(self.data_root, self.mask_dir, mask_filename)
        
        # Load images
        img = Image.open(img_path).convert('RGB')
        mask_rgb = Image.open(mask_path).convert('RGB')
        
        # Convert RGB mask to class indices
        mask_np = np.array(mask_rgb)
        mask = rgb_to_label(mask_np)
        
        return img, mask
    
    def load_mosaic_img_and_mask(self, index):
        """Load mosaic augmented image and mask"""
        # Select 4 images (current + 3 random)
        indexes = [index] + [random.randint(0, len(self.img_ids) - 1) for _ in range(3)]
        
        # Load the 4 images and masks
        img_a, mask_a = self.load_img_and_mask(indexes[0])
        img_b, mask_b = self.load_img_and_mask(indexes[1])
        img_c, mask_c = self.load_img_and_mask(indexes[2])
        img_d, mask_d = self.load_img_and_mask(indexes[3])
        
        # Convert PIL images to numpy arrays
        img_a, mask_a = np.array(img_a), np.array(mask_a)
        img_b, mask_b = np.array(img_b), np.array(mask_b)
        img_c, mask_c = np.array(img_c), np.array(mask_c)
        img_d, mask_d = np.array(img_d), np.array(mask_d)
        
        # Get dimensions
        w = self.img_size[1]
        h = self.img_size[0]
        
        # Define crop center
        start_x = w // 4
        start_y = h // 4
        # Random coordinates for the splice center
        offset_x = random.randint(start_x, (w - start_x))
        offset_y = random.randint(start_y, (h - start_y))
        
        # Calculate crop sizes for each quadrant
        crop_size_a = (offset_x, offset_y)  # top-left
        crop_size_b = (w - offset_x, offset_y)  # top-right
        crop_size_c = (offset_x, h - offset_y)  # bottom-left
        crop_size_d = (w - offset_x, h - offset_y)  # bottom-right
        
        # Create random crop augmentations
        random_crop_a = albu.RandomCrop(width=crop_size_a[0], height=crop_size_a[1])
        random_crop_b = albu.RandomCrop(width=crop_size_b[0], height=crop_size_b[1])
        random_crop_c = albu.RandomCrop(width=crop_size_c[0], height=crop_size_c[1])
        random_crop_d = albu.RandomCrop(width=crop_size_d[0], height=crop_size_d[1])
        
        # Apply crops
        croped_a = random_crop_a(image=img_a.copy(), mask=mask_a.copy())
        croped_b = random_crop_b(image=img_b.copy(), mask=mask_b.copy())
        croped_c = random_crop_c(image=img_c.copy(), mask=mask_c.copy())
        croped_d = random_crop_d(image=img_d.copy(), mask=mask_d.copy())
        
        # Extract cropped images and masks
        img_crop_a, mask_crop_a = croped_a['image'], croped_a['mask']
        img_crop_b, mask_crop_b = croped_b['image'], croped_b['mask']
        img_crop_c, mask_crop_c = croped_c['image'], croped_c['mask']
        img_crop_d, mask_crop_d = croped_d['image'], croped_d['mask']
        
        # Concatenate crops to form mosaic
        top = np.concatenate((img_crop_a, img_crop_b), axis=1)
        bottom = np.concatenate((img_crop_c, img_crop_d), axis=1)
        img = np.concatenate((top, bottom), axis=0)
        
        top_mask = np.concatenate((mask_crop_a, mask_crop_b), axis=1)
        bottom_mask = np.concatenate((mask_crop_c, mask_crop_d), axis=1)
        mask = np.concatenate((top_mask, bottom_mask), axis=0)
        
        # Ensure memory layout is C-contiguous
        mask = np.ascontiguousarray(mask)
        img = np.ascontiguousarray(img)
        
        return img, mask

# Visualization utility function
def visualize_dataset_sample(dataset, idx=None, figsize=(15, 10)):
    """
    Visualize a sample from the dataset
    
    Args:
        dataset: LandUseDataset instance
        idx: Sample index to visualize. If None, a random sample is chosen
        figsize: Size of the matplotlib figure
    """
    if idx is None:
        idx = random.randint(0, len(dataset) - 1)
    
    sample = dataset[idx]
    img = sample['img'].permute(1, 2, 0).numpy()  # [H, W, C]
    mask = sample['gt_semantic_seg'].numpy()  # [H, W]
    
    # Denormalize image if needed
    if img.max() <= 1:
        img = (img * 255).astype(np.uint8)
    
    # Create a colored mask using the palette
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_idx, color in enumerate(PALETTE):
        colored_mask[mask == class_idx] = color
    
    plt.figure(figsize=figsize)
    
    plt.subplot(1, 2, 1)
    plt.title(f"Image: {sample['img_id']}")
    plt.imshow(img)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("Segmentation Mask")
    plt.imshow(colored_mask)
    plt.axis('off')
    
    # Create legend with class names and colors
    patches = []
    for i, class_name in enumerate(CLASSES):
        color = [c/255.0 for c in PALETTE[i]]  # Normalize color for matplotlib
        patches.append(plt.Rectangle((0, 0), 1, 1, fc=color))
    
    # Add legend outside the plot
    plt.figlegend(patches, CLASSES, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.05))
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage
    dataset = LandUseDataset(data_root='E:/2025/full_landsat_20cls/train', mode='train')
    visualize_dataset_sample(dataset, idx=None)