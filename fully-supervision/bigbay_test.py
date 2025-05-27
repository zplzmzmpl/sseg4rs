import ttach as tta
import multiprocessing.pool as mpp
import multiprocessing as mp
import time
from train_supervision import *
import argparse
from pathlib import Path
import numpy as np
import torch
import os
import random
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import rasterio
from geoseg.datasets.bigbay_dataset import CLASSES, PALETTE


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def rgb_to_label(rgb_mask):
    """Convert RGB mask to class indices."""
    h, w = rgb_mask.shape[0], rgb_mask.shape[1]
    mask = np.zeros(shape=(h, w), dtype=np.uint8)
    
    for idx, color in enumerate(PALETTE):
        # Check if pixel colors match the palette color
        # Using a small tolerance to handle potential minor variations in TIF encoding
        color_match = np.logical_and.reduce([
            np.abs(rgb_mask[:,:,0] - color[0]) < 5,
            np.abs(rgb_mask[:,:,1] - color[1]) < 5,
            np.abs(rgb_mask[:,:,2] - color[2]) < 5
        ])
        mask[color_match] = idx
    
    return mask

def read_tif(file_path):
    """Read TIF file and return numpy array."""
    with rasterio.open(file_path) as src:
        if src.count == 3:  # RGB image
            # Read all bands and transpose to HWC format
            img = np.dstack([src.read(i+1) for i in range(3)])
            return img
        elif src.count == 1:  # Single band image
            return src.read(1)
        else:
            raise ValueError(f"Unsupported number of bands: {src.count} in {file_path}")
            
    return None


def label2rgb(mask):
    """Convert class indices to RGB mask based on the palette."""
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    
    for idx, color in enumerate(PALETTE):
        mask_rgb[mask == idx] = color
        
    return mask_rgb


def img_writer(inp):
    (mask, mask_id, rgb) = inp
    if rgb:
        # Save as TIF with the RGB color scheme
        mask_name_tif = mask_id + '.tif'
        mask_rgb = label2rgb(mask)
        
        # Create a TIF file with RGB values
        height, width, bands = mask_rgb.shape
        with rasterio.open(
            mask_name_tif, 
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=bands,
            dtype=mask_rgb.dtype
        ) as dst:
            for i in range(bands):
                dst.write(mask_rgb[:, :, i], i+1)
    else:
        # Save as TIF with class indices
        mask_tif = mask.astype(np.uint8)
        mask_name_tif = mask_id + '.tif'
        
        # Create a single-band TIF file with class values
        height, width = mask_tif.shape
        with rasterio.open(
            mask_name_tif, 
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=mask_tif.dtype
        ) as dst:
            dst.write(mask_tif, 1)


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, required=True, help="Path to config")
    arg("-o", "--output_path", type=Path, help="Path where to save resulting masks.", required=True)
    arg("-t", "--tta", help="Test time augmentation.", default=None, choices=[None, "d4", "lr"])
    arg("--rgb", help="whether output rgb images", action='store_true')
    return parser.parse_args()


def process_label_input(label_input):
    """Process RGB label inputs to convert them to class indices."""
    if len(label_input.shape) == 4 and label_input.shape[1] == 3:
        # If input is [batch, 3, H, W], convert to [batch, H, W, 3]
        label_input = label_input.permute(0, 2, 3, 1).cpu().numpy()
        batch_size = label_input.shape[0]
        processed_labels = np.zeros((batch_size, label_input.shape[1], label_input.shape[2]), dtype=np.int64)
        
        for i in range(batch_size):
            processed_labels[i] = rgb_to_label(label_input[i])
        
        return torch.from_numpy(processed_labels)
    elif len(label_input.shape) == 3 and label_input.shape[0] == 3:
        # If input is [3, H, W], convert to [H, W, 3]
        label_rgb = label_input.permute(1, 2, 0).cpu().numpy()
        processed_label = rgb_to_label(label_rgb)
        return torch.from_numpy(processed_label).unsqueeze(0)
    
    return label_input

class TifDataset:
    """Custom dataset class for TIF files if needed to integrate with existing code.
    This is a placeholder in case you need to implement a custom dataset loader."""
    
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.transform = transform
        self.img_files = sorted(list(self.img_dir.glob('*.tif')))
        
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        img_name = img_path.stem
        img_id = img_name.replace('_image', '')
        label_path = self.label_dir / f"{img_id}_label.tif"
        
        # Read image and label as TIF files
        img = read_tif(str(img_path))
        label = read_tif(str(label_path))
        
        # Convert label from RGB to class indices
        if len(label.shape) == 3:  # If RGB
            label = rgb_to_label(label)
        
        # Apply any transforms
        if self.transform:
            transformed = self.transform(image=img, mask=label)
            img = transformed['image']
            label = transformed['mask']
        
        return {
            'img': img,
            'gt_semantic_seg': label,
            'img_id': img_name
        }


def main():
    args = get_args()
    seed_everything(42)

    config = py2cfg(args.config_path)
    args.output_path.mkdir(exist_ok=True, parents=True)

    # Update config with custom classes if needed
    config.classes = CLASSES
    config.num_classes = len(CLASSES)
    # Add the palette to the config if needed
    config.palette = PALETTE

    model = Supervision_Train.load_from_checkpoint(
        os.path.join(config.weights_path, config.test_weights_name + '.ckpt'), config=config)
    model.cuda()
    model.eval()
    evaluator = Evaluator(num_class=config.num_classes)
    evaluator.reset()
    
    if args.tta == "lr":
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.VerticalFlip()
            ]
        )
        model = tta.SegmentationTTAWrapper(model, transforms)
    elif args.tta == "d4":
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.VerticalFlip(),
                # tta.Rotate90(angles=[90]),
                tta.Scale(scales=[0.75, 1.0, 1.25, 1.5], interpolation='bicubic', align_corners=False)
            ]
        )
        model = tta.SegmentationTTAWrapper(model, transforms)

    # Use the dataset from config
    # If you need to override with custom TifDataset, uncomment below
    # test_dataset = TifDataset(img_dir=config.test_img_dir, label_dir=config.test_label_dir, transform=config.test_transform)
    # Otherwise use the one from config
    test_dataset = config.test_dataset

    with torch.no_grad():
        test_loader = DataLoader(
            test_dataset,
            batch_size=2,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )
        results = []
        for input in tqdm(test_loader):
            # raw_prediction NxCxHxW
            raw_predictions = model(input['img'].cuda())

            image_ids = input["img_id"]
            masks_true = input['gt_semantic_seg']
            
            # Process RGB labels if needed
            if len(masks_true.shape) == 4 and masks_true.shape[1] == 3:
                masks_true = process_label_input(masks_true)

            raw_predictions = nn.Softmax(dim=1)(raw_predictions)
            predictions = raw_predictions.argmax(dim=1)

            for i in range(raw_predictions.shape[0]):
                mask = predictions[i].cpu().numpy()
                evaluator.add_batch(pre_image=mask, gt_image=masks_true[i].cpu().numpy())
                mask_name = image_ids[i]
                results.append((mask, str(args.output_path / mask_name), args.rgb))
                
    iou_per_class = evaluator.Intersection_over_Union()
    f1_per_class = evaluator.F1()
    OA = evaluator.OA()
    
    for class_name, class_iou, class_f1 in zip(config.classes, iou_per_class, f1_per_class):
        print('F1_{}:{}, IOU_{}:{}'.format(class_name, class_f1, class_name, class_iou))
    
    print('F1:{}, mIOU:{}, OA:{}'.format(
        np.nanmean(f1_per_class[:-1]), 
        np.nanmean(iou_per_class[:-1]), 
        OA)
    )
    
    t0 = time.time()
    mpp.Pool(processes=mp.cpu_count()).map(img_writer, results)
    t1 = time.time()
    img_write_time = t1 - t0
    print('images writing spends: {} s'.format(img_write_time))


if __name__ == "__main__":
    main()