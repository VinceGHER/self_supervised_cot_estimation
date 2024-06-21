import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

import os
import cv2

class COTDataset(Dataset):

    def __init__(self, root_dir, transform_input=None, transform_common=None, config=None, confidence=False):

        """

        Args:

            root_dir (string): Directory with all the images and masks.

            transform (callable, optional): Optional transform to be applied on a sample.

        """
        self.root_dir = root_dir
        self.transform_input = transform_input
        self.transform_common = transform_common

        self.rgb_files = self.find_files(root_dir, "rgb")
        self.mask_files = self.find_files(root_dir, "mask")
        self.segmentation_files = self.find_files(root_dir, "segmentation")
        self.depth_files = self.find_files(root_dir, "depth")


        if confidence:
            self.confidence_files = self.find_files(root_dir, "confidence")
        self.config = config
        self.confidence = confidence

        assert len(self.rgb_files) == len(self.mask_files) == len(self.segmentation_files) == len(self.depth_files), "Number of files in each folder must be the same"

    def find_files(self, root_dir, name):
        return [os.path.join(root_dir, name, file) for file in sorted(os.listdir(os.path.join(root_dir, name)))]

    def __len__(self):
        return len(self.mask_files)



    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

    
        rgb_path = self.rgb_files[idx]
        mask_path = self.mask_files[idx]


        # Load images and masks

        image = Image.open(rgb_path).convert("RGB")
        mask = np.load(mask_path)
        seg = np.load(self.segmentation_files[idx])

        if self.confidence:
            confidence = np.load(self.confidence_files[idx])
            confidence = torch.from_numpy(confidence).float()
            confidence = confidence.unsqueeze(0)

        depth = cv2.imread(self.depth_files[idx], cv2.IMREAD_UNCHANGED)

        depth = np.array(depth)/self.config['depth']['depth_to_meters']
        depth[depth > self.config['depth']['max_depth']] = self.config['depth']['max_depth']
        depth[depth <= self.config['depth']['min_depth']] = self.config['depth']['max_depth']

        # normalize the depth
        depth = (depth - self.config['depth']['min_depth']) / (self.config['depth']['max_depth'] - self.config['depth']['min_depth'])


        mask = torch.from_numpy(mask).float()
        seg = torch.from_numpy(seg).float()
        depth = torch.from_numpy(depth).float()


        mask[mask <= 0] = 0
        mask[mask > self.config['cot']['wall_cot']] = self.config['cot']['wall_cot']

        mask = mask.unsqueeze(0)
        seg = seg.unsqueeze(0)
        depth = depth.unsqueeze(0)

        # combine mask and image

        if self.transform_input:
            image = self.transform_input(image)

        if self.confidence:
            combined = torch.cat((image, mask, seg, depth,confidence), dim=0)
        else:
            combined = torch.cat((image, mask, seg, depth), dim=0)

        if self.transform_common:
            combined = self.transform_common(combined)

        image = combined[:3, :, :]
        mask = combined[3, :, :].unsqueeze(0)
        seg = combined[4, :, :].unsqueeze(0)
        depth = combined[5, :, :].unsqueeze(0)

        if self.confidence:
            confidence = combined[6, :, :].unsqueeze(0)

        image = image.permute(0, 1, 2)
        mask = mask.permute(0, 1, 2)
        seg = seg.permute(0, 1, 2)
        depth = depth.permute(0, 1, 2)
 

        if self.confidence:
            confidence = confidence.permute(0, 1, 2)


        sample = {
            'image': image, 
            'mask': mask, 
            'seg': seg, 
            'depth':depth,
            'timestamp':os.path.basename(mask_path)[:-4],
            'confidence': torch.zeros_like(mask) if not self.confidence else confidence,
        }


        return sample

