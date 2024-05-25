from ..tools import check_file_path, filename_to_timestamp
import os 
import numpy as np
from PIL import Image
class DatasetAnalyser:
    def __init__(self, dataset_path):
        self.dataset_path = check_file_path(dataset_path)
        self.folder = None
    def __iter__(self):
        folder_path = check_file_path(self.dataset_path, self.folder)
        for mask_path in os.listdir(check_file_path(folder_path,'mask')):
            # data = {'mask': None, 'rgb': None, 'depth': None, 'confidence': None,'timestamp': None}
            mask = np.load(check_file_path(folder_path,'mask',mask_path))
            timestamp = filename_to_timestamp(mask_path)
            rgb = np.array(Image.open(check_file_path(folder_path, 'rgb', timestamp + '.jpg')))
            depth = np.array(Image.open(check_file_path(folder_path, 'depth', timestamp + '.png')))
            # if os.path.exists(check_file_path(folder_path, 'confidence', timestamp + '.npy')):
            #     confidence = np.load(check_file_path(folder_path, 'confidence', timestamp + '.npy'))
            yield {'mask': mask, 'rgb': rgb, 'depth': depth, 'timestamp': timestamp}

    def __call__(self, folder):
        self.folder =folder
        return self