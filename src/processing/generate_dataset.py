import os 


from typing import List
import datetime
import numpy as np
from PIL import Image

from src.experiment import Experiment
from src.processing.processor import Processor
from src.tools import filename_to_timestamp,check_file_path
import shutil

class GenerateDataset(Processor):
    def __init__(self,folder='datasets'):
        self.folder = folder

    def process(self,exps,split=[0.8,0.1,0.1]):
        if sum(split) < 0.99 or sum(split) > 1.01:
            raise ValueError("Split must sum to 1")
        data_files = []
        for data in exps:
            exp,sections = data


            success = True
            # check if the exp has the necessary files
            for folder in [exp.rgb_path, exp.depth_path, exp.mask_path]:
                if not os.path.exists(folder):
                    print(f"Missing {folder} folder in {exp}")
                    success = False
                    break
            if not success:
                continue
            masks = os.listdir(exp.mask_path)
            masks.sort()
            masks = np.array(masks)
            indexs = []
            for section in sections:
                if section[1] == -1:
                    indexs.extend(list(range(section[0],len(masks))))
                else:
                    indexs.extend(list(range(section[0],section[1])))
            print(indexs)

            for file in masks[indexs]:
                # find the corresponding rgb and depth files
                timestamp_str = filename_to_timestamp(file)
                rgb_file = check_file_path(exp.rgb_path, timestamp_str+'.jpg')
                depth_file = check_file_path(exp.depth_path, timestamp_str+'.png')
                seg_file = check_file_path(exp.seg_path, timestamp_str+'.npy')

                if rgb_file is None or depth_file is None:
                    print(f"Could not find corresponding files for {file}")
                    continue

                # add to data_files in the form of as dict
                data_files.append({
                    "mask": check_file_path(exp.mask_path, file),
                    "rgb": rgb_file,
                    "depth": depth_file,
                    "segmentation":seg_file,
                })
        print("length of data_files: ", len(data_files))

        # split into train, valid and test
        np.random.shuffle(data_files)
        train_end = int(split[0]*len(data_files))
        valid_end = int((split[0]+split[1])*len(data_files))
        train_files = data_files[:train_end]
        valid_files = data_files[train_end:valid_end]
        test_files = data_files[valid_end:]

        # generate a new id dataset folder
        # set datetime to local time

        timestamp_folder = datetime.datetime.now().astimezone().strftime('%Y-%m-%d_%H-%M-%S')
        dataset_folder_path = os.path.join("datasets", "dataset-"+timestamp_folder)
        # save the files into the dataset folder
        for i, files in enumerate([train_files, valid_files, test_files]):
            folder = ["train", "valid", "test"][i]
            for file in files:
                for key, path in file.items():
                    os.makedirs(os.path.join(dataset_folder_path, folder, key), exist_ok=True)
                    shutil.copy(path, os.path.join(dataset_folder_path, folder, key, os.path.basename(path)))

                    # # if jpg convert it to png and save
                    # if path.endswith(".jpg"):
                    #     im = Image.open(os.path.join(dataset_folder_path, folder, key, os.path.basename(path)))
                    #     im.save(os.path.join(dataset_folder_path, folder, key, os.path.basename(path)[:-4]+".png"))
                    #     os.remove(os.path.join(dataset_folder_path, folder, key, os.path.basename(path)))
        print(f"Dataset saved to {dataset_folder_path}")
        print(f"Train: {len(train_files)}, Valid: {len(valid_files)}, Test: {len(test_files)}")
        
