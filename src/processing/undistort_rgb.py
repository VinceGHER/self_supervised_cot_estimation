

from typing import Dict

from tqdm import tqdm

from src.experiment import Experiment
from src.processing.processor import Processor
from src.tools import check_file_path, filename_to_timestamp, load_calib_file

import cv2
import os 

class Undistort_rgb(Processor):
    def __init__(self):
        super().__init__()

    def process(self,exp: Experiment):
        files = os.listdir(exp.rgb_path)
        for file in tqdm(files):
            img = cv2.imread(check_file_path(exp.rgb_path, file))
            calib_path = check_file_path(exp.calib_path, filename_to_timestamp(file) + ".yaml")
            calib = load_calib_file(calib_path)
            K = calib['camera_matrix']['data']
            distortion_coefficients = calib['distortion_coefficients']['data']
            h, w = img.shape[:2]
            # Unnormalize the image
            new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(K, distortion_coefficients, (w, h), 1, (w, h))
            img_new = cv2.undistort(img, K, distortion_coefficients, None, new_camera_matrix)

            cv2.imshow("Original", img)
            cv2.imshow("Undistorted", img_new)
            cv2.waitKey(0)
            
            cv2.imwrite(check_file_path(exp.rgb_undistorted_path, file), img_new)
            
