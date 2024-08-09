import glob
import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
import torchvision
from tqdm import tqdm
from transformers import pipeline
import multiprocessing as mp
# import sys
# sys.path.append("modules/full_segment_anything")

# from generate_mask import show_points
# from modules.full_segment_anything.utils.utils import show_lbk_masks
from src.experiment import Experiment
from src.processing.processor import Processor
from src.tools import check_file_path, clear_folder, filename_to_timestamp
# from modules.full_segment_anything.build_sam import sam_model_registry
# from modules.full_segment_anything.utils.amg import build_all_layer_point_grids
from torchvision.transforms import functional as F
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from scipy.ndimage import grey_closing,grey_opening
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_masks_on_image(raw_image, masks):
  plt.imshow(np.array(raw_image))
  ax = plt.gca()
  ax.set_autoscale_on(False)
  for mask in masks:
      show_mask(mask['segmentation'], ax=ax, random_color=True)
  plt.axis("off")
  plt.show()

def run_model_on_half_dataset(items):
    files, projected_path,rgb_path, mask_path,seg_path,device,params, score_thresh,kernel_size,max_cost_transport = items
    with torch.inference_mode():
        sam = sam_model_registry["vit_h"](checkpoint="models/sam_vit_h_4b8939.pth")
        sam.to(device=device)

        predictor = SamAutomaticMaskGenerator(sam,**params)
        for file in (pbar := tqdm(files)):
            pbar.set_description(f"Processing {file}")
            image_projected = np.load(check_file_path(projected_path, file))

            green_mask = np.logical_and(image_projected < max_cost_transport * 0.8, image_projected > 0)
            area_green_mask = np.sum(green_mask)/(image_projected.shape[0]*image_projected.shape[1])
            if  area_green_mask > 0.8:
                print("green mask too big, discarding mask", file,area_green_mask)
                continue
            if area_green_mask < 0.05:
                print("green mask too small, discarding mask", file,area_green_mask)
                continue
            red_mask = (image_projected > max_cost_transport * 0.8 ) & (image_projected < max_cost_transport)
            red_mask = grey_opening(red_mask, size=(2,2))

            image_path = check_file_path(rgb_path, filename_to_timestamp(file) + ".jpg")
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            masks = predictor.generate(image)

            segmented_image = np.zeros((image.shape[0],image.shape[1])).astype(np.float32) 
            for i,mask in enumerate(masks):
                segmented_image[mask['segmentation']] = i+1
            np.save(os.path.join(seg_path, file), segmented_image)
            keep_masks = []
            mask_image = np.zeros((image.shape[0],image.shape[1])).astype(np.float32) 
            for mask in masks:
                if (mask["segmentation"].shape[0] < 0):
                    continue
                path_score = score_mask_cost_mask(mask['segmentation'], green_mask)
                wall_score = score_mask_cost_mask(mask['segmentation'], red_mask)

                if path_score > 0.2 and path_score > wall_score:
                    keep_masks.append(mask)
                    cot = np.mean(image_projected[(mask['segmentation']) & (green_mask)])
                    mask_image[mask["segmentation"]] = cot
                    continue
                if wall_score > 0.2 and wall_score > path_score:
                    keep_masks.append(mask)
                    mask_image[mask["segmentation"]] = max_cost_transport
                    continue

            score_mask_green = compute_green_mask_score(mask_image, max_cost_transport)
            score_mask_red = compute_red_mask_score(mask_image, max_cost_transport)

            if score_mask_green < 0.05 or score_mask_green > 0.8:
                print("score_mask_green too low/high, discarding mask", file, score_mask_green)
                continue
            if score_mask_red > 0.7:
                print("score_mask_red too high, discarding mask", file, score_mask_red)
                continue

            # mask_image= cv2.morphologyEx(mask_image, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)))
            # mask_image = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2BGR)
            # cv2.imwrite(os.path.join(mask_path, file), mask_image)
            mask_image = grey_closing(mask_image, size=(kernel_size,kernel_size))
            np.save(os.path.join(mask_path, file), mask_image)
def score_mask_cost_mask(mask, projected):
    r = np.sum(np.logical_and(mask, projected))/np.sum(mask)
    return r
def compute_green_mask_score(mask_image,max_cost_transport):
    green_mask = np.logical_and(mask_image < max_cost_transport * 0.8, mask_image > 0)
    area_green_mask = np.sum(green_mask)/(mask_image.shape[0]*mask_image.shape[1])
    return area_green_mask

def compute_red_mask_score(mask_image,max_cost_transport):
    red_mask = (mask_image > max_cost_transport * 0.8)
    area_red_mask = np.sum(red_mask)/(mask_image.shape[0]*mask_image.shape[1])
    return area_red_mask

class GenerateMask(Processor):
    def __init__(self,params_sam:dict,score_thresh:float,kernel_size:int,max_cost_transport:float):
        super().__init__()
        self.params_sam = params_sam
        self.score_thresh = score_thresh
        self.kernel_size = kernel_size
        self.max_cost_transport = max_cost_transport
    def prepare_image(self,image):
        # Get the dimensions of the original image
        # height, width = image.shape[:2]
    
        # # Calculate padding to make the image square
        # # Since width > height, we need to pad the top and bottom
        # top_padding = (width - height) // 2
        # bottom_padding = width - height - top_padding
        
        # # Create padding
        # # cv2.copyMakeBorder(image, top, bottom, left, right, borderType, value for border)
        # # We're padding with white color, which is [255, 255, 255] in BGR format
        # image = cv2.copyMakeBorder(image, top_padding, bottom_padding, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        trans = torchvision.transforms.Compose([torchvision.transforms.Resize((768,1024))])
       
        image = torch.as_tensor(image).cuda()
        return trans(image.permute(2, 0, 1))

    def process_multicore(self,exp: Experiment):

        clear_folder(exp.mask_path)
        clear_folder(exp.seg_path)
        projected_path = exp.projected_path
        rgb_path = exp.rgb_path
        mask_path = exp.mask_path
        seg_path = exp.seg_path

        projected_files = os.listdir(projected_path)
        projected_files_1 = projected_files[:len(projected_files)//4]
        projected_files_2 = projected_files[len(projected_files)//4:len(projected_files)//2]
        projected_files_3 = projected_files[len(projected_files)//2:3*len(projected_files)//4]
        projected_files_4 = projected_files[3*len(projected_files)//4:]

        items=[
            (projected_files_1, projected_path, rgb_path, mask_path,seg_path,"cuda:0", self.params_sam, self.score_thresh,self.kernel_size,self.max_cost_transport),
            (projected_files_2, projected_path, rgb_path, mask_path,seg_path,"cuda:0", self.params_sam, self.score_thresh,self.kernel_size,self.max_cost_transport),
            (projected_files_3, projected_path, rgb_path, mask_path,seg_path,"cuda:1", self.params_sam, self.score_thresh,self.kernel_size,self.max_cost_transport),
            (projected_files_4, projected_path, rgb_path, mask_path,seg_path,"cuda:1", self.params_sam, self.score_thresh,self.kernel_size,self.max_cost_transport)
        ]

        # run_model_on_half_dataset(items[0])
    

        with mp.Pool(processes=len(items)) as pool:
            for chunk_result in tqdm(pool.imap_unordered(run_model_on_half_dataset, items), total=len(items)):
                pass
        
        # batch_size = 16
        # projected_files = os.listdir(projected_path)
        # # input_point =self.create_uniform_grid(640,480,512).cuda()
        # input_point = torch.as_tensor(build_all_layer_point_grids(16, 0, 1)[0] * 1024, dtype=torch.int64).cuda()
        # input_label = torch.tensor([1 for _ in range(input_point.shape[0])]).cuda()
        # with tqdm(total=len(projected_files), desc="Processing in chunks") as pbar:
        #     for i in range(0, len(projected_files), batch_size):
        #         batch = projected_files[i:i+batch_size]
        #         image_projected = [cv2.imread(check_file_path(projected_path,file)) for file in batch]
        #         image_projected = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in image_projected]
        #         green_masks = [cv2.inRange(image, (0, 150, 0), (100, 255, 20)) for image in image_projected]
        #         green_locations = [np.where(mask == 255) for mask in green_masks]

        #         image_paths = [check_file_path(rgb_path, filename_to_timestamp(file) + ".jpg") for file in batch]
        #         images = [Image.open(image_path) for image_path in image_paths]
        #         # images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images]
        #         # fitler out images with no green pixels
        #         images = [image for image, green_location in zip(images, green_locations) if len(green_location[0]) > 0]
                # batched_input = [
                #     {
                #         'image': self.prepare_image(x),
                #         'point_coords': input_point,
                #         'point_labels': input_label,
                #         'original_size': x.shape[1:]
                #     } for x in images
                # ]
                # outputs = sam(images, points_per_batch=64)

                # # masks = outputs[0]["masks"]
                # # show_masks_on_image(images[0], masks)
                # # refined_masks = sam.individual_forward(batched_input, multimask_output=True)
                # # plt.figure(figsize=(5,5))
                # # plt.imshow(batched_input[0]['image'].permute(1,2,0).cpu().numpy())
                # # show_lbk_masks(refined_masks[0].cpu().numpy(), plt)
                # # # show_points(input_point.cpu().numpy(), input_label.cpu().numpy(), plt.gca())
                # # plt.title(f"[Full Grid] LBK Refined Mask", fontsize=18)
                # # plt.axis('on')
                # # plt.show()

                # green_masks = np.array(green_masks)
                # for i,masks in enumerate(outputs):
                #     masks= np.array(masks['masks'])
                #     keep_masks = []
                #     green_mask = green_masks[i]
                #     file = batch[i]
                #     mask_image = np.zeros((green_mask.shape[0],green_mask.shape[1])).astype(np.uint8) 
                #     for mask in masks:
                #         if self.check_if_mask_is_in_projected(mask, green_mask):
                #             keep_masks.append(mask)
                #             mask_image[mask] = 255
                #     score_mask = self.compute_score(mask_image, green_mask)
                #     if (score_mask < score_thresh):
                #         print("score too low, discarding mask", file, score_mask)
                #         pbar.update(1)
                #         continue
                #     mask_image= cv2.morphologyEx(mask_image, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)))
                #     mask_image = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2BGR)
                #     cv2.imwrite(os.path.join(mask_path, file), mask_image)
                #     pbar.update(1)
                
            
    def process(self,exp: Experiment):

        clear_folder(exp.mask_path)
        clear_folder(exp.seg_path)
        projected_path = exp.projected_path
        rgb_path = exp.rgb_path
        mask_path = exp.mask_path
        seg_path = exp.seg_path

        projected_files = [os.listdir(projected_path)[100:160]]

        items=[]
        for projected_file in projected_files:
            items.append((projected_file, projected_path, rgb_path,mask_path,seg_path,"cuda:0", self.params_sam, self.score_thresh,self.kernel_size,self.max_cost_transport))
        

            # run_model_on_half_dataset(items[0])
        for item in items:
            run_model_on_half_dataset(item)
        