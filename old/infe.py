from matplotlib import pyplot as plt
import torch
import yaml
import wandb
import numpy as np
import cv2
from master_thesis_v2.src.ml_orchestrator.transforms.transforms_builder import TransformBuilder
from master_thesis_v2.src.models.model_builder import model_builder
from master_thesis_v2.src.tools import check_file_path

class InferenceManager:
    def __init__(self,model_path="vincekillerz/base-traversability-estimation-v2/saved_model:v30"):
        run = wandb.init()
        artifact = run.use_artifact(model_path, type='model')
        artifact_dir = artifact.download()
        # artifact_dir="artifacts/saved_model:v36"
        self.config = yaml.safe_load(open(check_file_path(artifact_dir, 'config.yaml')))
        self.model_weight_path = check_file_path(artifact_dir, 'trained_model.pth')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('model_weight_path:', self.model_weight_path)
        self.transform_builder = TransformBuilder(self.config['transforms'])

        self.transform_common = self.transform_builder.build_transform_common_validation()
        self.transform_inputs = self.transform_builder.build_transforms_inputs_validation()

        model = model_builder(self.config['model_builder'])
        model.load_state_dict(torch.load(self.model_weight_path,map_location=self.device))

        model.to(self.device)
        model.eval()  # Set the model to evaluation mode
        self.model = model
        print("Model loaded successfully")


    def predict(self, color_image, depth_image,K,D):
        # undistort the image
        color_image = self.undistort_image(color_image, K, D)
        config = self.config
        d = np.nan_to_num(depth_image, nan=0)
        print(np.max(d),np.mean(d))
        depth = np.array(depth_image)/config['depth']['depth_to_meters']
        
        depth = np.nan_to_num(depth, nan=config['depth']['max_depth'])

        
        depth[depth > config['depth']['max_depth']] = config['depth']['max_depth']
        depth[depth <= config['depth']['min_depth']] = config['depth']['max_depth']
        depth_numpy = (depth - config['depth']['min_depth']) / (config['depth']['max_depth'] - config['depth']['min_depth'])
        print(np.max(depth_numpy),np.min(depth_numpy))
        cmap = plt.get_cmap('nipy_spectral')
        colored_array = cmap(depth_numpy)  # This returns a 480x640x4 array with RGBA values

        # Convert the RGBA values to RGB (ignore the alpha channel)
        # output_color = (colored_array[:, :, :3] * 255).astype(np.uint8)  # Convert to uint8 type
        # output_color =cv2.cvtColor(output_color, cv2.COLOR_RGB2BGR)
        # cv2.imshow('depth',output_color)
        # cv2.waitKey(0)
        depth = torch.from_numpy(depth_numpy).float()
        depth = depth.unsqueeze(0)
        image = self.transform_inputs(color_image)
        combined = torch.cat((image, depth), dim=0)
        combined = self.transform_common(combined)

        image = combined[:3, :, :]
        depth = combined[3, :, :].unsqueeze(0)
        image = image.permute(0, 1, 2)
        depth = depth.permute(0, 1, 2)
       
        # Dummy model forward
        # Add dimension for batch
        color_image_batch = image.unsqueeze(0).to(self.device)
        depth_image_batch = depth.unsqueeze(0).to(self.device)
        # print(color_image_batch)
        # print(depth_image_batch)


        with torch.no_grad():
            output = self.model(color_image_batch, depth_image_batch)
            output = output[0][0].cpu().numpy()
            # print(output)

                # Apply the colormap nipy_spectral
        cmap = plt.get_cmap('nipy_spectral')
        outnormalized = (output - 0) / (8-0)
        colored_array = cmap(outnormalized)  # This returns a 480x640x4 array with RGBA values

        # Convert the RGBA values to RGB (ignore the alpha channel)
        output_color = (colored_array[:, :, :3] * 255).astype(np.uint8)  # Convert to uint8 type
        return output,output_color