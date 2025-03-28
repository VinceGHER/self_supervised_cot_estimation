import time
import torch
import yaml
import wandb
from torcheval.metrics import R2Score, MeanSquaredError
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
from thop import profile
from mpl_toolkits.axes_grid1 import make_axes_locatable
from src.ml_orchestrator.dataset import COTDataset
from src.ml_orchestrator.loss.loss_builder import loss_builder
from src.ml_orchestrator.transforms.transforms_builder import TransformBuilder
from torch.utils.data import DataLoader

from .metrics.MAE_metric import MAEMetric
from src.models.model_builder import model_builder
from ..tools import check_file_path

class Tester():
    def __init__(self,names) -> None:
        self.run = wandb.init()
        self.models = []
        self.idx = [10,20,40,12]
        self.names = names
        for name in names:
            
            artifact = self.run.use_artifact(name[0], type='model')
            artifact_dir = artifact.download()
            config = yaml.safe_load(open(check_file_path(artifact_dir, 'config.yaml')))
            model_path = check_file_path(artifact_dir, 'trained_model.pth')

            self.models.append({
                'config': config,
                'model_path': model_path,
                'name': name[1]
            })


    def get_test_dataset(self,config):
        transform_builder = TransformBuilder(config['transforms'])
        dataset_folder = check_file_path("datasets",config['ml_orchestrator']['dataset_name'])
        test_dataset = COTDataset(
            confidence=False,
            root_dir=check_file_path(dataset_folder,"test_manually_labelled"), 
            transform_input=transform_builder.build_transforms_inputs_validation(),
            transform_common=transform_builder.build_transform_common_validation(), 
            config=config, 
        )
        print("Test dataset length: ",len(test_dataset))
        return test_dataset
    
    def test(self,index):
        config = self.models[index]['config']
        model_path = self.models[index]['model_path']
        

        test_dataset = self.get_test_dataset(config)    
        test_loader = DataLoader(
            test_dataset, 
            batch_size=1, 
            shuffle=False,
            pin_memory=True,
            num_workers=0,
        )
        device = config['ml_orchestrator']['device']
        model = model_builder(config)
        model.to(device)

        # load model
        print(model_path)
        model.load_state_dict(torch.load(model_path))

    
        r2_score = R2Score(device=device)
        mse = MeanSquaredError(device=device)
        mae = MAEMetric()
        model.eval()


        r2_scores=[]
        mse_scores=[]
        mae_scores=[]
        inference_times_scores=[]
        with torch.no_grad():
            for batch in test_loader:
                inputs = batch['image'].to(device)
                depth = batch['depth'].to(device)
                masks = batch['mask'].to(device)

                start= time.time()
                outputs = model(inputs,depth,None,None)
                
                torch.cuda.synchronize()
                end = time.time()
                
                inference_time = (end-start)*1000
                inference_times_scores.append(inference_time)
                print("Inference time: ",inference_time)
                for i in range(len(outputs)):
                    r2_score.reset()   
                    mse.reset()    
                    mae.reset()

                    flat__outputs = outputs[i].reshape(-1)
                    flat_masks = masks[i].reshape(-1)
        
                    r2_score.update(flat__outputs, flat_masks)
                    mse.update(flat__outputs, flat_masks)  
                    mae.update(flat__outputs, flat_masks)

                    r2_scores.append(r2_score.compute().cpu().numpy())    
                    mse_scores.append(mse.compute().cpu().numpy())
                    mae_scores.append(mae.compute())


                

        self.models[index]['r2_score'] =r2_scores
        self.models[index]['mse_score'] = mse_scores
        self.models[index]['mae_score'] = mae_scores
        self.models[index]['inference_times'] = inference_times_scores





        # get batch of these idx
        outputs = []
        with torch.no_grad():
            for i in self.idx:
                inputs = test_dataset[i]['image'].unsqueeze(0).to(device)
                depth = test_dataset[i]['depth'].unsqueeze(0).to(device)
                output = model(inputs,depth,None,None)
                output= nn.functional.interpolate(output, size=(480,640), mode='bilinear', align_corners=False)
                outputs.append(output[0].cpu().detach().permute(1,2,0).numpy())
                if 'macs' not in self.models[index]:
                    macs, params = profile(model, inputs=(inputs,depth,None,None))
                    gmacs = macs/1e9
                    gparams = params/1e6

        self.models[index]['output'] = outputs
        self.models[index]['gmacs'] = gmacs
        self.models[index]['gparams'] =gparams

    def run_tests(self):
        for i in range(len(self.models)):
            self.test(i)
        return self.models
    
    def plot_results(self):
        names = [x['name'] for x in self.models]
        r2_score = [x['r2_score'] for x in self.models]
      
        mse_score = [x['mse_score'] for x in self.models]


        mae_score = [x['mae_score'] for x in self.models]

        gmacs = [x['gmacs'] for x in self.models]
        gparams = [x['gparams'] for x in self.models]
        inference_times = [x['inference_times'] for x in self.models]

        # matplotlib subplot r2 mse mae barchart
        fig, ax = plt.subplots(3,1,figsize=(10,15))
        width=0.3
        ax[0].grid(zorder=0)
        ax[0].boxplot(np.array(r2_score).T*100,widths=width,zorder=3,labels=np.array(names))
        ax[0].set_title('R2 comparison between models')
        # ax[0].set_xticks(names)
        ax[0].set_ylabel('R2 (in %)')
        # add grid
        

        ax[1].grid(zorder=0)
        ax[1].boxplot(np.array(mse_score).T,widths=width,zorder=3,labels=np.array(names))
        # ax[1].boxplot(names,mse_score,width=width,yerr=mse_std,zorder=3)
        ax[1].set_title('Mean Squared Error comparison between models')
        # ax[1].set_xticks(names)
        ax[1].set_ylabel('Mean Squared Error (unitless)')
        # add grid  
       
        ax[2].grid(zorder=0)
        ax[2].boxplot(np.array(mae_score).T,widths=width,zorder=3,labels=np.array(names))
        ax[2].set_title('Mean Absolute Error comparison between models')
        # ax[2].set_xticks(names)
        ax[2].set_ylabel('Mean Absolute Error (unitless)') 
         



        plt.show()
        # Create the scatter plot
        mean_mae = [np.mean(x) for x in mae_score]
        mean_mse = [np.mean(x) for x in mse_score]  

        # plt.boxplot(mae_score, positions=np.round(gmacs,1),widths=5)
        standard_dev_mae = np.std(mae_score,axis=1)
        standard_dev_mse = np.std(mse_score,axis=1)
        inference_time_mean = [np.mean(x) for x in inference_times]
        print("mse_score")
        print(names)
        print(mean_mse)
        print(standard_dev_mse)
        print("inference_times")
        print(names)
        print(inference_time_mean)


        # plt.scatter(gmacs, mean_mae)
        # add grid
        plt.grid()
        # Add text next to each point
        for i, (x_val, y_val) in enumerate(zip(inference_time_mean, mean_mse)):
            # plt.text(x_val+3, y_val, names[i], ha='left', va='center', size=10)
            plt.errorbar(x_val, y_val, yerr=standard_dev_mae[i], fmt='o',label=names[i])
        plt.title('Mean Squared Error vs Inference time (in ms)')
        # Add labels
        plt.xlabel('Inference time (in ms)')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.ylabel('Mean Squared Error (unitless)')
        # Show the plot
        plt.show()
        config = {
            'transforms': {
                'normalize_input': {
                    'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]
                },
                'resize': {
                    'size': (480,640)
                },
            },
            'confidence': False,
            'ml_orchestrator': {
                'device': 'cuda:1',
                'dataset_name': 'dataset-2024-06-12_12-33-29',
            },
            'depth': {
                'depth_to_meters': 1000,
                'max_depth': 10,
                'min_depth': 0,
            },
            'cot': {
                'unknown_cot': 0,
                'wall_cot':2,
                'max_plot_cot': 2,
                'min_plot_cot': 0.5,
            }

        }
        test_dataset = self.get_test_dataset(config)
        fig, ax = plt.subplots(len(self.idx),len(self.models)+3,figsize=(20,10))
        # matplotlib subplot images
        for i in range(len(self.idx)):
            image = test_dataset[self.idx[i]]['image'].permute(1,2,0).numpy()
            image = image * config['transforms']['normalize_input']['std'] + config['transforms']['normalize_input']['mean']
            image[image > 1] = 1
            image[image < 0] = 0
            ax[i,0].imshow(image)
            ax[i,0].set_title('Original Image')
            ax[i,0].axis('off')
            ax[i,1].imshow(test_dataset[self.idx[i]]['depth'].permute(1,2,0).numpy(), cmap='nipy_spectral', vmin=0, vmax=1)
            ax[i,1].set_title('Depth')
            ax[i,1].axis('off')
            ax[i,2].imshow(test_dataset[self.idx[i]]['mask'].permute(1,2,0).numpy(), cmap='nipy_spectral', vmin=0.5, vmax=2)
            ax[i,2].set_title('Ground Truth')
            ax[i,2].axis('off')
            for j in range(len(self.models)):
                im = ax[i,3+j].imshow(self.models[j]['output'][i],vmin=0.5,vmax=2,cmap='nipy_spectral')
                ax[i,3+j].set_title(self.models[j]['name'])
                ax[i,3+j].axis('off')
                if j == len(self.models)-1:
                    divider = make_axes_locatable(ax[i,3+j])
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    fig.colorbar(im, cax=cax, orientation='vertical')
        plt.show()
