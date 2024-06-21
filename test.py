from src.ml_orchestrator.dataset import COTDataset
from src.models.model_builder import model_builder
import torch
from src.ml_orchestrator.transforms.transforms_builder import TransformBuilder

from src.tools import check_file_path
import torchvision.transforms as v2

from src.visualization.plotter import Plotter

if __name__ == '__main__':
    config_model_builder = {
        'model': 'AsymFormerB0_T',
        'depth': True,
    }

    config_ml_orchestrator = {
        "learning_rate": 1e-4, # 1e-3
        "epochs":500,
        'dataset_name': 'dataset-2024-06-12_12-33-29',
        'valid_dataset_name': 'valid_manually_labelled',
        "batch_size": 12,
        "optimizer": "adamw",
        "device": "cuda",
        'num_workers': 5,
        'persistent_workers': True,
        "valid_epoch":5,
        'distributed_training': True,
        'showdata': True,
    }
    config_loss = {
        'loss': 'TraversabilityLossL1',
        'confidence_threshold': 0.5,
    }
    config_cot = {
        'unknown_cot': 0,
        'wall_cot':2,
        'max_plot_cot': 2,
        'min_plot_cot': 0.5,
    }
    config_depth = {
        'depth_to_meters': 1000,
        'max_depth': 10,
        'min_depth': 0,
    }
    config_transforms = {
        'horizontal_flip': 0.5,
        'color_jitter': {
            'brightness': 0.2,
            'contrast': 0.2,
            'saturation': 0.2,
            'hue': 0.05,
        },
        'gaussian_blur':  {
            'kernel_size': 5,
            'sigma': (0.1, 1.5),
            'p': 0.5,
        },
        'normalize_input' : {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
        },
        'resize': {
            'size': (480,640),
        },
        'resize_crop': {
            'size': (480,640),
            'scale': (0.3, 1.0),
            'ratio': (0.8, 1.2),
        },
    }

    config = {
        'project_name':"traversability-estimation-v4",
        'exp_name':f"exp-{config_model_builder['model']}",
        'model_builder': config_model_builder,
        'ml_orchestrator': config_ml_orchestrator,
        'cot': config_cot,
        'transforms': config_transforms,
        'depth': config_depth,
        'loss': config_loss,
        'confidence':True,
    }
    import torch
    import torchvision.transforms as transforms
    from torchvision.transforms import functional as F
    import random

    class CopyPasteAugmentation:
        def __init__(self, transform_prob=0.5,degres=(-70,70),translate=(0.1,0.5), scale=(0.5, 0.8),threshold_mask=0.1):
            # Probability to apply copy-paste augmentation
            self.transform_prob = transform_prob
            self.degres = degres
            self.translate = translate
            self.scale = scale
            self.threshold_mask = threshold_mask
            self.transform = v2.Compose([
               v2.RandomAffine(degrees=self.degres, translate=self.translate, scale=self.scale),
            ])


        def __call__(self, image_batch):
            batch_size = image_batch['image'].size(0)

            for i in range(batch_size):
                if random.random() < self.transform_prob:
                    j = random.randint(0, batch_size - 1)
                    self.apply_copy_paste(i,j,image_batch)
            return image_batch
        def apply_transform(self,img_i, img_j, mask):
            return img_i * (1 - mask)  + img_j * mask
        def apply_copy_paste(self, i, j, image_batch):

            mask_j = image_batch['mask'][j]
            masks,counts = torch.unique(mask_j,return_counts=True)
            pourcentage = counts / torch.sum(counts)   
            masks = masks[torch.where(pourcentage > self.threshold_mask)]



            masks = masks[masks != 0]


            if len(masks) == 0:
                return
            
            mask_index = masks[random.randint(0,len(masks)-1)]
            print("mask_index",mask_index)
            mask = torch.where(mask_j == mask_index, 1, 0)
            
            img_j = image_batch['image'][j]
            mask_j = image_batch['mask'][j]
            seg_j = image_batch['seg'][j]
            confidence_j = image_batch['confidence'][j]

            data = torch.cat((img_j, mask_j, seg_j, confidence_j,mask), dim=0)
            data.unsqueeze(0)

            data = self.transform(data)

            img_j = data[:3, :, :]
            mask_j = data[3, :, :].unsqueeze(0)
            seg_j = data[4, :, :].unsqueeze(0)
            confidence_j = data[5, :, :].unsqueeze(0)
            mask = data[6, :, :].unsqueeze(0)

            image_batch['image'][i] = self.apply_transform(image_batch['image'][i], img_j, mask)
            image_batch['mask'][i] = self.apply_transform(image_batch['mask'][i], mask_j, mask)
            image_batch['seg'][i] = self.apply_transform(image_batch['seg'][i], seg_j, mask)
            image_batch['confidence'][i] = self.apply_transform(image_batch['confidence'][i], confidence_j, mask)



    dataset_folder = check_file_path("datasets",config['ml_orchestrator']['dataset_name'])
    transform_builder = TransformBuilder(config['transforms'])

    # extend transform
    

    train_dataset = COTDataset(
        confidence=config['confidence'],
        root_dir=check_file_path(dataset_folder,"train"), 
        transform_input=transform_builder.build_transforms_inputs(),
        transform_common=transform_builder.build_transform_common(), 
        config=config, 
    )
    # batch
    copyPasteAugmentation=CopyPasteAugmentation()
    def collate_fn(batch):
        """
        Custom collate function to collate a list of samples into a single batch.
        Args:
            batch (list): List of dictionaries where each dictionary represents a single sample.
        
        Returns:
            dict: A dictionary where each key contains a batched tensor.
        """
        # Use the first sample to get the keys
        keys = batch[0].keys()

        output={}
        for key in keys:
            data = [sample[key] for sample in batch]
            if data[0] is None:
                output[key]=None
            elif isinstance(data[0], torch.Tensor):
                output[key]=torch.stack(data)
            else:
                output[key]=data
            
        # Apply copyPasteAugmentation on the collated batch
        augmented_batch = copyPasteAugmentation(output)
        
        return augmented_batch
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['ml_orchestrator']['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn
    )
    plotter = Plotter(config=config)

    for batch in train_loader:

        batch = {k: v[:4] for k, v in batch.items()}
        plotter.plot_batch_dataset(batch)

        print(batch)
        break