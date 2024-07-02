import torch
from src.ml_orchestrator.dataset import COTDataset
from src.ml_orchestrator.loss.loss_builder import loss_builder
from src.ml_orchestrator.transforms.transforms_builder import TransformBuilder
from torch.utils.data import DataLoader
import numpy as np
from src.models.model_builder import model_builder
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from src.tools import check_file_path



# load dino
dino = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
embed_dim = dino.embed_dim
patch_size = 16

print("Embedding dimension:", embed_dim)
print("Patch size:", patch_size)




config_model_builder = {
    'model': 'ResNet18ConfidenceV2',
    'depth': True,
}

config_ml_orchestrator = {
    "learning_rate": 0.001,
    "epochs": 100,
    'dataset_name': 'dataset-2024-06-12_12-33-29',
    'valid_dataset_name': 'valid',
    "batch_size": 70,
    "optimizer": "adamw",
    "device": "cuda",
    'num_workers': 5,
    "valid_epoch":20,
    'persistent_workers': True,
    'distributed_training': True,
    'showdata': False,
}
config_loss = {
    'loss': 'ConfidenceLossV2',
    'confidence_threshold': 0.1,
}
config_cot = {
    'unknown_cot': 0,
    'wall_cot': 2,
    'max_plot_cot': 2,
    'min_plot_cot': 0,
}
config_depth = {
    'depth_to_meters': 1000,
    'max_depth': 10,
    'min_depth': 0,
}
config_transforms = {

    'normalize_input' : {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225],
    },
    'resize': {
        'size': (480,640)
    },
    "copy_paste":False,
    # 'rotate': (-180,180),
}

config = {
    'project_name':"confidence-estimation-v3",
    'exp_name':f"exp-{config_model_builder['model']}",
    'model_builder': config_model_builder,
    'ml_orchestrator': config_ml_orchestrator,
    'cot': config_cot,
    'transforms': config_transforms,
    'depth': config_depth,
    'loss': config_loss,
    'confidence':False,
}



def reject_outliers(data, m = 7.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else np.zeros(len(d))
    return data[s<m]




transform_builder = TransformBuilder(config['transforms'])

dataset_folder = check_file_path("datasets",config['ml_orchestrator']['dataset_name'])

train_dataset = COTDataset(
    confidence=config['confidence'],
    root_dir=check_file_path(dataset_folder,"train"), 
    transform_input=transform_builder.build_transforms_inputs(),
    transform_common=transform_builder.build_transform_common(), 
    config=config, 
)

train_loader = DataLoader(
    train_dataset, 
    batch_size=150, 
    shuffle=True,
    pin_memory=True,
    num_workers=config['ml_orchestrator']['num_workers']
)
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)
device  = torch.device("cuda")
dino = dino.to(device)

# fix the dino model
for param in dino.parameters():
    param.requires_grad = False

autoencoder = autoencoder.to(device)

for i in range(100):
    running_loss = 0.0
    for j,batch in enumerate(train_loader):
        inputs = batch['image'].to(device)
        depth = batch['depth'].to(device)
        masks = batch['mask'].to(device)
        segs = batch['seg'].to(device)
        confidence = batch['confidence'].to(device)

        h_patch = inputs.size(2) // patch_size
        w_patch = inputs.size(3) // patch_size

        feature_map =  dino.get_intermediate_layers(inputs, n=1)[0][:, 1:, :].view(inputs.size(0),h_patch,w_patch,embed_dim)

        seg_i = F.interpolate(segs, size=(h_patch,w_patch), mode='nearest').squeeze(1)
        mask_i = F.interpolate(masks, size=(h_patch,w_patch), mode='nearest').squeeze(1)


        unique, counts = torch.unique(segs[i], return_counts=True)

        batch_map = []
        pos_map = []

        for h in range(len(seg_i)):
            unique, counts = torch.unique(seg_i[h], return_counts=True)
            for v in range(len(unique)):
                if counts[v] / (h_patch*w_patch) < 0.02:
                    continue
                mean_diff = torch.mean(feature_map[h][seg_i[h] == unique[v]],dim=(0))
                batch_map.append(mean_diff)

                pos = torch.sum((mask_i[h] < config['cot']['wall_cot'])&(mask_i[h] > 0)&(seg_i[h] == unique[v]))
                if pos/counts[v] > 0.9:
                    pos_map.append(True)
                else:
                    pos_map.append(False)
        batch_map = torch.stack(batch_map).to(device)
        pos_map = torch.tensor(pos_map).to(device)
        # batch_map = feature_map.reshape(-1,embed_dim)
        # pos_map = ((mask_i < config['cot']['wall_cot'])&(mask_i > 0)).reshape(-1)




        outputs = autoencoder(batch_map)

        # outputs_unbatched = outputs.view(inputs.size(0),h_patch,w_patch,384)

        loss = F.mse_loss(outputs, batch_map, reduction='none')
        loss = torch.mean(loss, dim=1)

        mask_i = mask_i.detach().cpu()


        loss_cpu = loss.detach().cpu().numpy()
        masks_error=loss_cpu.flatten()
        masks_error_pos=loss_cpu[pos_map.cpu()].flatten()



        # interpolate the masks to the same size as the output mode

        if i%5==0  and j == 0:

            masks_error_pos_mean = np.array(masks_error_pos).mean()
            masks_error_pos_std= np.array(masks_error_pos).std()
            tuning_parameter = 1

            confidence=np.exp(- (masks_error-masks_error_pos_mean)**2 / (2*(masks_error_pos_std*tuning_parameter)**2) )
            confidence = np.where(loss_cpu < masks_error_pos_mean, np.ones_like(confidence), confidence)
            # filter outliers
            masks_error = reject_outliers(np.array(masks_error))
            masks_error_pos = reject_outliers(np.array(masks_error_pos))
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.hist(masks_error, bins=500, alpha=1, label='All samples')
            ax.hist(masks_error_pos, bins=500, alpha=1, label='Labeled samples')
            ax.legend()

            # draw the confidence map
            x = np.linspace(np.min(masks_error), np.max(masks_error), 100)
            y = np.exp(- (x-masks_error_pos_mean)**2 / (2*(masks_error_pos_std*tuning_parameter)**2) )
            ax.plot(x, y, 'r', linewidth=2, label='Confidence')
            plt.show()


        
            # fig, ax = plt.subplots(1,2,figsize=(10, 5))
            # output = F.interpolate(torch.from_numpy(confidence.reshape(inputs.size(0),h_patch,w_patch)).unsqueeze(1), size=(480,640), mode='nearest')[0,0,:,:].detach().cpu().numpy()
            # ax[0].imshow(output,vmin=0,vmax=1)
            # ax[1].imshow(inputs[0].detach().permute(1,2,0).cpu().numpy())
            # plt.show()

        loss_pos = loss[pos_map].mean()
        loss_pos.backward()
        optimizer.step()
        running_loss += loss_pos.item()

    print("epoch",i,"loss",running_loss)
