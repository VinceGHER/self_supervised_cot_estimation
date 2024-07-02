import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb
import matplotlib.pyplot as plt
from src.tools import reject_outliers
from ...visualization.plotter import Plotter

class ConfidenceLossV2(nn.Module):
    def __init__(self, config,rank,wall_cot):
        """
        Initialize the custom loss function.
        
        Parameters:
        - max_cost_transport: threshold value from configuration to distinguish between different mask values.
        - scale_factor: scaling factor for the loss_recovery component. Default is 100.
        """
        super(ConfidenceLossV2, self).__init__()
        self.config = config
        self.plotter = Plotter(config)
        self.tuning_parameter = 1
        self.rank = rank
        self.wall_cot=wall_cot
    def forward(self, outputs, masks, segs, confidence, iteration, epoch,plot=False,return_confidence=False):
        """
        Compute the custom loss given the model inputs, reconstructions, outputs, and masks.

        Parameters:
        - inputs: the original inputs to the model.
        - output_reco: the reconstructed inputs by the model.
        - outputs: the predicted outputs by the model.
        - masks: the ground truth masks for computing the loss.

        Returns:
        - loss_total: the total computed loss.
        """
        # MSE between the reconstructed inputs and the original inputs
        # inputs_resize = F.interpolate(inputs, size=(24,32), mode='nearest')
        (outputs,inputs,enc1,dec1) = outputs


        seg_i = F.interpolate(segs, size=(enc1.size(2),enc1.size(3)), mode='nearest').squeeze(1)
        mask_i = F.interpolate(masks, size=(enc1.size(2),enc1.size(3)), mode='nearest').squeeze(1)

        reco_error = F.mse_loss(enc1, dec1, reduction='none')
        reco_error = torch.mean(reco_error,dim=1)


        batch_map = []
        pos_map = []
        for h in range(len(seg_i)):
            unique, counts = torch.unique(seg_i[h], return_counts=True)
            group_pos = []
            for v in range(len(unique)):
                if counts[v] / (enc1.size(2)*enc1.size(3)) < 0.01:
                    continue
                
                pos = torch.sum((mask_i[h] < self.wall_cot)&(mask_i[h] > 0)&(seg_i[h] == unique[v]))
                mean_error_mask = torch.mean(reco_error[h][seg_i[h] == unique[v]])
                batch_map.append(mean_error_mask)
                if pos/counts[v] > 0.01:
                    group_pos.append(torch.tensor([True,unique[v]]))
                else:
                    group_pos.append(torch.tensor([False,unique[v]]))
            pos_map.append(torch.stack(group_pos).to(inputs.device))

        batch_map = torch.stack(batch_map)
    
        flat_reco_error_pos = batch_map[torch.cat(pos_map)[:,0]==1]
        if self.rank == 0 and iteration==0 and plot or return_confidence:

            masks_error_pos_mean =flat_reco_error_pos.mean()
            masks_error_pos_std= flat_reco_error_pos.std()
            tuning_parameter = 1

            confidence=torch.exp(- (batch_map-masks_error_pos_mean)**2 / (2*(masks_error_pos_std*tuning_parameter)**2) )
            confidence = torch.where(batch_map < masks_error_pos_mean, torch.ones_like(confidence), confidence)
        
            confidences_batch = torch.ones(masks.shape)  
            index_confidence = 0          
            for h in range(len(pos_map)):
                for e in pos_map[h][:,1]:
                    confidences_batch[h][segs[h] == e] = confidence[index_confidence]
                    index_confidence+=1
                if plot and h >=4:
                    break
            # filter outliers
            reco_error_mask = batch_map.detach().cpu().numpy().flatten()
            reco_pos = flat_reco_error_pos.detach().cpu().numpy().flatten()
            masks_error = reject_outliers(np.array(reco_error_mask))
            masks_error_pos = reject_outliers(np.array(reco_pos))
            
            
            if return_confidence:
                return confidences_batch

            


            wandb.log({
                f"plot_confidence": wandb.Image(self.plotter.plot_confidence(inputs[0:4].detach().cpu(),confidences_batch[0:4].detach())),

                f"distribution": wandb.Image(self.plotter.plot_error_confidence(masks_error,masks_error_pos,masks_error_pos_mean.cpu().detach().numpy(),masks_error_pos_std.cpu().detach().numpy(),tuning_parameter)),
            }, step=epoch)

        targets = torch.where(masks >= self.wall_cot, torch.zeros_like(inputs), inputs)
        mse_recov = F.mse_loss(outputs, targets, reduction='none')
        mse_recov =  torch.sum(mse_recov,dim=1).unsqueeze(1)



        
        loss_recov = mse_recov[masks > 0].mean()
  
        # Total loss
        # loss_total = loss_trav + self.scale_factor * loss_recov
        loss_total = loss_recov + flat_reco_error_pos.mean()

        return loss_total