import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb

from ...visualization.plotter import Plotter

class ConfidenceLoss(nn.Module):
    def __init__(self, config,rank):
        """
        Initialize the custom loss function.
        
        Parameters:
        - max_cost_transport: threshold value from configuration to distinguish between different mask values.
        - scale_factor: scaling factor for the loss_recovery component. Default is 100.
        """
        super(ConfidenceLoss, self).__init__()
        self.config = config
        self.plotter = Plotter(config)
        self.tuning_parameter = 1
        self.rank = rank
    def forward(self, outputs, masks, confidence, i, epoch):
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
        (outputs,initial,enc1,dec1) = outputs

        enc1_loss = F.mse_loss(enc1, dec1, reduction='none')
        enc1_loss = torch.sum(enc1_loss,dim=1).unsqueeze(1)

        mse_reco_pos = enc1_loss[(masks < self.config['cot']['wall_cot'])&(masks > 0)]
        mse_reco_pos_mean = mse_reco_pos.mean()
        mse_reco_pos_std= mse_reco_pos.std()
        # print("mse_reco_pos_mean",mse_reco_pos_mean)
        # print("mse_reco_pos_std",mse_reco_pos_std)
        # Calculating the mse_trav and conditional loss_trav
        # mse_trav = F.mse_loss(outputs, masks, reduction='none')
        confidence=torch.exp(- (enc1_loss-mse_reco_pos_mean)**2 / (2*(mse_reco_pos_std*self.tuning_parameter)**2) )
        confidence = torch.where(enc1_loss < mse_reco_pos_mean, torch.ones_like(confidence), confidence)
        # confidence_resize = F.interpolate(confidence, size=(outputs.shape[2],outputs.shape[3]), mode='nearest')
        # confidence[smoothed_mse_recov<mse_reco_pos_mean]=1


        # loss_trav = mse_trav * (masks >= self.max_cost_transport) * (1-confidence_resize) \
        #             + mse_trav * (masks < self.max_cost_transport)
        # loss_trav = loss_trav.mean()

        # Loss for the recovery part
        targets = torch.where(masks >= self.config['cot']['wall_cot'], torch.zeros_like(initial), initial)
        mse_recov = F.mse_loss(outputs, targets, reduction='none')
        mse_recov =  torch.sum(mse_recov,dim=1).unsqueeze(1)
        loss_recov = mse_recov[masks > 0].mean()
  
        # Total loss
        # loss_total = loss_trav + self.scale_factor * loss_recov
        loss_total = 10*loss_recov + mse_reco_pos_mean
        if self.rank == 0 and i==0:
            wandb.log({
                f"plot_confidence": wandb.Image(self.plotter.plot_confidence(masks[0:4].detach(),confidence[0:4].detach())),

            }, step=epoch)

        return loss_total,confidence
