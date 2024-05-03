import torch
import torch.nn as nn
import torch.nn.functional as F

class StandardLoss(nn.Module):
    def __init__(self, config):
        """
        Initialize the custom loss function.
        
        Parameters:
        - max_cost_transport: threshold value from configuration to distinguish between different mask values.
        - scale_factor: scaling factor for the loss_recovery component. Default is 100.
        """
        super(StandardLoss, self).__init__()
        self.config = config

    def forward(self, outputs, masks, confidence,epoch):
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

        # targets = torch.where(masks == 0, torch.ones_like(masks)*self.config['cot']['wall_cot'], masks)
        mse_loss = F.mse_loss(outputs, masks, reduction='none')

        # Filtering the MSE loss based on the mask condition
        # confidence and not masks
        no_loss_mask = ~( masks == 0 )
        selected_loss = mse_loss[no_loss_mask]

        return selected_loss.mean()