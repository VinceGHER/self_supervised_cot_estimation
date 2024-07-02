import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb
import matplotlib.pyplot as plt

from src.tools import reject_outliers

from ...visualization.plotter import Plotter

class ConfidenceLossV3(nn.Module):
    def __init__(self, config,rank):
        """
        Initialize the custom loss function.
        
        Parameters:
        - max_cost_transport: threshold value from configuration to distinguish between different mask values.
        - scale_factor: scaling factor for the loss_recovery component. Default is 100.
        """
        super(ConfidenceLossV3, self).__init__()
        self.config = config
        self.plotter = Plotter(config)
        self.tuning_parameter = 1
        self.rank = rank
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
        (dummy_outputs,outputs,batch_map,pos_map,seg_i,mask_i,inputs)= outputs


 
        loss = F.mse_loss(outputs, batch_map, reduction='none')
        loss = torch.mean(loss, dim=1)


        mask_i = mask_i.detach().cpu()


        loss_cpu = loss.detach().cpu().numpy()
        masks_error=loss_cpu.flatten()
        flat_pos_map = torch.cat(pos_map)[:,0].cpu()==1
        masks_error_pos=loss_cpu[flat_pos_map].flatten()


        if self.rank == 0 and iteration==0 and plot or return_confidence:

            masks_error_pos_mean =torch.tensor(masks_error_pos).mean()
            masks_error_pos_std= torch.tensor(masks_error_pos).std()
            tuning_parameter = 1

            confidence=torch.exp(- (loss.flatten()-masks_error_pos_mean)**2 / (2*(masks_error_pos_std*tuning_parameter)**2) )
            confidence = torch.where(loss.flatten() < masks_error_pos_mean, torch.ones_like(confidence), confidence)

            confidences_batch = torch.ones_like(masks)  
            index_confidence = 0          
            for h in range(len(pos_map)):
                for e in pos_map[h][:,1]:
                    confidences_batch[h][segs[h] == e] = confidence[index_confidence]
                    index_confidence+=1
                if plot and h >=4:
                    break
            # filter outliers
            masks_error = reject_outliers(np.array(masks_error))
            masks_error_pos = reject_outliers(np.array(masks_error_pos))
            
            
            if return_confidence:
                return confidences_batch

            


            wandb.log({
                f"plot_confidence": wandb.Image(self.plotter.plot_confidence(inputs[0:4].detach(),confidences_batch[0:4].detach())),

                f"distribution": wandb.Image(self.plotter.plot_error_confidence(masks_error,masks_error_pos,masks_error_pos_mean.numpy(),masks_error_pos_std.numpy(),tuning_parameter)),
            }, step=epoch)

        return loss[flat_pos_map].mean()
        # reco_error = F.mse_loss(enc1, dec1, reduction='none')
        # reco_error = torch.sum(reco_error,dim=1).unsqueeze(1)

        # masks_error=[]
        # masks_error_pos=[]
        # loss=0
        # for i in range(len(outputs)):
        #     confidence_segs = torch.zeros_like(confidence[i])
        #     unique, counts = torch.unique(segs[i], return_counts=True)
        #     for j in range(len(unique)):
        #         if counts[j] / (480*640) < 0.05:
        #             continue
        #         mean = torch.mean(reco_error[i][segs[i] == unique[j]])

        #         masks_error.append(mean.detach().cpu().numpy())
     
        #         pos = torch.sum((masks[i] < self.config['cot']['wall_cot'])&(masks[i] > 0)&(segs[i] == unique[j]))
        #         neg = torch.sum((masks[i] >= self.config['cot']['wall_cot'])&(segs[i] == unique[j]))
        #         if neg > 0:
        #             v = torch.sum(outputs[i],dim=0).unsqueeze(0)
        #             loss += torch.pow(v[segs[i] == unique[j]],2).mean()
        #         if pos > 0:
        #             loss += mean
        #             masks_error_pos.append(mean.detach().cpu().numpy())



        # masks_error_pos_mean = np.array(masks_error_pos).mean()
        # masks_error_pos_std= np.array(masks_error_pos).std()




        # # print("mse_reco_pos_mean",mse_reco_pos_mean)
        # # print("mse_reco_pos_std",mse_reco_pos_std)
        # # Calculating the mse_trav and conditional loss_trav
        # # mse_trav = F.mse_loss(outputs, masks, reduction='none')
        # confidence=np.exp(- (masks_error-masks_error_pos_mean)**2 / (2*(masks_error_pos_std*self.tuning_parameter)**2) )
        # confidence = np.where(masks_error < masks_error_pos_mean, np.ones_like(confidence), confidence)
        # # confidence_resize = F.interpolate(confidence, size=(outputs.shape[2],outputs.shape[3]), mode='nearest')
        # # confidence[smoothed_mse_recov<mse_reco_pos_mean]=1


        # # loss_trav = mse_trav * (masks >= self.max_cost_transport) * (1-confidence_resize) \
        # #             + mse_trav * (masks < self.max_cost_transport)
        # # loss_trav = loss_trav.mean()

        # # Loss for the recovery part
        # targets = torch.where(masks >= self.config['cot']['wall_cot'], torch.zeros_like(initial), initial)
        # mse_recov = F.mse_loss(outputs, targets, reduction='none')
        # mse_recov =  torch.sum(mse_recov,dim=1).unsqueeze(1)



        
        # loss_recov = mse_recov[masks > 0].mean()
  
        # # # Total loss
        # # # loss_total = loss_trav + self.scale_factor * loss_recov
        # loss_total = 100*loss_recov + loss

        # if self.rank == 0 and iteration==0 and plot:
            
        #     wandb.log({
        #         # f"plot_confidence": wandb.Image(self.plotter.plot_confidence(masks[0:4].detach(),confidence[0:4].detach())),

        #         f"distribution": wandb.Image(self.plotter.plot_error_confidence(masks_error,masks_error_pos)),
        #     }, step=epoch)

        # return loss_total
