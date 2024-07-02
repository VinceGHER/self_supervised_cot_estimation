import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np


class Plotter:
    def __init__(self,config) -> None:
        self.config = config

    def get_normalized_vmin_max(self):
        vmax = self.config['cot']['max_plot_cot']
        vmin = self.config['cot']['min_plot_cot']
        return vmin,vmax

    def batch_to_numpy(self,batch):
        images = batch['image'].cpu().permute(0, 2, 3, 1).numpy()
        masks = batch['mask'].cpu().permute(0, 2, 3, 1).numpy()
        coef = batch['confidence'].cpu().permute(0, 2, 3, 1).numpy()
        depth = batch['depth'].cpu().permute(0, 2, 3, 1).numpy()
        return images,masks,coef,depth

    def plot_batch_dataset(self,batch):
        nb_image = len(batch['image'])
        vmin,vmax = self.get_normalized_vmin_max()
        images,masks,coef,depth= self.batch_to_numpy(batch)
        fig, ax = plt.subplots(nb_image, 4, figsize=(10, 7))

        for i in range(nb_image):
            image = self.unnormalize_image(images[i])
            ax[i,0].imshow(image)
            ax[i,0].set_title('Original Image')
            ax[i,1].imshow(depth[i], cmap='nipy_spectral', vmin=0, vmax=1)
            ax[i,1].set_title('Depth')
            im1 = ax[i,2].imshow(masks[i], cmap='nipy_spectral', vmin=vmin, vmax=vmax)
            ax[i,2].set_title('Mask')
            self.add_colorbar(ax[i,2],fig,im1)
            im2 = ax[i,3].imshow(coef[i]>self.config['loss']['confidence_threshold'], cmap='nipy_spectral', vmin=0, vmax=1)
            ax[i,3].set_title('Confidence')
            self.add_colorbar(ax[i,3],fig,im2)
        fig.tight_layout()
        plt.show()

    def generate_image(self,fig):
        fig.tight_layout()
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')   
        image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return image
    def add_colorbar(self,ax,fig,im):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')

    def plot_error_confidence(self,masks_error,masks_error_pos,masks_error_pos_mean,masks_error_pos_std,tuning_parameter):
        # plot distribution of confidence for mse_recov and mse_reco_pos
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(masks_error, bins=500, alpha=1, label='All samples')
        ax.hist(masks_error_pos, bins=500, alpha=1, label='Labeled samples')
        ax.legend()

        # draw the confidence map
        x = np.linspace(np.min(masks_error), np.max(masks_error), 100)
        y = np.exp(- (x-masks_error_pos_mean)**2 / (2*(masks_error_pos_std*tuning_parameter)**2) )
        ax.plot(x, y, 'r', linewidth=2, label='Confidence')
        
        return self.generate_image(fig)
    

    def plot_confidence(self,images,confidences):
        fig, ax = plt.subplots(4, 2, figsize=(16, 3*4))
        for i in range(4):
            image = images[i].cpu().permute(1, 2, 0).numpy()
            image = self.unnormalize_image(image[:,:,:3])
            confidence = confidences[i].permute(1, 2, 0).numpy()

            ax[i,0].imshow(image)
            ax[i,0].set_title('Original Image')
            im1 = ax[i,1].imshow(confidence, cmap='nipy_spectral', vmin=0, vmax=1)
            ax[i,1].set_title('Confidence')
            self.add_colorbar(ax[i,1],fig,im1)
        return self.generate_image(fig)
    
    # def create_distribution_image(self, output_reco,cof,reco,reco_pos):
    #     # output_reco = self.unnormalize_image(output_reco)
    #     mse_reco_pos_mean = np.mean(reco_pos)
    #     mse_reco_pos_std= np.std(reco_pos)
    #     # plot line
    #     x_line = np.linspace(min(reco_pos), max(reco_pos), 1000)
    #     y_line = np.exp(- (x_line-mse_reco_pos_mean)**2 / (2*(mse_reco_pos_std)**2) )
    #     y_line[x_line<mse_reco_pos_mean]=1
    #     # Calculating the mse_trav and conditional loss_trav
    #     fig, ax = plt.subplots(1,4,figsize=(4*4, 3))
    #     try:
    #         ax[0].hist(reco, bins=100)
    #         ax[0].hist(reco_pos, bins=100)
    #     except:
    #         print('error')
    #     # ax[1].imshow(output_reco)
    #     ax[2].imshow(cof,cmap='nipy_spectral', vmin=0, vmax=1)
    #     ax[3].plot(x_line, y_line, color='red')
    #     fig.tight_layout()
    #     fig.canvas.draw()
    #     image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')   
    #     image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    #     plt.close(fig)
    #     return image
    def create_image(self, batch, outputs):
        
        images,masks,confidence,depth = self.batch_to_numpy(batch)
        predicted_masks = outputs.cpu().permute(0, 2, 3, 1).numpy()

        vmin,vmax = self.get_normalized_vmin_max()
        
        max_image = 4
        fig, ax = plt.subplots(max_image, 5, figsize=(16, 3*max_image))
    
        for i in range(min(len(images),max_image)):    
            img = self.unnormalize_image(images[i])
            
            ax[i,0].imshow(img)
            ax[i,0].set_title('Original Image')
            ax[i,1].imshow(depth[i], cmap='nipy_spectral', vmin=0, vmax=1)
            ax[i,1].set_title('Depth')

            ax[i,2].imshow(predicted_masks[i][:,:,:3], alpha=0.75,cmap='nipy_spectral', vmin=vmin, vmax=vmax)
            ax[i,2].set_title('Predicted Mask')
            ax[i,3].imshow(img)
            im2 = ax[i,3].imshow(masks[i], alpha=0.75,cmap='nipy_spectral', vmin=vmin, vmax=vmax)
            ax[i,3].set_title('True Mask')    
            self.add_colorbar(ax[i,2],fig,im2)
            ax[i,4].imshow(confidence[i])
            ax[i,4].set_title('Confidence')
        # save as image array

        return self.generate_image(fig)  

       
    def unnormalize_image(self, image):
        if 'normalize_input' in self.config['transforms']:
            image = image * self.config['transforms']['normalize_input']['std'] + self.config['transforms']['normalize_input']['mean']
            image[image > 1] = 1
            image[image < 0] = 0
        return image
    # def plot_batch_dataset(self,images,masks,coef): 
    #     nb_image = len(images)

    #     vmin,vmax = self.get_normalized_vmin_max()
    #     fig, ax = plt.subplots(nb_image, 3, figsize=(10, 5))
    #     for i in range(nb_image):
    #         image = self.unnormalize_image(images[i])
    #         ax[i,0].imshow(image)
    #         ax[i,1].imshow(masks[i], cmap='nipy_spectral', vmin=vmin, vmax=vmax)
    #         ax[i,2].imshow(coef[i]>0.1, cmap='nipy_spectral', vmin=0, vmax=1)
    #     plt.show()
    # def plot_batch_result(self,inputs,masks,outputs,cof):
    #     nb_image = inputs.shape[0]
    #     fig, ax = plt.subplots(nb_image, 4, figsize=(15, 5))
    #     vmin,vmax = self.get_normalized_vmin_max()
    #     # vmax = 3
    #     for i in range(nb_image):
    #         image = inputs[i].cpu().permute(1, 2, 0).numpy()
    #         coef = cof[i].cpu().permute(1, 2, 0).numpy()
    #         # unnormalize the image
    #         image = self.unnormalize_image(image)
    #         predicted_mask =  outputs[i].cpu().permute(1, 2, 0).numpy()
    #         true_mask = masks[i].cpu().permute(1, 2, 0).numpy()

    #         ax[i,0].imshow(image)
    #         ax[i,0].set_title('Original Image')
    #         ax[i,1].imshow(predicted_mask, cmap='nipy_spectral', vmin=vmin, vmax=vmax)
    #         ax[i,1].set_title('Predicted Mask')
    #         im0 = ax[i,2].imshow(true_mask, cmap='nipy_spectral', vmin=vmin, vmax=vmax)
    #         ax[i,2].set_title('True Mask')
    #         ax[i,3].imshow(coef, cmap='nipy_spectral', vmin=0, vmax=1)
    #         divider = make_axes_locatable( ax[i,2])
    #         cax = divider.append_axes('right', size='5%', pad=0.05)
    #         fig.colorbar(im0, cax=cax, orientation='vertical')
    #     plt.show()