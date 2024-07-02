import torch
from torchvision.transforms import v2
import random

class CopyPasteAugmentation:
    def __init__(self, wallcot,transform_prob=0.5,repeat=2,degres=(-70,70),translate=(0.1,0.5), scale=(0.5, 0.8),threshold_mask=0.1,unknowncot=0):
        # Probability to apply copy-paste augmentation
        self.transform_prob = transform_prob
        self.degres = degres
        self.translate = translate
        self.scale = scale
        self.threshold_mask = threshold_mask
        self.wall_cot = wallcot
        self.transform = v2.Compose([
            v2.RandomAffine(degrees=self.degres, translate=self.translate, scale=self.scale),
        ])
        self.repeat = repeat


    def __call__(self, image_batch):
        batch_size = image_batch['image'].size(0)

        for i in range(batch_size):
            for _ in range(self.repeat):
                if random.random() < self.transform_prob:
                    j = random.randint(0, batch_size - 1)
                    self.apply_copy_paste(i,j,image_batch)
        return image_batch
    def apply_transform(self,img_i, img_j, mask):
        return img_i * (1 - mask)  + img_j * mask
    def apply_copy_paste(self, i, j, image_batch):

        mask_j = image_batch['mask'][j]
        # set the mask to 0 for the wall cot
        mask_j = torch.where(mask_j >= self.wall_cot, 0, mask_j)
        masks,counts = torch.unique(mask_j,return_counts=True)
        pourcentage = counts / torch.sum(counts)   
        masks = masks[torch.where(pourcentage > self.threshold_mask)]



        masks = masks[masks != 0]


        if len(masks) == 0:
            return
        
        mask_index = masks[random.randint(0,len(masks)-1)]
    
        mask = torch.where(mask_j == mask_index, 1, 0)
        
        img_j = image_batch['image'][j]
        mask_j = image_batch['mask'][j]
        seg_j = image_batch['seg'][j]
        depth_j = image_batch['depth'][j]
        confidence_j = image_batch['confidence'][j]

        data = torch.cat((img_j, mask_j, seg_j, confidence_j,mask,depth_j), dim=0)
        data.unsqueeze(0)

        data = self.transform(data)

        img_j = data[:3, :, :]
        mask_j = data[3, :, :].unsqueeze(0)
        seg_j = data[4, :, :].unsqueeze(0)
        confidence_j = data[5, :, :].unsqueeze(0)
        mask = data[6, :, :].unsqueeze(0)
        depth_j = data[7, :, :].unsqueeze(0)

        image_batch['image'][i] = self.apply_transform(image_batch['image'][i], img_j, mask)
        # image_batch['depth'][i] = self.apply_transform(image_batch['depth'][i], depth_j, mask)
        image_batch['mask'][i] = self.apply_transform(image_batch['mask'][i], mask_j, mask)
        image_batch['seg'][i] = self.apply_transform(image_batch['seg'][i], seg_j, mask)
        image_batch['confidence'][i] = self.apply_transform(image_batch['confidence'][i], confidence_j, mask)
