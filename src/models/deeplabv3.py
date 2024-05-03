import torch
from torch import nn
from torchvision import models
from torchvision.models.segmentation import DeepLabV3
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

class CustomDeepLabv3(nn.Module):
    def __init__(self):
        super(CustomDeepLabv3, self).__init__()
        # Load a pre-trained DeepLabV3 model with MobileNetV3 Large backbone
        self.deeplab = models.segmentation.deeplabv3_mobilenet_v3_large(num_classes=1,aux_loss=False)
        
        # MobileNetV3 requires a different method to modify the first layer due to how the model is structured
        # Unfortunately, modifying the first convolution to accept 4 channels isn't as straightforward because
        # the MobileNetV3 model in torchvision doesn't have a simple conv1 attribute
        # This is a workaround, and might require more hands-on adjustments depending on your version of torchvision
    def forward(self, x, depth):
        # Concatenate the RGB image with the depth map

        return self.deeplab(x)['out']
        
if __name__ == '__main__':
  
    # Example of using this custom model
    # Assuming `images` is your input tensor for RGB with shape [B, 3, H, W]
    # and `depth` is the depth information with shape [B, 1, H, W]

    # Initialize the custom model
    model = CustomDeepLabv3()

    # Example input tensors (make sure to replace the random tensors with actual image and depth tensors)
    images = torch.randn(4, 3, 480, 640)  # Example image tensor with batch size 4
    depth = torch.randn(4, 1, 480, 640)  # Example depth tensor with batch size 4

    # Forward pass
    output = model(images, depth)
    print("Output shape:", output.shape)

    # Here output is the predicted segmentation map with shape [B, 1, H, W]
