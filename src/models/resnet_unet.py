import torch
import torch.nn as nn
from thop import profile
from torchvision.models import resnet50, ResNet50_Weights,resnet18,ResNet18_Weights
class ResNetUnet(nn.Module):

    def __init__(self, depth=False,out_channels=1):
        super(ResNetUnet, self).__init__()

        base_model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.depth = depth
        print("Using depth:",depth)
        # Encoder
        if depth:
            self.initial = nn.Sequential(
                nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False),
                *list(base_model.children())[1:3]
            )
        else:
            self.initial = nn.Sequential(*list(base_model.children())[:3])
        self.encoder1 = base_model.layer1
        self.encoder2 = base_model.layer2
        self.encoder3 = base_model.layer3
        self.encoder4 = base_model.layer4

        # Decoder: Make deeper with additional ConvTranspose2d + Conv2d layers
        self.decoder4 = self._make_decoder_layer(2048 + 1024, 1024, additional_layers=2)
        self.decoder3 = self._make_decoder_layer(1024 + 512, 512, additional_layers=2)
        self.decoder2 = self._make_decoder_layer(512 + 256, 256, additional_layers=2)
        self.decoder1 = self._make_decoder_layer(256 + 64, 64, additional_layers=2)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.final_upsample = nn.Upsample(size=(240, 320), mode='bilinear', align_corners=True)

    def _make_decoder_layer(self, in_channels, out_channels, additional_layers=2):
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        ]
        # Add additional conv layers to make the decoders deeper
        for _ in range(additional_layers):
            layers.extend([
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ])
        return nn.Sequential(*layers)

    def forward(self, x, depth=None):
        # Encoder
        if self.depth:
            x = torch.cat([x, depth], dim=1)
        x_initial = self.initial(x)
        x1 = self.encoder1(x_initial)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)

        # Decoder with skip connections
        x_up = self.decoder4(torch.cat([x4, self._upsample(x3, x4)], dim=1))
        x_up = self.decoder3(torch.cat([x_up, self._upsample(x2, x_up)], dim=1))
        x_up = self.decoder2(torch.cat([x_up, self._upsample(x1, x_up)], dim=1))
        x_up = self.decoder1(torch.cat([x_up, self._upsample(x_initial, x_up)], dim=1))
        
        x_up = self.final_conv(x_up)
        out = self.final_upsample(x_up)

        return out

    def _upsample(self, x, target):
        return nn.functional.interpolate(x, size=(target.size(2), target.size(3)), mode='bilinear', align_corners=True)

if __name__ == "__main__":

    model = ResNetUnet(depth=True)
    print(model)
    input_tensor = torch.randn(1, 3, 240, 320)  # Input sized as desired
    depth_tensor = torch.randn(1, 1, 240, 320)  # Depth sized as desired
    output_tensor = model(input_tensor,depth_tensor)

    print(output_tensor.shape)  # Expected to be (1, 1, 240, 320)

    macs, params = profile(model, inputs=(input_tensor, depth_tensor,))
    print(macs / (1000 ** 3))
    print(params / (1000 ** 2))
