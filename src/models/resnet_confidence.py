import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ResNet18Confidence(nn.Module):
    def __init__(self):
        super(ResNet18Confidence, self).__init__()
        
        # Initialize ResNet18 with pretrained weights
        resnet18_model = resnet18(weights=ResNet18_Weights.DEFAULT)
        # Modify the first convolution layer to accept 4 channels instead of 3.
        resnet18_model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Splitting the encoder into two main parts for potential fine-tuning or modification flexibility
        self.encoder_part1 = nn.Sequential(*(list(resnet18_model.children())[:5])) # Early layers
        self.encoder_part2 = nn.Sequential(*(list(resnet18_model.children())[5:-2])) # Later layers minus the fully connected layer

        # Decoder could be split into separate parts for detailed configuration or modification
        self.decoder_part1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),  # From 512 to 256
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),  # To 128
            nn.ReLU()
        )
        self.decoder_part2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  # To 64
            nn.ReLU(),
        )
        self.decoder_part3 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),  # To 64
            nn.ReLU(),
            nn.ConvTranspose2d(64, 4, kernel_size=2, stride=2),  # Back to 4 channels
            nn.Sigmoid()  # Ensure output is in [0, 1]
        )

    def forward(self, input, depth,segs=None,masks=None):
        initial = torch.cat([input, depth], dim=1)
        # x = nn.functional.interpolate(initial, size=(224, 224), mode='bilinear')
        enc1 = self.encoder_part1(initial)
        x = self.encoder_part2(enc1)
        x = self.decoder_part1(x)
        print(x.shape)
        dec1 = self.decoder_part2(x)
        x = self.decoder_part3(dec1)
        # x = nn.functional.interpolate(x, size=(480, 640), mode='bilinear')

        # enc1 = nn.functional.interpolate(enc1, size=(480, 640), mode='bilinear')
        # dec1 = nn.functional.interpolate(dec1, size=(480, 640), mode='bilinear')
        return (x, initial, enc1, dec1)
    

def test_ResNet18Confidence():
    model = ResNet18Confidence()
    input = torch.randn(1, 3, 480, 640)  # Example input with RGB channels
    depth = torch.randn(1, 1, 480, 640)  # Example depth input
    (output, init, enc1, dec1) = model(input, depth)
    print("Output shape:", output.shape)
    print("Encoder part 1 shape:", enc1.shape)
    print("Decoder part 1 shape:", dec1.shape)
    # Perform assertions or checks on the output and intermediate results
    assert output.shape == (1, 4, 480, 640)  # Ensure the output shape is correct
    # assert enc1.shape ==  (1, 64, 480, 640)  # Ensure the shape of the intermediate encoder output is correct
    # assert dec1.shape ==  (1, 64, 480, 640)  # Ensure the shape of the intermediate decoder output is correct
    
    print("Test passed successfully!")
    

# Run the test
if __name__ == "__main__":
    test_ResNet18Confidence()
