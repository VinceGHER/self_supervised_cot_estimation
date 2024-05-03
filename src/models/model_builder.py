from importlib import import_module
import torch.nn as nn
import torchvision

from ..models.swinunet import SwinUNet
from ..models.fcn_resnet50 import FCN_resnet50
from ..models.deeplabv3 import CustomDeepLabv3
from ..models.resnet_unet import ResNetUnet
from ..models.resnet_confidence import ResNet18Confidence
from ..models.asymformer.AsymFormer import B0_T
from ..models.dformer.models.builder import EncoderDecoder as segmodel
def model_builder(config):
    if config['model'] == 'ResNetUnet':
        return ResNetUnet(depth=config['depth'])
    elif config['model'] == 'ResNet18Confidence':
        return ResNet18Confidence()
    elif config['model'] == 'AsymFormerB0_T':
        return B0_T(num_classes=1)
    elif config['model'] == 'DFormer':
        BatchNorm2d = nn.BatchNorm2d
        config_path = "src.models.dformer.local_configs.NYUDepthv2.DFormer_Large"
        config = getattr(import_module(config_path), "C")
        model = segmodel(cfg=config, norm_layer=BatchNorm2d)
        return model
    elif config['model'] == 'DeepLabV3Large':
        return CustomDeepLabv3()
    elif config['model']=='fcn_resnet50':
        return FCN_resnet50()
    elif config['model'] == 'SwinUnet':
        return SwinUNet(224, 224, 4, 32, 1,3,4)
    elif config['model'] == 'SwinUnet480':
        return SwinUNet(224, 224, 4, 32, 1,3,4)
    else:
        raise ValueError(f"Model {config['model']} not implemented")
    