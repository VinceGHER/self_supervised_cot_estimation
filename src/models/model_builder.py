from importlib import import_module
import torch.nn as nn
import torch
import torchvision

from ..models.swinunet import SwinUNet
from ..models.fcn_resnet50 import FCN_resnet50
from ..models.deeplabv3 import CustomDeepLabv3
from ..models.resnet_unet import ResNetUnet
from ..models.resnet_confidence import ResNet18Confidence
from ..models.asymformer.AsymFormer import B0_T
from ..models.dformer.models.builder import EncoderDecoder as segmodel
from ..models.RGBX_Semantic_Segmentation.cmx import CMX
from ..models.resnet_confidencev2 import ResNetAutoencoderV2
from ..models.dino_confidence import DinoConfidence

def model_builder(config):
    model_name = config['model_builder']['model']
    if model_name == 'ResNetUnet':
        return ResNetUnet(depth=config['model_builder']['depth'])
    elif model_name== 'ResNet18Confidence':
        return ResNet18Confidence()
    elif model_name== 'ResNet18ConfidenceV2':
        return ResNetAutoencoderV2()
    elif model_name== 'DinoConfidence':
        return DinoConfidence(config['cot']['wall_cot'],config['ml_orchestrator']['device'])
    elif model_name == 'AsymFormerB0_T':
        model = B0_T(num_classes=1)
        # print(model)
        # s = torch.load('models/AsymFormer_NYUv2.pth')
        # state = s['state_dict']
        # del state['Decoder.linear_pred.weight'] 
        # del state['Decoder.linear_pred.bias']
        # t = model.load_state_dict(state,strict=False)
        # assert t.missing_keys == ['Decoder.linear_pred.weight', 'Decoder.linear_pred.bias']
        return model
    elif model_name== 'DFormer':
        BatchNorm2d = nn.BatchNorm2d
        config_path = "src.models.dformer.local_configs.NYUDepthv2.DFormer_Large"
        config = getattr(import_module(config_path), "C")
        model = segmodel(cfg=config, norm_layer=BatchNorm2d)
        return model
    elif model_name == 'DeepLabV3Large':
        return CustomDeepLabv3()
    elif model_name=='fcn_resnet50':
        return FCN_resnet50()
    elif model_name == 'SwinUnet':
        return SwinUNet(224, 224, 4, 32, 1,3,4)
    elif model_name == 'SwinUnet480':
        return SwinUNet(224, 224, 4, 32, 1,3,4)
    elif model_name== 'CMX':
        return CMX()
    else:
        raise ValueError(f"Model {model_name} not implemented")
