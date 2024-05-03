import torch
from torchvision.transforms import v2

class TransformBuilder:
    def __init__(self, config):
        self.config = config

    def build_transforms_inputs(self):
        transforms_inputs = []
        if 'color_jitter' in self.config:
            transforms_inputs.append(v2.ColorJitter(
                brightness=self.config['color_jitter']['brightness'],
                contrast=self.config['color_jitter']['contrast'],
                saturation=self.config['color_jitter']['saturation'],
                hue=self.config['color_jitter']['hue']
            ))
        if 'gaussian_blur' in self.config:
            transforms_inputs.append(v2.RandomApply([
                v2.GaussianBlur(
                    kernel_size=self.config['gaussian_blur']['kernel_size'],
                    sigma=self.config['gaussian_blur']['sigma']
                )
            ], p=self.config['gaussian_blur']['p']))
        transforms_inputs.append(v2.ToImage())
        transforms_inputs.append(v2.ToDtype(torch.float32, scale=True))
        if 'normalize_input' in self.config:
            transforms_inputs.append(v2.Normalize(
                mean=self.config['normalize_input']['mean'],
                std=self.config['normalize_input']['std']
            ))
        return v2.Compose(transforms_inputs)

    def build_transform_common(self):
        transforms_common = []
        if 'horizontal_flip' in self.config:
            transforms_common.append(v2.RandomHorizontalFlip(p=self.config['horizontal_flip']))
        if 'resize' in self.config:
            transforms_common.append(v2.Resize(self.config['resize']['size'], interpolation=v2.InterpolationMode.NEAREST))
        return v2.Compose(transforms_common)
    def build_transform_common_validation(self):
        transforms_common = []
        if 'resize' in self.config:
            transforms_common.append(v2.Resize(self.config['resize']['size'], interpolation=v2.InterpolationMode.NEAREST))
        return v2.Compose(transforms_common)
    def build_transforms_inputs_validation(self):
        transforms_inputs = []
        transforms_inputs.append(v2.ToImage())
        transforms_inputs.append(v2.ToDtype(torch.float32, scale=True))
        if 'normalize_input' in self.config:
            transforms_inputs.append(v2.Normalize(
                mean=self.config['normalize_input']['mean'],
                std=self.config['normalize_input']['std']
            ))
        return v2.Compose(transforms_inputs)
    def get_transforms(self):
        return self.build_transforms_inputs(), self.build_transform_common()
