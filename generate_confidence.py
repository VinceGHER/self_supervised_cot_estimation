import wandb
from torchvision.transforms import v2
from src.ml_orchestrator.trainer import train
wandb.login()

config_model_builder = {
    'model': 'ResNet18Confidence',
    'depth': True,
}

config_ml_orchestrator = {
    "learning_rate": 0.001,
    "epochs": 300,
    'dataset_name': 'dataset-2024-04-25_14-58-19',
    "batch_size": 24,
    "optimizer": "adam",
    "device": "cuda",
    'num_workers': 0,
}
config_loss = {
    'loss': 'ConfidenceLoss',
    'confidence_threshold': 0.1,
}
config_cot = {
    'unknown_cot': 3,
    'wall_cot': 8,
    'max_plot_cot': 10,
    'min_plot_cot': 0,
}
config_depth = {
    'depth_to_meters': 1000,
    'max_depth': 20,
    'min_depth': 0,
}
config_transforms = {
    'normalize_input' : {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225],
    },
    'resize': {
        'size': (240,320),
    },
}

config = {
    'model_builder': config_model_builder,
    'ml_orchestrator': config_ml_orchestrator,
    'cot': config_cot,
    'transforms': config_transforms,
    'depth': config_depth,
    'loss': config_loss,
    'confidence':False,
}



train(
    config=config,
    project_name="base-confidence-estimation-v2",
    exp_name=f"exp-{config['model_builder']['model']}-{wandb.util.generate_id()}",
    show_data=True
)

