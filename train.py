import wandb
from torchvision.transforms import v2
from src.ml_orchestrator.trainer import main

if __name__ == "__main__":
    wandb.login()

    config_model_builder = {
        'model': 'SwinUnet',
        'depth': True,
    }

    config_ml_orchestrator = {
        "learning_rate": 1e-4, # 1e-3
        "epochs": 300,
        'dataset_name': 'dataset-2024-04-25_14-58-19',
        "batch_size": 16,
        "optimizer": "adamw",
        "device": "cuda",
        'num_workers': 5,
        'persistent_workers': True,
        "valid_epoch":5,
        'distributed_training': False,
    }
    config_loss = {
        'loss': 'TraversabilityLossL1',
        'confidence_threshold': 0.5,
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
        'horizontal_flip': 0.5,
        'color_jitter': {
            'brightness': 0.8,
            'contrast': 0.8,
            'saturation': 0.8,
            'hue': 0.05,
        },
        'gaussian_blur':  {
            'kernel_size': 5,
            'sigma': (0.1, 1.5),
            'p': 0.5,
        },
        'normalize_input' : {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
        },
        'resize': {
            'size': (448,448),
        },
    }

    config = {
        'project_name':"base-traversability-estimation-v2",
        'exp_name':f"exp-{config_model_builder['model']}-{wandb.util.generate_id()}",
        'model_builder': config_model_builder,
        'ml_orchestrator': config_ml_orchestrator,
        'cot': config_cot,
        'transforms': config_transforms,
        'depth': config_depth,
        'loss': config_loss,
        'confidence':True,
    }



    main(
        config=config
    )

