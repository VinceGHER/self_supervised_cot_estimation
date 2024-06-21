import wandb
from torchvision.transforms import v2
from src.ml_orchestrator.trainer import main

if __name__ == "__main__":
    wandb.login()

    config_model_builder = {
        'model': 'AsymFormerB0_T',
        'depth': True,
    }

    config_ml_orchestrator = {
        "learning_rate": 1e-4, # 1e-3
        "epochs":500,
        'dataset_name': 'dataset-2024-06-12_12-33-29',
        'valid_dataset_name': 'valid_manually_labelled',
        "batch_size": 12,
        "optimizer": "adamw",
        "device": "cuda",
        'num_workers': 8,
        'persistent_workers': True,
        "valid_epoch":2,
        'distributed_training': False,
        'showdata': True,
    }
    config_loss = {
        'loss': 'TraversabilityLossL1',
        'confidence_threshold': 0.5,
    }
    config_cot = {
        'unknown_cot': 0,
        'wall_cot':2,
        'max_plot_cot': 2,
        'min_plot_cot': 0.5,
    }
    config_depth = {
        'depth_to_meters': 1000,
        'max_depth': 10,
        'min_depth': 0,
    }
    config_transforms = {
        'horizontal_flip': 0.5,
        'color_jitter': {
            'brightness': 0.2,
            'contrast': 0.2,
            'saturation': 0.2,
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
            'size': (480,640),
        },
        'resize_crop': {
            'size': (480,640),
            'scale': (0.5, 1.0),
            'ratio': (0.8, 1.2),
        },
        'copy_paste': True,
    }

    config = {
        'project_name':"traversability-estimation-v4",
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

