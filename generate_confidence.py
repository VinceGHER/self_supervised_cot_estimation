import wandb
from torchvision.transforms import v2
from src.ml_orchestrator.trainer import main
import multiprocessing
if __name__ == "__main__":

    wandb.login()

    config_model_builder = {
        'model': 'ResNet18Confidence',
        'depth': True,
    }

    config_ml_orchestrator = {
        "learning_rate": 0.001,
        "epochs": 100,
        'dataset_name': 'dataset-2024-06-12_12-33-29',
        'valid_dataset_name': 'valid',
        "batch_size": 24,
        "optimizer": "adamw",
        "device": "cuda",
        'num_workers': 5,
        "valid_epoch":20,
        'persistent_workers': True,
        'distributed_training': True,
        'showdata': False,
    }
    config_loss = {
        'loss': 'ConfidenceLoss',
        'confidence_threshold': 0.1,
    }
    config_cot = {
        'unknown_cot': 0,
        'wall_cot': 2,
        'max_plot_cot': 2,
        'min_plot_cot': 0,
    }
    config_depth = {
        'depth_to_meters': 1000,
        'max_depth': 10,
        'min_depth': 0,
    }
    config_transforms = {
        'normalize_input' : {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
        },
        'resize': {
            'size': (480,640)
        },
    }

    config = {
        'project_name':"confidence-estimation-v3",
        'exp_name':f"exp-{config_model_builder['model']}-{wandb.util.generate_id()}",
        'model_builder': config_model_builder,
        'ml_orchestrator': config_ml_orchestrator,
        'cot': config_cot,
        'transforms': config_transforms,
        'depth': config_depth,
        'loss': config_loss,
        'confidence':False,
    }

    main(
        config=config,
    )

