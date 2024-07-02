import wandb
import multiprocessing as mp
from src.ml_orchestrator.trainer import sweep


if __name__ == "__main__":
    wandb.login()


    config_model_builder = {
        'model': {"values":["AsymFormerB0_T","DFormer","CMX","fcn_resnet50"]},
        'depth': {'values':[True]},
    }

    config_ml_orchestrator = {
        "learning_rate": {"values": [1e-4]},  # Sweep through different learning rates
        "epochs":  {'values':[150]},
        'valid_dataset_name':  {'values':['valid_manually_labelled']},
        'dataset_name':  {'values':['dataset-2024-06-12_12-33-29']},
        "batch_size":  {'values':[4]},
        "optimizer":  {'values':["adamw"]},
        "device":  {'values':["cuda:0"]},
        'num_workers':  {'values':[5]},
        'persistent_workers':  {'values':[True]},
        "valid_epoch": {'values':[1]},
        'distributed_training':  {'values':[False]},
        'showdata':  {'values':[False]},
    }
    config_loss = {
        'loss': {'values':['TraversabilityLossL1','StandardLossUnlabelledNeg','StandardLoss']},
        'confidence_threshold':  {'values':[0.5]},
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
    # config_transforms2 = {
    #     'normalize_input' : {
    #         'mean': [0.485, 0.456, 0.406],
    #         'std': [0.229, 0.224, 0.225],
    #     },
    #     'resize': {
    #         'size': (480,640),
    #     },
    #     'copy_paste': False,
    # }

    config = {
        'project_name': {'values':["traversability-estimation-v6"]},
        'exp_name': {'values':[f"exp-{config_model_builder['model']}-{wandb.util.generate_id()}"]},
        'model_builder': {'parameters':config_model_builder},
        'ml_orchestrator': {"parameters":config_ml_orchestrator},
        'cot':  {'values':[config_cot]},
        'transforms': {'values':[config_transforms]},
        'depth':  {'values':[config_depth]},
        'loss': {"parameters":config_loss},
        'confidence':{'values':[True]},
    }

    sweep_config = {
        "name": "Sweep",
        "method": "grid",  # Change this to "random" for random search
        "parameters": config,  # Use the config dictionary you want to sweep
    }

    sweep_id = wandb.sweep(sweep_config, project=config['project_name']['values'][0])
    # sweep_id="vincekillerz/traversability-estimation-v3/vwihdauy"
    wandb.agent(sweep_id, function=sweep)
    
