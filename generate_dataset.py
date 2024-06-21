from src.processing.generate_dataset import GenerateDataset
from src.experiment import Experiment
from src.processing.extract_rtabmap_db import ExtractRtabmapDB
from src.processing.extract_trajectory import ExtractTrajectory
from src.processing.generate_mask import GenerateMask
from src.processing.generate_projected_path import GenerateProjectPath
from src.processing.undistort_rgb import Undistort_rgb
from src.robot import M4V1Robot
from src.visualization.comparer import Comparer

import multiprocessing
if __name__ == "__main__":
    multiprocessing.set_start_method('forkserver')
    
    exps = [
        # # Experiment("exp1-1"), # cut end
        # Experiment("exp1-4"),
        # Experiment("exp1-5"),
        # Experiment("exp1-7"),
        # Experiment("exp1-8"),
        # Experiment("exp1-9"),
        # Experiment("exp1-10"),
        # # Experiment("exp2-1"),
        # # Experiment("exp2-3"),
        # # Experiment("exp2-6"),
        # # Experiment("exp2-7"),
        # # Experiment("exp2-8"),
        # Experiment("exp2-9"),
        # # Experiment("exp2-11"),
        # # Experiment("exp2-12"),
        # # Experiment("exp2-13"),
        # # Experiment("expcollect2"),

        # Experiment("exp1-1"),
        # Experiment("exp1-4"),
        # Experiment("exp1-5"),
        # Experiment("exp1-7"),
        # Experiment("exp2-1"),
        # Experiment("exp2-3"),
        # Experiment("exp2-6"),
        # Experiment("exp2-7"),
        # Experiment("exp2-8"),
        # Experiment("exp2-9"),
        # Experiment("exp2-11"),
        # Experiment("exp2-13"),
        Experiment("expcollect2")
    ]
    
    robot = M4V1Robot()
    processor_unnormalize_rgb = Undistort_rgb()
    processor_extract_rtabmap_db = ExtractRtabmapDB('rtabmap-export')
    processor_extract_trajectory= ExtractTrajectory()
    config_projector = {
        "height_offset_path": 0.08,
        "distance_path": 5,
        # "offset_current_time": 2,
        'min_cost_of_transport':0.5,
        'max_cost_of_transport':2,
        'min_speed':0.4,
        'back_horizon': 1,
        'debug':False,
        'debug_index':6
    }
    processor_generate_projected_path = GenerateProjectPath(config_projector)
    params_sam = {
            "pred_iou_thresh": 0.95,
            "min_mask_region_area":1000,
            "stability_score_thresh":0.95
    }

    processor_generate_mask = GenerateMask(
        params_sam,
        score_thresh=0.2,
        kernel_size=8,
        max_cost_transport=config_projector['max_cost_of_transport'])
    comparer = Comparer(config_projector['min_cost_of_transport'],config_projector['max_cost_of_transport'],100)
    processor_generate_dataset = GenerateDataset()

    for exp in exps:
    # comparer.plot_robot_camera(robot)
        # processor_extract_rtabmap_db.process(exp)
        # processor_unnormalize_rgb.process(exp)
        poses = processor_extract_trajectory.process(exp)
        processor_generate_projected_path.plot_whole_trajectory(exp,poses,robot)
        # processor_generate_projected_path.process_multicore(exp, poses,robot)
        # processor_generate_mask.process_multicore(exp)
        # # print(exp)
        # comparer.compare(exp,save_video=True)
    
    # processor_generate_dataset.process([
    #     (Experiment("exp1-1"), [(0,159)]),
    #     (Experiment("exp1-4"), [(0,64),(170,-1)]),
    #     (Experiment("exp1-5"), [(54,-1)])   ,
    #     (Experiment("exp1-7"), [(88,122)])    ,
    #     (Experiment("exp2-1"),[(4,15),(22,36)]),
    #     (Experiment("exp2-3"),[(0,33)]),
    #     (Experiment("exp2-6"),[(0,3)]),
    #     (Experiment("exp2-7"),[(0,-1)]),
    #     (Experiment("exp2-8"),[(0,-1)]),
    #     (Experiment("exp2-9"),[(27,44),(57,122)]),
    #     (Experiment("exp2-11"),[(0,19)]),
    #     (Experiment("exp2-13"),[(0,41),(49,101),(105,112),(130,198),(218,234),(237,-1)]),
    #     (Experiment("expcollect2"),[(38,56),(73,120)]),

        # (Experiment("exp1"),0,-1),
        # (Experiment("exp2"),0,-1),
        # (Experiment("exp3"),0,-1),
        # (Experiment("exp4"),0,-1),
        # (Experiment("exp5"),0,-1),
        # (Experiment("exp7"),0,-1),
        # (Experiment("exp8"),0,-1),
        # (Experiment("exp9"),0,-1),
        # (Experiment("exp10"), 37,100),
        # (Experiment("expcollect1"),0,-1),
        # (Experiment("expcollect2"),0,113),
        # ]  
    # )
    # comparer.compute_bias_and_variance(exp)
    # print(poses)