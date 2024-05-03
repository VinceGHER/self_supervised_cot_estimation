import os
class Experiment:
    def __init__(self, 
                 name, 
                 rgb="rtabmap_rgb", 
                 calib="rtabmap_calib",
                 pose="rtabmap_poses.txt",
                 cloud="rtabmap_cloud.ply",
                 projected_path="projected",
                 mask="mask",
                 depth="rtabmap_depth",
                 seg="segmentation",
                 bag="mavros.bag"
                 ):
        self.name = name
        self.rgb_path = os.path.abspath(os.path.join("exps", name, rgb))
        self.calib_path = os.path.abspath(os.path.join("exps", name, calib))
        self.pose_path = os.path.abspath(os.path.join("exps", name, pose))
        self.cloud_path = os.path.abspath(os.path.join("exps", name, cloud))
        self.projected_path = os.path.abspath(os.path.join("exps", name, projected_path))
        self.mask_path = os.path.abspath(os.path.join("exps", name, mask))
        self.depth_path = os.path.abspath(os.path.join("exps", name, depth))
        self.bag_path = os.path.abspath(os.path.join("exps", name, bag))
        self.seg_path = os.path.abspath(os.path.join("exps", name, seg))

    def __str__(self) -> str:
        if self.rgb_path is not None:
            # print number of rgb images
            rgb_files = os.listdir(self.rgb_path)
            num_rgb_files = len(rgb_files)
            return f"Experiment name: {self.name}, Number of RGB images: {num_rgb_files}"
        else:
            return f"Experiment name: {self.name}, Not processed yet"
    
    def get_experiment_folder(self):
        return os.path.abspath(f"exps/{self.name}")
    

class ExperimentManager:
    pass