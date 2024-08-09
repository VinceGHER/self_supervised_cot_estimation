from scipy.spatial.transform import Rotation as R
import numpy as np

from src.experiment import Experiment
from src.processing.processor import Processor
from src.tools import check_file_path, run_command
class Pose:
    """
    A class to represent a pose  
    """
    def __init__(self, timestamp, x, y, z, qx, qy, qz, qw):
        self.timestamp = timestamp
        self.x = x
        self.y = y
        self.z = z
        self.qx = qx
        self.qy = qy
        self.qz = qz
        self.qw = qw
    def get_euler_angles(self):
        r = R.from_quat([self.qx, self.qy, self.qz, self.qw])
        return r.as_euler('xyz', degrees=True)

    def set_euler_angles(self, euler_angles_updated):
        r = R.from_euler('xyz', euler_angles_updated, degrees=True)
        self.qx, self.qy, self.qz, self.qw = r.as_quat()
    def get_coordinates(self):
        return np.array([self.x, self.y, self.z])
    def __str__(self):
        return f"Pose({self.timestamp}, {self.x}, {self.y}, {self.z}, {self.qx}, {self.qy}, {self.qz}, {self.qw})"

    def __repr__(self):
        return str(self)

    def get_homogenous_matrix(self):
        """Convert a pose to a homogenous matrix"""
        r = R.from_quat([self.qx, self.qy, self.qz, self.qw])
        # homogeneous transformation matrix from local to global frame
        H = np.eye(4)
        H[:3, :3] = r.as_matrix()
        H[:3, 3] = [self.x, self.y, self.z]
        return H
    
class ExtractTrajectory(Processor):
    def __init__(self):
        super().__init__()
    
    def process(self,exp: Experiment):
        return self.load_trajectory(exp.pose_path)


    def load_trajectory(self, path: str,remove_header=True) :
        """Load the poses from a file in format timestamp x y z qx qy qz qw"""
        check_file_path(path)
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        if remove_header:
            lines = lines[1:]  # remove header
        poses = []
        for line in lines:
            timestamp, x, y, z, qx, qy, qz, qw = line.split()
            pose_obj = Pose(float(timestamp), float(x), float(y), float(z),
                        float(qx), float(qy), float(qz), float(qw))
            poses.append(pose_obj)
        return poses