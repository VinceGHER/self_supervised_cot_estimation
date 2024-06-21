from src.tools import transform_matrix_rotation,transform_matrix_translation
import numpy as np
import open3d as o3d
class Robot:
    def __init__(self,width,height,length, robot_body_to_camera,mass) -> None:
        self.width = width
        self.height = height
        self.length = length
        self.robot_body_to_camera = robot_body_to_camera
        self.mass = mass

    def get_robot_body_to_camera(self):
        return self.robot_body_to_camera
    
    def get_camera_to_robot_body(self):
        return np.linalg.inv(self.robot_body_to_camera)
    
    def get_robot_mesh(self):
        robot_body = o3d.geometry.TriangleMesh.create_box(self.width,self.length,self.height)
        robot_body.translate(np.array([-self.width/2,-self.length/2,-self.height]))
        return robot_body

class M4V1Robot(Robot):
    def __init__(self) -> None:
        width = 0.34
        height = 0.33
        length = 0.70
        mass = 3
        robot_body_to_camera = transform_matrix_translation(0.02,0,0) @ transform_matrix_translation(0,0.2925,0) @ transform_matrix_rotation(-90,0,0) 

        super().__init__(width,height,length,robot_body_to_camera,mass)