from typing import List
import open3d as o3d
import numpy as np
from tqdm import tqdm

from src.experiment import Experiment
from src.processing.extract_trajectory import Pose
from src.processing.processor import Processor
from src.robot import Robot
from src.tools import GRAVITY_CONSTANT, check_file_path, clear_folder, apply_transform, encode_float, load_calib_file, SPEED_KM_TO_MS
from bagpy import bagreader
import pandas as pd
import multiprocessing
# config_projector = {
#     # "ransac_n": 3,
#     # "ransac_num_iterations": 1000,
#     # "min_inliners": 20,
#     # "max_height": 0.5,
#     "height_offset_path": 0,
#     # "distance_threshold": 0.1,
#     "distance_path": 3,
# }

class Projector:
    def __init__(self,config):
        self.config = config
    def create_mesh_for_pose_index(self,pcd, trajectory_arg:List[Pose], pose_index:int, distance_path:str, robot:Robot,BATTERY_MSG:str,use_color_map:bool=False):
        """ Create a mesh for a given pose index and horizon"""


        points_trajectory = []
        triangles_trajectory = []
        colors_trajectory = []

        battery_msg_table = pd.read_csv(BATTERY_MSG)

        battery_msg_time = np.array(battery_msg_table['Time'])
        battery_msg_current = np.array(battery_msg_table['current'])
        battery_msg_voltage = np.array(battery_msg_table['voltage'])

        speed_acc=[]

        horizon=0
        total_distance = 0
        while total_distance<distance_path:
            horizon+=1
            if pose_index+horizon>=len(trajectory_arg):
                break
            total_distance += np.linalg.norm(trajectory_arg[pose_index+horizon-1].get_coordinates()-trajectory_arg[pose_index+horizon].get_coordinates())
        pose_index = pose_index - self.config['back_horizon']
        horizon = horizon + self.config['back_horizon']
        for i,pose_arg in enumerate(trajectory_arg[pose_index:pose_index+horizon]):
            world_to_robot_body =  pose_arg.get_homogenous_matrix() @ robot.get_camera_to_robot_body() 
            # check if we are at the end of the trajectory
            if pose_index+i+1 >= len(trajectory_arg):
                continue
            if pose_index+i-1 < 0:
                continue

            current_pose_index = np.where(
                (battery_msg_time >= trajectory_arg[pose_index+i-1].timestamp) &
                (battery_msg_time < trajectory_arg[pose_index+i+1].timestamp)) 

            if len(current_pose_index[0]) < 2:
                print("Error: no current pose found")
                continue       
            power_agv = - np.mean(battery_msg_current[current_pose_index]*battery_msg_voltage[current_pose_index])
            # print("found current pose",len(current_pose))
            deltat = trajectory_arg[pose_index+i+1].timestamp - trajectory_arg[pose_index+i-1].timestamp 
            total_energy = power_agv * deltat
            distance = np.linalg.norm(trajectory_arg[pose_index+i-1].get_coordinates()-trajectory_arg[pose_index+i+1].get_coordinates())
            speed = distance/deltat
            speed_acc.append(speed)
            if speed < self.config['min_speed']*SPEED_KM_TO_MS:
                print("Error: speed out of range", speed)
                continue
            cost_of_transport = power_agv/(robot.mass*speed*GRAVITY_CONSTANT)
            print(f"i={i}\tcot:{cost_of_transport:.4f}\tpower:{power_agv:.4f}\tdistance:{distance:.4f}\ttotal_energy:{total_energy:.4f}\tspeed:{speed:.4f}")
            # print("cost of transport:",cost_of_transport)
            if cost_of_transport < self.config['min_cost_of_transport'] or cost_of_transport > self.config['max_cost_of_transport']:
                print("Error: cost of transport out of range", cost_of_transport)
                continue

            # normalize the cost of transport with self.config
            cost_of_transport_norm = (cost_of_transport - self.config['min_cost_of_transport']) / (self.config['max_cost_of_transport'] - self.config['min_cost_of_transport'])
            # pcd.transform(np.linalg.inv(world_to_robot_body))

            # min_x_pcd = np.min(np.asarray(pcd.points)[:, 0])
            # max_x_pcd = np.max(np.asarray(pcd.points)[:, 0])
            # min_y_pcd = np.min(np.asarray(pcd.points)[:, 1])
            # max_y_pcd = np.max(np.asarray(pcd.points)[:, 1])
            # min_z_pcd = np.min(np.asarray(pcd.points)[:, 2])
            # max_z_pcd = np.max(np.asarray(pcd.points)[:, 2])

            # box = o3d.geometry.AxisAlignedBoundingBox(min_bound=(min_x_pcd, min_y_pcd, 0),
            #                                           max_bound=(max_x_pcd, max_y_pcd,max_z_pcd))

            
            # pcd_crop = pcd.crop(box)

            # DEBUG
            # pcd.paint_uniform_color([1, 0, 0])

            # coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
            # o3d.visualization.draw_geometries([coordinate_frame,pcd_crop,box,robot.get_robot_mesh()])
            # pcd.transform(world_to_robot_body)
            # if len(pcd_crop.points) < self.config['ransac_n']:
            #     print("Error: not enough points", len(pcd_crop.points))
            #     continue
            # plane_model, inliers = pcd_crop.segment_plane(distance_threshold=self.config['distance_threshold'],
            #                              ransac_n=self.config['ransac_n'],
            #                              num_iterations=self.config['ransac_num_iterations'])
            
            # a, b, c, d = plane_model
            # height_plane = - d / c
            # print("max height",d)
            # if height_plane > 0 or len(inliers) < self.config['min_inliners'] or np.linalg.norm(height_plane) > self.config['max_height']:
            #     print("Error:", height_plane, len(inliers))
            #     continue
            # get max of height of inliers

            # height_path = np.max(np.asarray(pcd_crop.points)[inliers][:, 2])  +  self.config['height_offset_path']
            height_path = -robot.height + self.config['height_offset_path']
            points_localframe = []
            points_localframe.append([robot.width/2,0 ,height_path])
            points_localframe.append([-robot.width/2,0 ,height_path])

            points_global = apply_transform(world_to_robot_body, points_localframe)
            # create a cube with 



            points_trajectory.extend(points_global)

            colors_trajectory.append(encode_float(cost_of_transport_norm,use_color_map))
            colors_trajectory.append(encode_float(cost_of_transport_norm,use_color_map))
            i = len(points_trajectory) - 1
            if i >= 3:
                triangles_trajectory.append([i-2, i-1, i])
                triangles_trajectory.append([i-3, i-1, i-2])
                # apend the triangles in the opposite direction
                triangles_trajectory.append([i, i-1, i-2])
                triangles_trajectory.append([i-2, i-1, i-3])

        if len(points_trajectory) == 0:
            print("Error: no points valid found")
            return
        # Create an Open3D TriangleMesh object
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(np.array(points_trajectory))
        mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles_trajectory))
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.array(colors_trajectory))  # Set vertex colors to shades of gray
        mesh.vertex_normals = o3d.utility.Vector3dVector([])  # Set the normal to be the same for all vertices
        # mesh.paint_uniform_color([0, 1, 0])  # Green color
        return mesh

    def load_point_cloud(self, path: str):
        """Load a point cloud from a file"""
        check_file_path(path)
        pcd_loaded = o3d.io.read_point_cloud(path)
        _, ind = pcd_loaded.remove_statistical_outlier(
            nb_neighbors=20, std_ratio=2.0)
        inlier_cloud = pcd_loaded.select_by_index(ind)
        # rotate -90 around z axis matrix
        rotate = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        inlier_cloud.rotate(rotate, center=(0, 0, 0))

        pcd = inlier_cloud
        pcd.paint_uniform_color([1, 1, 1])
        pcd.normals = o3d.utility.Vector3dVector([])
        return pcd

    def custom_draw_geometry(self, width,height,camera_intrinsic,pcd ,trajectory_arg, pose_arg, path_projected_arg,
                             distance_path, robot:Robot,BATTERY_MSG):
        """Create a custom draw geometry"""

        pose_index = trajectory_arg.index(pose_arg)
        if len(trajectory_arg) - pose_index-5 < 0:
            return
        mesh = self.create_mesh_for_pose_index(pcd,trajectory_arg, pose_index, distance_path, robot, BATTERY_MSG)
        # robot_mesh = robot.get_robot_mesh()

        # pitches = np.array([
        #     trajectory_arg[pose_index-2].get_euler_angles(),
        #     trajectory_arg[pose_index-1].get_euler_angles(),
        #     trajectory_arg[pose_index].get_euler_angles(),
        #     trajectory_arg[pose_index+1].get_euler_angles(),
        #     trajectory_arg[pose_index+2].get_euler_angles(),
        # ])
        # print("pitches",pitches)

        # pose_arg.set_euler_angles(np.mean(pitches,axis=0))


        world_to_robot_body =  pose_arg.get_homogenous_matrix() @ robot.get_camera_to_robot_body() 
        pcd.transform(np.linalg.inv(world_to_robot_body))
        colors= np.asarray(pcd.colors)
        points = np.asarray(pcd.points)

        x_min = -distance_path
        x_max = distance_path
        y_min = -distance_path
        y_max = distance_path
        z_min = robot.height

        selected_indices = np.where((points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
                            (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
                            (points[:, 2] >= z_min))
        colors[selected_indices] = [0.9,0.9,0.9]
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd.transform(world_to_robot_body)
        






        extrinsic = pose_arg.get_homogenous_matrix()
        extrinsic = np.linalg.inv(extrinsic)

        vis = o3d.visualization.Visualizer()
        vis.create_window('camera_view', width=width,
                          height=height, visible=True)
        
        render_option = vis.get_render_option()
        render_option.point_color_option = o3d.visualization.PointColorOption.Color
        vis.add_geometry(mesh)
        if pcd is not None:
            vis.add_geometry(pcd)
        ctr = vis.get_view_control()
        ctr.set_constant_z_near(0.001)
        init_param = ctr.convert_to_pinhole_camera_parameters()
        init_param.intrinsic = o3d.camera.PinholeCameraIntrinsic(width=width, height=height, 
                                                                 fx=camera_intrinsic[0, 0], fy=camera_intrinsic[1, 1], 
                                                                 cx=camera_intrinsic[0, 2], cy=camera_intrinsic[1, 2])
        init_param.extrinsic = extrinsic
        ctr.convert_from_pinhole_camera_parameters(init_param,True)

        vis.poll_events()
        vis.update_renderer()
        image = np.asarray(vis.capture_screen_float_buffer())
        # unnormlize the image
        image = image * (self.config['max_cost_of_transport'] - self.config['min_cost_of_transport']) + self.config['min_cost_of_transport']
        # image[image == self.config['max_cost_of_transport']] = 0
        image = image[:,:,0]
        np.save(f"{path_projected_arg}/{pose_arg.timestamp:.6f}.npy",image)
        # vis.capture_screen_image()
        # show the image
        vis.destroy_window()
        pcd.paint_uniform_color([1, 1, 1])
        # vis.close()
    def process_pose(self,items):
        calib_path, projected_path, cloud_path, chunk,poses,robot,BATTERY_MSG,POSITION_MSG = items
        
        pcd = self.load_point_cloud(cloud_path)
        for pose in chunk:
            calib = load_calib_file(check_file_path(calib_path,(f"{pose.timestamp:.6f}")+".yaml"))   
            self.custom_draw_geometry(
                    calib["image_width"],
                    calib["image_height"],
                    calib["camera_matrix"]["data"], 
                    pcd,
                    poses, pose, projected_path,self.config['distance_path'],
                    robot,BATTERY_MSG)
            # 5, 0.34, 0.315-0.08
    def plot_whole_trajectory(self,poses,robot,cloud_path,BATTERY_MSG):
        pcd = self.load_point_cloud(cloud_path)
        mesh = self.create_mesh_for_pose_index(pcd,poses,1,1000000,robot,BATTERY_MSG,use_color_map=True)

        pcd.paint_uniform_color([1, 0, 0])
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        o3d.visualization.draw_geometries([mesh,pcd])
class GenerateProjectPath(Processor):
    def __init__(self,config_projector):
        super().__init__()
        self.config_projector = config_projector

    def generate_chunk(self, exp:Experiment, poses: List[Pose], chunk_size: int,robot:Robot):
        items = []
        b = bagreader(check_file_path(exp.bag_path))
        BATTERY_MSG = b.message_by_topic('/mavros_node/battery')
        POSITION_MSG = b.message_by_topic('/mavros_node/global_position/local')
        for i in range(0, len(poses), chunk_size):
            chunk = poses[i:i + chunk_size]
            items.append((
                exp.calib_path,
                exp.projected_path,
                exp.cloud_path,
                chunk,
                poses,
                robot,
                BATTERY_MSG,
                POSITION_MSG))
        return items
    def process_multicore(self, exp:Experiment, poses:List[Pose],robot:Robot):
        clear_folder(exp.projected_path)        
        # split poses in chunks
        core_process = multiprocessing.cpu_count()
        chunk_size = len(poses)//core_process
        items = self.generate_chunk(exp,poses,chunk_size,robot)

        projector = Projector(self.config_projector)
        with multiprocessing.Pool(processes=core_process) as pool:
            for result in tqdm(pool.imap_unordered(projector.process_pose,items),total=len(items)):
                pass
    def plot_whole_trajectory(self,exp:Experiment,poses,robot):
        projector = Projector(self.config_projector)
        b = bagreader(check_file_path(exp.bag_path))
        BATTERY_MSG = b.message_by_topic('/mavros_node/battery')
        projector.plot_whole_trajectory(poses,robot,exp.cloud_path,BATTERY_MSG)
    def process(self, exp: Experiment, poses: List[Pose],robot:Robot):
        clear_folder(exp.projected_path)
        projector = Projector(self.config_projector)
        items = self.generate_chunk(exp,poses,1,robot)
        # projector.plot_whole_trajectory(poses,items[0][-1])
        # items = items[132:]
        # projector.process_pose(items[40])
        for item in tqdm(items):
            projector.process_pose(item)

