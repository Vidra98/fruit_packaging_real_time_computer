from PyLQR.sim import KDLRobot
from PyLQR.system import PosOrnPlannerSys, PosOrnKeypoint
from PyLQR.solver import BatchILQRCP, BatchILQR, ILQRRecursive
from PyLQR.utils import primitives, PythonCallbackMessage

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 
import cv2
import matplotlib.cm as cm
from tqdm import tqdm

import logging
import sys

from scipy.spatial.transform import Rotation 

from contact_grasp.srv import *
from cv_bridge import CvBridge, CvBridgeError

sys.path.append("/home/vdrame/catkin_ws/src/py_panda/PyPanda")
from PyPanda import Robot
import rospy
from PyPanda import Utils

from utils.camera_utils import RealCamera, RealCameraROS
from utils.transform_utils import *
from utils.iLQR_wrapper import iLQR
from utils.visualisation_utils import depth2pc
from utils.ROS_utils import generate_grasps_client, format_pointcloud_msg, run_action, get_camera_pose

import argparse
from scipy.spatial.transform import Rotation

import time

from sensor_msgs.msg import Image, PointCloud2

import rospy
from contact_grasp.srv import contactGraspnetPointcloud2, contactGraspnetPointcloud2Response
# from contact_grasp.transform_utils import pose_inv

import json
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointField, CameraInfo
from std_msgs.msg import Header

import open3d as o3d

def restart_controller(homing=False):
    rbt.stop_controller()
    rbt.error_recovery()
    rbt.switch_controller("joint_velocity_controller")
    if homing:
        rbt.gripper.homing()

if __name__ == "__main__":
    # hyperparameters
    angle_range = [-140, 40]
    grasp_horizon = 60
    view_horizon = 30
    pos_threshold = 0.015

    bridge = CvBridge()
    rospy.init_node("python_node",anonymous=False)

    # Target position
    dispose_pos = np.array([0.1, 0.66, 0.1])
    dispose_orn_wxyz = np.array([0, 1, 0.35, 0])

    # Load robot
    rbt = Robot("panda", use_gripper=True)
    rbt.gripper.homing()
    traj_gen = iLQR(rbt)

    # Camera connexion
    camera_connexion = "ROS"
    if camera_connexion == "ROS":
        camera = RealCameraROS()
        intrinsic, distortion = camera.getIntrinsic()
    elif camera_connexion == "pyWrapper":
        camera = RealCamera()
        camera.start()
        #retrieve image and depth to initialise camera, otherwise image is very dark
        for i in range(15):
            rgb, depth_image, depth_scale = camera.get_rgb_depth()
    else:
        raise Exception("Please choose a valid camera connexion method: ROS or pyWrapper")

    #load pos and orn from json file
    with open('config/views_pos.json') as json_file:
        views_pos = json.load(json_file)
        
    eef_pos, eef_quat = [], []
    restart_controller(homing=True)

    keys = views_pos.keys()
    pc_fused = []
    pc_colors_fused = []
    init_pose = np.eye(4)

    for view_idx, key in enumerate(keys):

        # Move to view
        view_jpos, view_x_pos, view_U, view_Ks, view_ds, pos_dif, orn_dif = traj_gen.direct_trajectory(rbt.q, rbt.dq, views_pos[key]["pos"], views_pos[key]["orn_wxyz"], view_horizon)
        view_U = np.array(view_U)
        success, idx, eef_pos, eef_quat = run_action(rbt, view_U, 20)
        rbt.active_controller.send_command(np.zeros(7))
        
        # Get image and depth
        img_cv, depth_cv, depth_scale = camera.get_rgb_depth()
        depth_cv = depth_cv * depth_scale

        # Get point cloud
        if view_idx == 0:
            init_pos = rbt.model.ee_pos_rel()
            init_orn_wxyz = rbt.model.ee_orn_rel()
            init_pose = get_camera_pose(rbt)
            current_pose = init_pose
            img_init = img_cv
            pc_fused, pc_colors_fused = depth2pc(depth_cv, intrinsic, img_cv)
            pc_fused, pc_colors_fused = regularize_pc(pc_fused, pc_colors_fused, 
                                                    downsampling_method="voxel", voxel_size=0.005,
                                                    outlier_filtering_method="radius", radius_param_arg=[25, 0.015])
        else:
            current_pose = get_camera_pose(rbt)
            pc_fused, pc_colors_fused = add_view2pc(pc_fused, pc_colors_fused, init_pose, current_pose, new_gbr=img_cv, 
                                                    new_depth=depth_cv, cam_intrisic=intrinsic, regularize=True, voxel_size=0.003)

        # Convert to ROS msg
        pc2_msg = format_pointcloud_msg(pc_fused, pc_colors_fused)
        bgr_msg = bridge.cv2_to_imgmsg(img_init, encoding="bgr8")

        orn, pos, opening, score, detected, detected_with_collision = generate_grasps_client(pc2_msg, bgr_msg)

        if (detected or detected_with_collision) and opening > 0.03:
            grasp_pos_world, grasps_orn_world_xyzw = poseCam2World(pos, orn, current_pose)     
            grasps_orn_world_xyzw = correct_angle(grasps_orn_world_xyzw, angle_range)
            grasp_orn_world_wxyz = convert_quat(grasps_orn_world_xyzw, to="wxyz")
            print("--------------------------------------------------------------------------")
            print("\n\ngrasps in world frame :\n pos :", grasp_pos_world, "\n grasps_orn_world :", grasps_orn_world_xyzw, "\ngrasp world rot",  Rotation.from_quat(grasps_orn_world_xyzw).as_euler("XYZ", degrees=True))
            print("------------------------------------ --------------------------------------")
            break

    grasp_jpos, grasp_x_pos, grasp_U, grasp_Ks, grasp_ds, pos_dif, orn_dif = traj_gen.grasping_trajectory(rbt.q, rbt.dq, grasp_pos_world, grasp_orn_world_wxyz, grasp_horizon)
    if pos_dif > pos_threshold:
        print("grasp trajectory please change viewpose")
   
    if opening + 0.015 >= 0.07:
        rbt.gripper.move(width=opening + 0.015)
    else:
        rbt.gripper.move(width=0.075)
        
    grasp_U = np.array(grasp_U)
    success, idx, eef_pos, eef_quat = run_action(rbt, grasp_U, 20)
    rbt.active_controller.send_command(np.zeros(7))
    rbt.gripper.move(width=opening-0.02)

    dispose_jpos, dispose_x_pos, dispose_U, dispose_Ks, dispose_ds, pos_dif, orn_dif = traj_gen.dispose_trajectory(rbt.q, rbt.dq, grasp_pos_world, grasp_orn_world_wxyz, dispose_pos, dispose_orn_wxyz, 120)
    dispose_U = np.array(dispose_U)
    success, idx, eef_pos, eef_quat = run_action(rbt, dispose_U, 20)
    rbt.active_controller.send_command(np.zeros(7))
    rbt.gripper.move(width=0.07)

    # return_horizon = 60
    # return_jpos, return_x_pos, return_U, return_Ks, return_ds, pos_dif, orn_dif = traj_gen.direct_trajectory(rbt.q, rbt.dq, init_pos, init_orn_wxyz, return_horizon)
    # return_U = np.array(return_U)
    # success, idx, eef_pos, eef_quat = run_action(rbt, return_U, 20)
    # rbt.active_controller.send_command(np.zeros(7))

    # rbt.stop_controller()