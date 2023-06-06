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

# sys.path.append("/home/vdrame/catkin_ws/src/py_panda/PyPanda")
from PyPanda import Robot
import rospy
from PyPanda import Utils

from utils.camera_utils import RealCamera, RealCameraROS
from utils.transform_utils import *
from utils.iLQR_wrapper import iLQR
from utils.visualisation_utils import depth2pc

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


def run_action(rbt, actions, control_freq, eef_pos=None, eef_quat=None, segmentation_type=None, show_agentview=False, object_range=[5,8]):
    success = False
    rate = rospy.Rate(int(control_freq))

    for idx, action in tqdm(enumerate(actions)):
        rbt.active_controller.send_command(action)
        rate.sleep()
        #env.sim.step()
        if eef_pos is not None:
            eef_pos.append(rbt.model.ee_pos_rel())
        if eef_quat is not None:
            eef_quat.append(rbt.model.ee_orn_rel())
    success = True
    return success, idx, eef_pos, eef_quat

def generate_grasps_client(pc2, bgr8):
    rospy.wait_for_service('generate_grasps_pc')
    try:
        print("calling service")
        generate_grasps = rospy.ServiceProxy('generate_grasps_pc', contactGraspnetPointcloud2)
        resp1 = generate_grasps(pc2, bgr8)
        return resp1.quat, resp1.pos, resp1.opening.data, resp1.score.data, resp1.detected.data, resp1.detected_with_collision.data
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

def format_pointcloud_msg(points, colors):
    points = np.hstack((points, colors)).astype(dtype=object)
    points[:,3:] = points[:,3:].astype(np.uint8)
    fields = [PointField('x', 0, PointField.FLOAT32, 1),
          PointField('y', 4, PointField.FLOAT32, 1),
          PointField('z', 8, PointField.FLOAT32, 1),
          # PointField('rgb', 12, PointField.UINT32, 1),
          PointField('r', 12, PointField.UINT8, 1),
          PointField('g', 13, PointField.UINT8, 1),
          PointField('b', 14, PointField.UINT8, 1),
          ]
    
    header = Header()
    header.frame_id = 'camera_link'
    pc2 = point_cloud2.create_cloud(header, fields, points)
    return pc2


def get_camera_pose(rbt):
    """ Get camera pose in robot base frame
    """
    ee_pose = np.eye(4)
    ee_pose[:3,:3] = quat2mat(convert_quat(rbt.model.ee_orn_rel(), to="xyzw")) #xyzw
    ee_pose[:3,3] = rbt.model.ee_pos_rel() 

    ee2hand = np.eye(4)
    ee2hand[2,3] = -0.1034

    with open('config/camera_calibration.json') as json_file:
        camera_calibration = json.load(json_file)

    camera_type = "L515" #D415 or L515

    hand2camera_pos = np.array(camera_calibration[camera_type]["pos"])
    hand2camera_quat = camera_calibration[camera_type]["quat_xyzw"] #xyzw

    # TODO todelete
    # #D415
    # # hand2camera_pos = np.array([0.0488546636437146,-0.03384417860749521,0.0512776975002817]) 
    # # hand2camera_quat = [0.012961267509189803,-0.0012768531849757236,0.7052247395136084,0.708864191484139] #xyzw 

    # #L515
    # hand2camera_pos = np.array([0.08329189218278059, 0.0014213145240625528, 0.0504764049956106]) 
    # hand2camera_quat = [0.01521805627198811, 0.00623363612254646, 0.712108725756912, 0.7018765669580811] #xyzw 

    hand2camera_mat = Rotation.from_quat(hand2camera_quat).as_matrix()

    hand2camera = np.eye(4)
    hand2camera[:3,:3] = hand2camera_mat
    hand2camera[:3,3] = hand2camera_pos

    current_pose = ee_pose @ ee2hand @ hand2camera

    return current_pose


bridge = CvBridge()
rospy.init_node("python_node",anonymous=True)

dispose_pos = np.array([0.1, 0.66, 0.1])
dispose_orn_wxyz = np.array([0, 1, 0.35, 0])

# Load robot
rbt = Robot("panda", use_gripper=True)
rbt.gripper.homing()

camera_connexion = "ROS"
if camera_connexion == "ROS":
    camera = RealCameraROS()
    intrinsic, distortion = camera.getIntrinsic()
elif camera_connexion == "pyWrapper":
    camera = RealCamera()
    camera.start()
    #retrieve image and depth
    for i in range(15):
        rgb, depth_image, depth_scale = camera.get_rgb_depth()

    rgb, depth_image, depth_scale = camera.get_rgb_depth()
    intrinsic, distortion = camera.getIntrinsic()
else:
    raise Exception("Please choose a valid camera connexion method: ROS or pyWrapper")



print("rbt end effector pos:", rbt.model.ee_pos_rel())
print("rbt end effector quat (wxyz):", rbt.model.ee_orn_rel())
print("eef rot",  Rotation.from_quat(convert_quat(rbt.model.ee_orn_rel(), to="xyzw")).as_euler("xyz", degrees=True))
print("--------------------------------------------------------------------------")

reference_pose = get_camera_pose(rbt)

img_init, depth_image, depth_scale = camera.get_rgb_depth()
depth_init = depth_image * depth_scale

pc = None
pc_colors = None
pc_init, pc_colors_init = depth2pc(depth_init, intrinsic, img_init)
pc2_msg = format_pointcloud_msg(pc_init.copy(), pc_colors_init.copy())
bgr_msg = bridge.cv2_to_imgmsg(img_init, encoding="bgr8")

init_pos = rbt.model.ee_pos_rel()
init_orn_wxyz = rbt.model.ee_orn_rel()

# view_init_pos = [0.17908377, 0.23450877, 0.44783125]
# view_init_orn_wxyz = [0.0030477, 0.92571715, 0.37800879, 0.01215875]
# view_front_pos = [0.36072976, 0.41291873, 0.30648647]
# view_front_orn_wxyz = [ 0.1655683,   0.81494402,  0.28065989, -0.47925298]
# view_right_pos = [0.10118165, 0.47394513, 0.30265239]
# view_right_orn_wxyz = [ 0.3190109,   0.94507174, -0.07051679,  0.00994117]
# view_back_pos = [0.00465632, 0.32838561, 0.26232907]
# view_back_orn_wxyz = [ 0.06700862,  0.96685827, -0.02494526,  0.24509727]

# view_pose = [get_camera_pose(rbt)]
# views_pos = [init_pos, view_front_pos, view_right_pos, view_back_pos]
# views_orn_wxyz = [init_orn_wxyz, view_front_orn_wxyz, view_right_orn_wxyz, view_back_orn_wxyz]

#load pos and orn from json file
with open('config/views_pos.json') as json_file:
    views_pos = json.load(json_file)

keys = views_pos.keys()
for key in keys:
    print(views_pos[key]["pos"])
    print(views_pos[key]["orn_wxyz"])    



eef_pos, eef_quat = [], []
traj_gen = iLQR(rbt)
rbt.stop_controller()
rbt.error_recovery()
rbt.switch_controller("joint_velocity_controller")
rbt.active_controller
#Ros CV bridge to convert data from opencv to ROSImage

# Call to the server
orn, pos, opening, score, detected, detected_with_collision = generate_grasps_client(pc2_msg, bgr_msg)
if (detected or detected_with_collision) and opening>0.03:
    grasp_pos_world, grasps_orn_world_xyzw = poseCam2World(pos, orn, reference_pose)
    grasp_orn_world_wxyz =  convert_quat(grasps_orn_world_xyzw, to="wxyz")

    print("--------------------------------------------------------------------------")
    print("\n\ngrasps in world frame :\n pos :", grasp_pos_world, "\n grasps_orn_world :", grasps_orn_world_xyzw, "\ngrasp world rot",  Rotation.from_quat(grasps_orn_world_xyzw).as_euler("xyz", degrees=True))
    print("--------------------------------------------------------------------------")
else:
    pc_fused = pc_init
    pc_colors_fused = pc_colors_init
    pos_dif = 1000
    for key in keys:
    # while (not detected or not detected_with_collision):
        horizon = 30
        while pos_dif > 0.01:
            view_jpos, view_x_pos, view_U, view_Ks, view_ds, pos_dif, orn_dif = traj_gen.direct_trajectory(rbt.q, rbt.dq, views_pos[key]["pos"], views_pos[key]["orn_wxyz"], horizon)
            horizon *= 2

        view_U = np.array(view_U)
        success, idx, eef_pos, eef_quat = run_action(rbt, view_U, 20)
        rbt.active_controller.send_command(np.zeros(7))
        
        img_cv, depth_cv, depth_scale = camera.get_rgb_depth()
        depth_cv = depth_cv * depth_scale

        current_pose = get_camera_pose(rbt)

        pc_fused, pc_colors_fused = add_view2pc(pc_fused, pc_colors_fused, reference_pose, current_pose, new_gbr=img_cv, 
                                                new_depth=depth_cv, cam_intrisic=intrinsic, regularize=True, voxel_size=0.003)
        pc2_msg = format_pointcloud_msg(pc_fused, pc_colors_fused)
        bgr_msg = bridge.cv2_to_imgmsg(img_init, encoding="bgr8")

        orn, pos, opening, score, detected, detected_with_collision = generate_grasps_client(pc2_msg, bgr_msg)

        print("detected :", detected)
        if (detected or detected_with_collision) and opening > 0.03:
            grasp_pos_world, grasps_orn_world_xyzw = poseCam2World(pos, orn, reference_pose)
            grasp_orn_world_wxyz = convert_quat(grasps_orn_world_xyzw, to="wxyz")

            print("--------------------------------------------------------------------------")
            print("\n\ngrasps in world frame :\n pos :", grasp_pos_world, "\n grasps_orn_world :", grasps_orn_world_xyzw, "\ngrasp world rot",  Rotation.from_quat(grasps_orn_world_xyzw).as_euler("xyz", degrees=True))
            print("------------------------------------ --------------------------------------")
            
            break
        # break

grasp_horizon = 60
grasp_jpos, grasp_x_pos, grasp_U, grasp_Ks, grasp_ds, pos_dif, orn_dif = traj_gen.grasping_trajectory(rbt.q, rbt.dq, grasp_pos_world, grasp_orn_world_wxyz, grasp_horizon)

grasp_q = grasp_jpos[-1]
grasp_dq = np.zeros_like(grasp_q)
dispose_jpos, dispose_x_pos, dispose_U, dispose_Ks, dispose_ds, pos_dif, orn_dif = traj_gen.dispose_trajectory(grasp_q, grasp_dq, grasp_pos_world, grasp_orn_world_wxyz, dispose_pos, dispose_orn_wxyz, 120)
iLQR.plot_trajectory(init_pos, grasp_pos_world, grasp_x_pos, dispose_x_pos)



rbt.stop_controller()
rbt.error_recovery()
rbt.switch_controller("joint_velocity_controller")
rbt.active_controller


rbt.gripper.move(width=opening + 0.015)


grasp_U = np.array(grasp_U)
success, idx, eef_pos, eef_quat = run_action(rbt, grasp_U[:-30], 20)
rbt.active_controller.send_command(np.zeros(7))
time.sleep(1)
success, idx, eef_pos, eef_quat = run_action(rbt, grasp_U[-30:], 20)
rbt.active_controller.send_command(np.zeros(7))
rbt.gripper.move(width=opening-0.015)

# print("rbt end effector pos:", rbt.model.ee_pos_rel())
# print("rbt end effector quat (wxyz):", rbt.model.ee_orn_rel())
# print("eef rot",  Rotation.from_quat(rbt.model.ee_orn_rel()).as_euler("xyz", degrees=True))
# print("--------------------------------------------------------------------------")
# print("\n\ngrasps in world frame :\n pos :", grasp_pos_world, "\n grasps_orn_world :", grasps_orn_world_xyzw, "\ngrasp world rot",  Rotation.from_quat(grasps_orn_world_xyzw).as_euler("xyz", degrees=True))
# print("--------------------------------------------------------------------------")
# print("\n\ILQR final pose in world frame :\n pos :", grasp_x_pos[-1][:3], "\n grasps_orn_world wxyz:", convert_quat(grasp_x_pos[-1][3:], to="xyzw"), "\ngrasp world rot",  Rotation.from_quat(convert_quat(grasp_x_pos[-1][3:], to="xyzw")).as_euler("xyz", degrees=True))


dispose_U = np.array(dispose_U)
success, idx, eef_pos, eef_quat = run_action(rbt, dispose_U, 20)
rbt.active_controller.send_command(np.zeros(7))
rbt.gripper.move(width=0.07)



rbt.gripper.homing()


return_horizon = 60
return_jpos, return_x_pos, return_U, return_Ks, return_ds, pos_dif, orn_dif = traj_gen.direct_trajectory(rbt.q, rbt.dq, init_pos, init_orn_wxyz, return_horizon)
return_U = np.array(return_U)
success, idx, eef_pos, eef_quat = run_action(rbt, return_U, 20)
rbt.active_controller.send_command(np.zeros(7))


rbt.stop_controller()