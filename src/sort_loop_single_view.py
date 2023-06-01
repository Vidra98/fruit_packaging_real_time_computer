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

from utils.camera_utils import RealCamera
from utils.transform_utils import *
from utils.iLQR_wrapper import iLQR
import argparse
from scipy.spatial.transform import Rotation

import time

from sensor_msgs.msg import Image
import rospy

plt.switch_backend('QT5Agg')
import mayavi.mlab as mlab

from utils.visualisation_utils import *

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

def generate_grasp_client(image, depth):
    rospy.wait_for_service('generate_grasps')
    try:
        generate_grasp = rospy.ServiceProxy('generate_grasps', contactGraspnet)
        resp1 = generate_grasp(image, depth)
        return resp1.quat, resp1.pos, resp1.opening.data, resp1.detected.data
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()

    # Load camera
    camera = RealCamera()
    camera.start()

    #retrieve image and depth
    for i in range(15):
        rgb, depth_image, depth_scale = camera.get_rgb_depth()

    rgb, depth_image, depth_scale = camera.get_rgb_depth()

    max_gripper_width = 0.07
    depth_scale = 0.001 # depth a publish under millimeter precision
    dispose_pos = np.array([0.1, 0.66, 0.1])
    dispose_orn_wxyz = np.array([0, 1, 0.35, 0])

    # #Connect to the robot
    rospy.init_node("python_node",anonymous=True)
    rbt = Robot("panda",use_gripper=True)

    init_pos = rbt.model.ee_pos_rel()
    init_orn_wxyz = rbt.model.ee_orn_rel()

    traj_gen = iLQR(rbt)

    trajectory_threshold = 0.015
    cam_K = camera.getIntrinsic()[0]
    z_range = [0.2, 5]

    rbt.stop_controller()
    rbt.error_recovery()
    rbt.switch_controller("joint_velocity_controller")
    rbt.active_controller
    rbt.gripper.homing()
    
    for i in range(5):
        ee_pose = np.eye(4)
        ee_pose[:3,:3] = quat2mat(convert_quat(rbt.model.ee_orn_rel(), to="xyzw")) #xyzw
        ee_pose[:3,3] = rbt.model.ee_pos_rel() 

        ee2hand = np.eye(4)
        ee2hand[2,3] = -0.1034

        hand2camera_pos = np.array([0.0488546636437146,-0.03384417860749521,0.0512776975002817]) 
        hand2camera_quat = [0.012961267509189803,-0.0012768531849757236,0.7052247395136084,0.708864191484139] #xyzw 
        hand2camera_mat = Rotation.from_quat(hand2camera_quat).as_matrix()

        hand2camera = np.eye(4)
        hand2camera[:3,:3] = hand2camera_mat
        hand2camera[:3,3] = hand2camera_pos

        camera_pose_world = ee_pose @ ee2hand @ hand2camera

        eef_pos, eef_quat = [], []

        img_cv, depth_image, depth_scale = camera.get_rgb_depth()
        depth_cv = depth_image * depth_scale

        #Ros CV bridge to convert data from opencv to ROSImage
        bridge = CvBridge()

        detected = False
        detection_attempt = 0
        detection_attempt_threshold = 5
        while not detected:

            if detection_attempt > detection_attempt_threshold:
                print("no grasping target detected")
                sys.exit()

            # Call to the server
            orn, pos, opening, detected = generate_grasp_client(bridge.cv2_to_imgmsg(img_cv, "bgr8"), bridge.cv2_to_imgmsg(depth_cv, "64FC1"))

            if not detected:
                detection_attempt += 1
                print("detection attempt number ",detection_attempt)
                continue

            grasp_pos_cam = np.array([pos.x, pos.y, pos.z])
            # grasp_orn_cam_xyzw_raw = np.array([orn.x, orn.y, orn.z, orn.w])
            grasp_orn_cam_xyzw_raw = Rotation.from_quat(np.array([orn.x, orn.y, orn.z, orn.w]))
            rot = np.array([0., 0., 90])
            grasp_rot_tmp = grasp_orn_cam_xyzw_raw.as_euler("XYZ", degrees=True)
            grasp_orn_eul = grasp_orn_cam_xyzw_raw.as_euler("XYZ", degrees=True) + rot
            grasp_orn_cam_xyzw = Rotation.from_euler("XYZ", grasp_orn_eul, degrees=True).as_quat()
            grasp_orn_cam_wxyz = convert_quat(grasp_orn_cam_xyzw, to="wxyz")
            grasp_mat_cam = quat2mat(grasp_orn_cam_xyzw)

            grasp_pos_world = camera_pose_world[:3,:3] @ grasp_pos_cam + camera_pose_world[:3,3]
            grasp_mat_world = camera_pose_world[:3,:3] @ grasp_mat_cam
            grasp_orn_world_xyzw = mat2quat(grasp_mat_world)
            grasp_orn_world_wxyz =  convert_quat(grasp_orn_world_xyzw, to="wxyz")

            print("--------------------------------------------------------------------------")
            print("grasp in camera frame :\n pos :", grasp_pos_cam, "\n grasp_orn_cam :", grasp_orn_cam_xyzw, "\ngrasp world rot",  Rotation.from_quat(grasp_orn_cam_xyzw).as_euler("xyz", degrees=True))
            print("\n\ngrasp in world frame :\n pos :", grasp_pos_world, "\n grasp_orn_world :", grasp_orn_world_xyzw, "\ngrasp world rot",  Rotation.from_quat(grasp_orn_world_xyzw).as_euler("xyz", degrees=True))
            print("--------------------------------------------------------------------------")
            
            
        # cv2.imshow('rgb', rgb)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        pred_grasp_cam = np.eye(4)
        pred_grasp_cam[:3,:3] = grasp_mat_cam
        pred_grasp_cam[:3,3] = grasp_pos_cam
        pred_grasp_cam = {1.0: np.array([pred_grasp_cam])}

        
        grasp_jpos, grasp_x_pos, grasp_U, grasp_Ks, grasp_ds = traj_gen.grasping_trajectory(rbt.q, rbt.dq, grasp_pos_world, grasp_orn_world_wxyz, 120)
        grasp_q = grasp_jpos[-1]
        grasp_dq = np.zeros_like(grasp_q)
        dispose_jpos, dispose_x_pos, dispose_U, dispose_Ks, dispose_ds = traj_gen.dispose_trajectory(grasp_q, grasp_dq, grasp_pos_world, grasp_orn_world_wxyz, dispose_pos, dispose_orn_wxyz, 120)
        print("dif between iLQR grasp and neural net approach :", grasp_x_pos[-1,:3] - grasp_pos_world)
        if np.sum(grasp_x_pos[-1,:3] - grasp_pos_world) > trajectory_threshold:
            print("Trajectory generator generated doesn't reach grasp target")
            continue
        iLQR.plot_trajectory(init_pos, grasp_pos_world, grasp_x_pos, dispose_x_pos)
        print("dif between iLQR grasp and neural net approach :", grasp_x_pos[-1,:3] - grasp_pos_world)

        img_pc = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        pc_full, pc_segments, pc_colors = extract_point_clouds(depth=depth_cv, K=cam_K, rgb=img_pc, z_range=z_range)

        for key in pred_grasp_cam:
            pred_grasp_cam[key][:,:3,3] -= 0.1034 * pred_grasp_cam[key][:,:3,2]


        pred_grasp_cam_raw = np.eye(4)
        pred_grasp_cam_raw[:3,:3] = quat2mat(np.array([orn.x, orn.y, orn.z, orn.w]))
        pred_grasp_cam_raw[:3,3] = grasp_pos_cam
        pred_grasp_cam_raw = {1.0: np.array([pred_grasp_cam_raw])}
        for key in pred_grasp_cam_raw:
            pred_grasp_cam_raw[key][:,:3,3] -= 0.1034 * pred_grasp_cam_raw[key][:,:3,2]
        cv2.imshow("rgb", img_cv)
        visualize_grasps(pc_full, pred_grasp_cam_raw, plot_opencv_cam=True, pc_colors=pc_colors, add_grasps=None)
        #camera.stop()

        # trajectory

        if opening + 0.01 >= max_gripper_width :
            rbt.gripper.move(width=opening)
        else:
            rbt.gripper.move(width=opening+0.015)
        grasp_U = np.array(grasp_U)
        success, idx, eef_pos, eef_quat = run_action(rbt, grasp_U[:80], 20)
        rbt.active_controller.send_command(np.zeros(7))
        key = cv2.waitKey(0)
        if key == ord("q"):
            sys.exit()

        success, idx, eef_pos, eef_quat = run_action(rbt, grasp_U[80:], 20)
        rbt.active_controller.send_command(np.zeros(7))
        rbt.gripper.move(width=opening-0.015)
        print("\n\ngrasp in world frame :\n pos :", grasp_pos_world, "\n grasp_orn_world :", grasp_orn_world_xyzw, "\ngrasp world rot",  Rotation.from_quat(grasp_orn_world_xyzw).as_euler("xyz", degrees=True))
        print("final iLQR pos", grasp_x_pos[-1])
        print("rbt ee pos :", rbt.model.ee_pos_world(), "\n orn :", rbt.model.ee_orn_world())
        key = cv2.waitKey(0)
        if key == ord("q"):
            sys.exit()

        dispose_U = np.array(dispose_U)
        success, idx, eef_pos, eef_quat = run_action(rbt, dispose_U, 20)
        rbt.active_controller.send_command(np.zeros(7))
        rbt.gripper.move(width=0.07)
        key = cv2.waitKey(0)
        if key == ord("q"):
            sys.exit()

        return_jpos, return_x_pos, return_U, return_Ks, return_ds = traj_gen.grasping_trajectory(rbt.q, rbt.dq, init_pos, init_orn_wxyz, 120)
        return_U = np.array(return_U)
        success, idx, eef_pos, eef_quat = run_action(rbt, return_U, 20)
        rbt.active_controller.send_command(np.zeros(7))
        cv2.destroyAllWindows()