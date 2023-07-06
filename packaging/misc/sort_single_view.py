#!/usr/bin/env python
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

from utils.camera_utils import RealCamera
from utils.transform_utils import *
import argparse
from scipy.spatial.transform import Rotation

def run_sim(rbt, actions, control_freq, eef_pos=None, eef_quat=None, segmentation_type=None, show_agentview=False, object_range=[5,8]):
    success = False
    rate = rospy.Rate(int(control_freq))

    for idx, action in enumerate(actions):
        rbt.active_controller.send_command(action)
        rate.sleep()
        #env.sim.step()
        if eef_pos is not None:
            eef_pos.append(rbt.model.ee_pos_rel())
        if eef_quat is not None:
            eef_quat.append(rbt.model.ee_orn_rel())

    success = True
    rbt.stop_controller()
    return success, idx

def generate_grasps_client(image, depth):
        rospy.wait_for_service('generate_grasps')
        try:
            generate_grasps = rospy.ServiceProxy('generate_grasps', contactGraspnet)
            resp1 = generate_grasps(image, depth)
            return resp1.quat, resp1.pos
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)

class iLQR():
    def __init__(self, rbt):
        #Load URDf for KDL and iLQR solver
        self._control_freq = 20
        
        self._dof = len(rbt.q)
        self._nb_state_var = self._dof
        self._nb_ctrl_var = self._dof
        self._dt = 1/(2*self._control_freq)

        PATH_TO_URDF = "/home/vdrame/project/client/panda_description/urdf/panda_arm_robosuite.urdf"
        BASE_FRAME = "panda_link0"
        #TIP_FRAME = "panda_link8"
        TIP_FRAME = "panda_grasptarget"

        # initial joint config
        q0 = rbt.q
        dq0 = [0]*rbt.dq

        self._qMax = np.array([2.87,   1.75,  2.8973, -0.05,  2.8973,  3.75,   2.8973])
        self._qMin = np.array([-2.87, -1.75, -2.8973, -3.05, -2.8973, -0.015, -2.8973])
        self._dqMax = np.array([2., 2., 2., 2., 2.6, 2.6, 2.6])

        self._rbt_KDL = KDLRobot(PATH_TO_URDF, BASE_FRAME, TIP_FRAME, q0, dq0)

        self.check_KDL_robosuite_alignement(rbt)

    def check_KDL_robosuite_alignement(self, rbt, verbose=False):
        # initial joint config
        q0 = rbt.q
        dq0 = [0]*rbt.dq.shape[0]

        #Set robot starting position for retrieving trajectory
        self._rbt_KDL.set_conf(q0, dq0, False)

        pos_dif = rbt.model.ee_pos_rel() - self._rbt_KDL.get_ee_pos()
        orn_dif = quat_distance(self._rbt_KDL.get_ee_orn(), rbt.model.ee_orn_rel())

        if verbose or np.any(np.abs(orn_dif)>0.1) or np.any(np.abs(pos))>0.01:
            print("difference between iLQR model ee and real robot ee")
            print(f"quat (zyzw) :{orn_dif}")
            print(f"pos:{pos_dif}")
        
    def generate_trajectory(self, rbt, keypoints, horizon):
        
        # initial joint config
        q0 = rbt.q
        dq0 = rbt.dq

        #Set robot starting position for retrieving trajectory
        self._rbt_KDL.set_conf(q0, dq0, False)

        # Each control signals have a penalty of 1e-5
        cmd_penalties = (np.ones(self._nb_ctrl_var)*1e-5).tolist()

        # It is not mandatory to set the limits, if you do not know them yet or do not want to use them. You can use this constructor:
        #sys = PosOrnPlannerSys(self._rbt_KDL, keypoints, cmd_penalties, horizon, 1, dt)
        sys = PosOrnPlannerSys(self._rbt_KDL, keypoints, cmd_penalties, self._qMax, self._qMin, horizon, 1, self._dt)

        u0_t = np.array([0]*(self._nb_ctrl_var-1) + [0])
        u0 = np.tile(u0_t, horizon-1)

        mu = sys.get_mu_vector(False)
        Q = sys.get_Q_matrix(False)

        planner = ILQRRecursive(sys)

        # callback to notify python code of the solver evolution
        cb = PythonCallbackMessage()

        # solver input :std::vector<VectorXd>& U0, int nb_iter, bool line_search, bool early_stop, CallBackMessage* cb
        jpos, x_pos, U, Ks, ds = planner.solve(
            u0.reshape((-1, self._nb_ctrl_var)), 50, True, True, cb)

        jpos, x_pos = np.asarray(jpos), np.asarray(x_pos)

        return jpos, x_pos, U, Ks, ds

    def grasping_trajectory(self, rbt, grasps_pos_base, grasps_orn, horizon = 120):
        Qtarget1 = np.diag([1,  # Tracking the x position
                            1,  # Tracking the y position
                            .1,  # Tracking the z position
                            .1,  # Tracking orientation around x axis
                            .1,  # Tracking orientation around y axis
                            .1])  # Tracking orientation around z axis

        print("grasp_orn", grasps_orn)
        print("grasps_pos_base", grasps_pos_base)
        print("rbt", rbt)
        rotation_mat_grasps = quat2mat(grasps_orn)
        target1_pos_base = grasps_pos_base.copy()
        target1_pos_base -= rotation_mat_grasps[:3,2] * 0.1 #rbt.get_ee_pos()[2].copy()
        target1_orn = grasps_orn.copy()

        target2_pos_base = grasps_pos_base.copy()
        target2_orn = grasps_orn.copy()

        target1_discrete_time = 2*horizon//3 - 1
        keypoint_1 = PosOrnKeypoint(target1_pos_base, target1_orn, Qtarget1, target1_discrete_time)

        Qtarget2 = np.diag([1, 1, 1, 1, 1, 1])
        target2_discrete_time = horizon - 1
        keypoint_2 = PosOrnKeypoint(target2_pos_base, target2_orn, Qtarget2, target2_discrete_time)

        #keypoints = [keypoint_1, keypoint_2]
        keypoints = [keypoint_1]

        return self.generate_trajectory(rbt, keypoints, horizon)

    def plot_trajectory(current_pos, target_pos, trajectory):
        # Plot the trajectory
        #fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_title("Planned trajectory")

        ax.scatter3D(current_pos[0], current_pos[1], current_pos[2], c='r', marker='o')
        ax.scatter3D(target_pos[0], target_pos[1], target_pos[2], c='g', marker='o')
        ax.plot3D(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 'green')

        # Plot the trajectory
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        ax.view_init(elev=30, azim=45)
        plt.legend()
        plt.show()

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()

    # #Connect to the robot
    rospy.init_node("python_node",anonymous=True)
    rbt = Robot("panda",use_gripper=True)

    print("rbt end effector pos:", rbt.model.ee_pos_rel())
    print("rbt end effector quat (wxyz):", rbt.model.ee_orn_rel())
    print("--------------------------------------------------------------------------")

    ee_pose = np.eye(4)
    ee_pose[:3,:3] = quat2mat(convert_quat(rbt.model.ee_orn_rel(), to="xyzw")) #xyzw
    ee_pose[:3,3] = rbt.model.ee_pos_rel()
    print("eef rot",  Rotation.from_quat(convert_quat(rbt.model.ee_orn_rel(), to="xyzw")).as_euler("xyz", degrees=True))

    # hand2camera_pos = [0.04911211, -0.0331901, 0.07007729]
    # hand2camera_quat = [0.71081385, -0.00487722, 0.70335828, -0.00264716] #xyzw 
    ee2camera_pos = np.array([0.0488546636437146,-0.03384417860749521,0.0512776975002817]) +  ee_pose[:3,2] * 0.1034
    ee2camera_quat = [0.012961267509189803,-0.0012768531849757236,0.7052247395136084,0.708864191484139] #xyzw     
    ee2camera_mat = Rotation.from_quat(ee2camera_quat).as_matrix()
    print("camera rot",  Rotation.from_quat(ee2camera_quat).as_euler("xyz", degrees=True))

    camera_pose = np.eye(4)
    camera_pose[:3,:3] = ee2camera_mat
    camera_pose[:3,3] = ee2camera_pos

    extrinc_pose = ee_pose @ camera_pose

    print("camera_pose :", camera_pose)
    print("ee_pose :", ee_pose)
    print("extrinc_pose :", extrinc_pose)

    print("cam pose in world frame:", Rotation.from_matrix(extrinc_pose[:3,:3]).as_euler("xyz", degrees=True))
    print("extrinsic * [0, 0, 0.4]:", extrinc_pose@np.array([0, 0, 0.4, 1]))
    print("--------------------------------------------------------------------------")    

    traj_gen = iLQR(rbt)

    #TODO
    base_pos= [0., 0., 0.]
    base_orn= [0., 0., 0., 1.]

    eef_pos, eef_quat = [], []

    init_pos = rbt.model.ee_pos_rel()
    init_orn = rbt.model.ee_orn_rel()

    #Load camera
    camera = RealCamera()
    camera.start()

    #Ros CV bridge to convert data from opencv to ROSImage
    bridge = CvBridge()

    #retrieve image and depth
    for i in range(30):
    	rgb, depth_image, depth_scale = camera.get_rgb_depth()

    cv2.imshow('rgb', rgb)
    cv2.waitKey(0)
    depth = depth_image * depth_scale

    # path = "/home/vdrame/mujoco/contact_graspnet/test_data/result0.npy"
    # Call to the server
    orn, pos = generate_grasps_client(bridge.cv2_to_imgmsg(rgb, "bgr8"), bridge.cv2_to_imgmsg(depth, "64FC1"))
    grasps_p_cam = np.array([pos.x, pos.y, pos.z])
    grasps_orn_cam = np.array([orn.x, orn.y, orn.z, orn.w])
    rotation_mat_grasps = quat2mat(grasps_orn_cam)

    grasps_p_world = extrinc_pose[:3,:3] @ grasps_p_cam + extrinc_pose[:3,3]
    grasps_mat_world = extrinc_pose[:3,:3] @ rotation_mat_grasps
    grasps_orn_world = mat2quat(grasps_mat_world)
    print("--------------------------------------------------------------------------")    

    print("grasps in camera frame :\n pos :", grasps_p_cam, "\n grasps_orn_cam :", grasps_orn_cam)
    print("\n\ngrasps in world frame :\n pos :", grasps_p_world, "\n grasps_orn_world :", grasps_orn_world)
    cv2.destroyAllWindows()

    jpos, x_pos, U, Ks, ds = traj_gen.grasping_trajectory(rbt, grasps_p_world, grasps_orn_world, 120)

    print("x_pos final", x_pos[-1])
    iLQR.plot_trajectory(init_pos, grasps_p_world, x_pos)
    camera.stop()
