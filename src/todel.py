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

sys.path.append("/home/vdrame/catkin_ws/src/py_panda/PyPanda")
from PyPanda import Robot
import rospy
from PyPanda import Utils

from utils.camera_utils import RealCamera
from utils.transform_utils import *
import argparse
from scipy.spatial.transform import Rotation

import time

from sensor_msgs.msg import Image
import rospy
def generate_grasps_client(image, depth):
    rospy.wait_for_service('generate_grasps')
    try:
        generate_grasps = rospy.ServiceProxy('generate_grasps', contactGraspnet)
        resp1 = generate_grasps(image, depth)
        return resp1.quat, resp1.pos, resp1.opening.data, resp1.detected.data
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)
# Load camera
camera = RealCamera()
camera.start()

#retrieve image and depth
for i in range(15):
    rgb, depth_image, depth_scale = camera.get_rgb_depth()

rgb, depth_image, depth_scale = camera.get_rgb_depth()
# cv2.imwrite("img.png", rgb)
# cv2.imwrite("depth.png", depth_image)
# cv2.imwrite("depth_rescale.png", depth_image*depth_scale)

depth_scale = 0.001 # depth a publish under millimeter precision
dispose_pos = np.array([0.1, 0.66, 0.1])
dispose_orn_wxyz = np.array([0, 1, 0.35, 0])

eef_pos, eef_quat = [], []

img_cv, depth_image, depth_scale = camera.get_rgb_depth()
depth_cv = depth_image * depth_scale

#Ros CV bridge to convert data from opencv to ROSImage
bridge = CvBridge()

# Call to the server
orn, pos, opening, detected = generate_grasps_client(bridge.cv2_to_imgmsg(img_cv, "bgr8"), bridge.cv2_to_imgmsg(depth_cv, "64FC1"))
# orn, pos = generate_grasps_client(img, depth)

if detected:
    grasps_pos_cam = np.array([pos.x, pos.y, pos.z])
    grasps_orn_cam_xyzw = np.array([orn.x, orn.y, orn.z, orn.w])
    grasps_orn_cam_wxyz = np.array([orn.w, orn.x, orn.y, orn.z])
    grasps_mat_cam = quat2mat(grasps_orn_cam_xyzw)

    grasps_pos_world = grasps_pos_cam 
    grasps_mat_world = grasps_mat_cam
    grasps_orn_world_xyzw = mat2quat(grasps_mat_world)
    grasps_orn_world_wxyz =  convert_quat(grasps_orn_world_xyzw, to="wxyz")

    print("--------------------------------------------------------------------------")
    print("grasps in camera frame :\n pos :", grasps_pos_cam, "\n grasps_orn_cam :", grasps_orn_cam_xyzw, "\ngrasp world rot",  Rotation.from_quat(grasps_orn_cam_xyzw).as_euler("xyz", degrees=True))
    print("\n\ngrasps in world frame :\n pos :", grasps_pos_world, "\n grasps_orn_world :", grasps_orn_world_xyzw, "\ngrasp world rot",  Rotation.from_quat(grasps_orn_world_xyzw).as_euler("xyz", degrees=True))
    print("--------------------------------------------------------------------------")

# cv2.imshow('rgb', rgb)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

pred_grasps_cam = np.eye(4)
pred_grasps_cam[:3,:3] = grasps_mat_world
pred_grasps_cam[:3,3] = grasps_pos_world
pred_grasps_cam = {1.0: np.array([pred_grasps_cam])}
pred_grasps_cam

plt.get_backend()
plt.switch_backend('QT5Agg')
plt.get_backend()
import mayavi.mlab as mlab
#print(mlab.get_engine())

from utils.visualisation_utils import *
cam_K = camera.getIntrinsic()[0]
z_range = [0.2, 5]
pc_full, pc_segments, pc_colors = extract_point_clouds(depth=depth_cv, K=cam_K, rgb=img_cv, z_range=z_range)

for key in pred_grasps_cam:
    print(key)
    #pred_grasps_cam[key][:,:3,3] -= 0.1034 * pred_grasps_cam[key][:,:3,2]

#show_image(self.rgb, self.segmap)
plt.switch_backend('QT5Agg')
visualize_grasps(pc_full, pred_grasps_cam, plot_opencv_cam=True, pc_colors=pc_colors, add_grasps=None)
