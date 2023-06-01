#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import cv2

from scipy.spatial.transform import Rotation 

import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose, PoseArray
from std_msgs.msg import Bool, String

from sensor_msgs.msg import Image, CameraInfo

import message_filters
import time

class GridGenerator:
    def __init__(self, parameter):
        self.markerOrigin_received_time = None
        self.markerOrigin_pos = None
        self.markerOrigin_mat = None

        self.markerTop_pos = None
        self.markerTop_mat = None
        self.markerTop_received_time = None

        self.rot_mat = None

        self.time_thres = parameter["time_threshold"]
        self.NumCellsx = parameter["NumCellsx"]
        self.NumCellsy = parameter["NumCellsy"]
        self.NumCellsz = int(self.NumCellsx - 1 + self.NumCellsy - 1)

        self.bridge = CvBridge()

        self.markers = {}

    def publish_image(self, grid_np, grid_flag, pub_image):
        image_msg_received = rospy.wait_for_message("/aruco_simple/result", Image, timeout=0.1)
        image = self.bridge.imgmsg_to_cv2(image_msg_received, desired_encoding="passthrough")
 
        camera_info = rospy.wait_for_message("/camera/color/camera_info", CameraInfo, timeout=0.1)
        intrinsic = np.array(camera_info.K).reshape((3, 3))
        
        grid_pxl = np.empty((grid_np.shape[0], grid_np.shape[1], 2))
        for i in range(grid_np.shape[0]):
            for j in range(grid_np.shape[1]):
                xyw = intrinsic @ np.array(grid_np[i, j]["pos"])
                grid_pxl[i, j] = xyw[:2]/xyw[2]
                grid_pxl[i, j] = grid_pxl[i, j].astype(int)
        # Plot grid on image
        image_plt = image.copy()
        for i in range(grid_pxl.shape[0]):
            for j in range(grid_pxl.shape[1]):
                if grid_flag[i, j] == 1:
                    cv2.circle(image_plt, (int(grid_pxl[i, j][0]), int(grid_pxl[i, j][1])), 5, (0, 0, 255), -1)
                else:
                    cv2.circle(image_plt, (int(grid_pxl[i, j][0]), int(grid_pxl[i, j][1])), 5, (0, 255, 0), -1)

        marker26_pxl = intrinsic @ np.array(self.markers["markerOrigin"]["pos"])
        marker26_pxl/=marker26_pxl[2]

        marker29_pxl = intrinsic @ np.array(self.markers["markerTop"]["pos"])
        marker29_pxl/=marker29_pxl[2]

        cv2.circle(image_plt, (int(marker26_pxl[0]), int(marker26_pxl[1])), 6, (255, 0, 0), 2)

        cv2.circle(image_plt, (int(marker29_pxl[0]), int(marker29_pxl[1])), 6, (255, 0, 0), 2)

        image_msg = self.bridge.cv2_to_imgmsg(image_plt, encoding="rgb8")
        pub_image.publish(image_msg)


    def generate_grid(self, args):
        marker_vector = self.markers["markerTop"]["pos"] - self.markers["markerOrigin"]["pos"]

        GridSize = self.rot_mat.T @ marker_vector

        grid_np = np.empty((int(self.NumCellsx), int(self.NumCellsy)), dtype=dict)
        grid_flag = np.ones((int(self.NumCellsx), int(self.NumCellsy)), dtype=np.uint8)*255

        grid = PoseArray()
        grid.header.frame_id = "camera_link"
        grid.header.stamp = rospy.Time.now()

        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = self.markers["markerOrigin"]["pos"][0], self.markers["markerOrigin"]["pos"][1], self.markers["markerOrigin"]["pos"][2]
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = 0, 0, 0, 1
        grid.poses.append(pose)

        grid_np[0, 0] = {"pos": self.markers["markerOrigin"]["pos"]}
        # grid_flag[0, 0] = 0

        grid_offsetx = self.rot_mat @ np.array([GridSize[0]/(self.NumCellsx - 1), 0, 0])
        grid_offsety = self.rot_mat @ np.array([0, GridSize[1]/(self.NumCellsy - 1), 0])
        grid_offsetz = self.rot_mat @ np.array([0, 0, GridSize[2]/(self.NumCellsz)])

        for x in range(int(self.NumCellsx)):
            for y in range(int(self.NumCellsy)):
                if x%2 is not 0:
                    offset = 0.5
                else:
                    offset = 0
                
                cell_pos = grid_np[0, 0]["pos"] + (x)*grid_offsetx + (y + offset)*grid_offsety + (x+y)*grid_offsetz
                grid_np[x, y] = {"pos": cell_pos}

                pose = Pose()
                pose.position.x, pose.position.y, pose.position.z = cell_pos[0], cell_pos[1], cell_pos[2]
                pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = 0, 0, 0, 1
                rospy.loginfo(grid_np)
                grid.poses.append(pose)

        grid_flag[0, 0] = 0
        grid_flag[-1, -1] = 0
        
        pub_PoseArray = args[0]
        pub_img = args[1]
        pub_DisposabilityGrid = args[2]

        pub_PoseArray.publish(grid)
        pub_DisposabilityGrid_msg = self.bridge.cv2_to_imgmsg(grid_flag, encoding="mono8")
        pub_DisposabilityGrid.publish(pub_DisposabilityGrid_msg)
        
        parameter = args[3]
        if parameter["publish_image"]:
            self.publish_image(grid_np, grid_flag, pub_img)

    def markerOrigin_callback(self, pose, args):
        self.markerOrigin_received_time = time.time()
        if self.markerTop_received_time is None or self.markerTop_mat is None or self.markerTop_pos is None:
            return
        if abs((self.markerOrigin_received_time - self.markerTop_received_time)) > self.time_thres:
            self.markerTop_received_time = None
            self.markerTop_mat = None
            self.markerTop_pos = None
            return
        
        self.markerOrigin_pos = np.array([pose.position.x, pose.position.y, pose.position.z])
        self.markers = {"markerOrigin": {"pos": self.markerOrigin_pos},
                        "markerTop": {"pos": self.markerTop_pos}}

        markerOrigin_quat = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
        self.markerOrigin_mat = Rotation.from_quat(markerOrigin_quat).as_matrix()

        self.rot_mat = (self.markerOrigin_mat + self.markerTop_mat) / 2

        self.generate_grid(args)

    def markerTop_callback(self, pose):
        self.markerTop_received_time = time.time()
        self.markerTop_pos = np.array([pose.position.x, pose.position.y, pose.position.z])
        markerTop_quat = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
        self.markerTop_mat = Rotation.from_quat(markerTop_quat).as_matrix()
 
if __name__ == '__main__':
    parameter = rospy.get_param('grid_generator')
    gridGen = GridGenerator(parameter)

    node_name = "grid_generator"
    rospy.init_node(node_name)
    pub_PoseArray = rospy.Publisher(node_name+'/Posearray_pub', PoseArray, queue_size=2)
    pub_img = rospy.Publisher(node_name+'/result', Image, queue_size=2)
    pub_DisposabilityGrid = rospy.Publisher(node_name+'/disposability_grid', Image, queue_size=2)

    rospy.Subscriber(parameter["aruco_pose_topic"], Pose, gridGen.markerOrigin_callback, callback_args=[pub_PoseArray, pub_img, pub_DisposabilityGrid, parameter], queue_size=1)
    rospy.Subscriber(parameter["aruco_pose2_topic"], Pose, gridGen.markerTop_callback, queue_size=1)

    rospy.spin()