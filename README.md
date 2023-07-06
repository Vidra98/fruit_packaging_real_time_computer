# Installation

## Prerequisites
- Two machines: Machine A with a real time kernel and Machine B with CUDA
- Machine A: Install PyLQR, PyPanda, and RLI-camera, if you are using the IDIAP Desktop, it is already install.
- Machine B: Install yolov8_seg_ros and CONTACT_GRASP_ROS

## Install ROS on both machine:
## Install PyLQR, PyPanda, and RLI-camera on Machine A
Follow the steps of these differents repository:

- https://gitlab.idiap.ch/rli/ilqr_planner
- https://gitlab.idiap.ch/rli/py_panda
- https://gitlab.idiap.ch/rli/rli-camera
   
## Install YOLOv8_ROS and CONTACT_GRASP_ROS on Machine B
Follow the steps of these differents repository:

- https://github.com/Vidra98/yolov8_seg_ros
- https://github.com/Vidra98/contact_grasp_ros

## Communicate between Machine A and Machine B
To communicate between Machine A and Machine B, you need to ensure that both machines communicate. I used a direct connexion by ethernet cable. You can use the IP addresses of the machines to establish communication.

Here is a tuto to set the communication :

And one to show how to communicate between machine (http://wiki.ros.org/ROS/Tutorials/MultipleMachines). Basically, you will need to change your ROS master to the one the computer A. 

"""
export ROS_MASTER_URI=[Master IP address]:11311
export ROS_IP=[Slave IP address]
"""

# Package Libraries
The following package libraries are required for running the packaging code:

- numpy
- cv2 (OpenCV)
- tqdm
- scipy
- rospy
- open3d

Make sure these libraries are installed on your system.

# Usage

## Launch Robot and Specify Number of Cells
1. Turn on Machine A and the robot.
2. Go to the robot IP address on a browser, unlock it and activate FCL
Use the robot launch file:
"""
roslaunch :HEYYYYYYYYYYYYYYYYYYYYYYY don't forget to add the command
"""

## Register Viewpoint
1. Set up the cameras and sensors on Machine A to capture the viewpoint using "register_viewpoint.ipynb".
2. Register the viewpoint using the provided script or notebook. Make sure to provide the necessary inputs for the viewpoint registration.

## Script Execution or Notebook Launch
You have two options for executing the code: running the provided script or launching the notebook.
