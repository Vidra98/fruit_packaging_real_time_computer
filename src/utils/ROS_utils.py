import rospy
from contact_grasp.srv import contactGraspnetPointcloud2, contactGraspnetPointcloud2Response
from sensor_msgs.msg import PointField
from std_msgs.msg import Header
from sensor_msgs import point_cloud2
import numpy as np
from scipy.spatial.transform import Rotation 
from utils.transform_utils import quat2mat, convert_quat
import json
from cv_bridge import CvBridge, CvBridgeError
import time

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

def run_action(rbt, actions, control_freq, eef_pos=None, eef_quat=None, segmentation_type=None, show_agentview=False, object_range=[5,8]):
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
    return success, idx, eef_pos, eef_quat

def get_camera_pose(rbt, ee_depth=-0.1034):
    """ Get camera pose in robot base frame
    """
    world2ee_pose = np.eye(4)
    world2ee_pose[:3,:3] = quat2mat(convert_quat(rbt.model.ee_orn_rel(), to="xyzw")) #xyzw
    world2ee_pose[:3,3] = rbt.model.ee_pos_rel() 

    # image are retrieved in camera_optical_frame reference, so we need to add a transformation from camera_optical_frame to camera_link
    # This rotation added to the panda hand rotation gives the identity matrix.
    ee2flange = np.eye(4)
    ee2flange[2,3] = ee_depth

    with open('config/camera_calibration.json') as json_file:
        camera_calibration = json.load(json_file)

    camera_type = "L515" #D415 or L515

    hand2camera_pos = np.array(camera_calibration[camera_type]["pos"])
    hand2camera_quat = camera_calibration[camera_type]["quat_xyzw"] #xyzw

    # TODO todelete
    # #L515
    # hand2camera_pos = np.array([0.08329189218278059, 0.0014213145240625528, 0.0504764049956106]) 
    # hand2camera_quat = [0.01521805627198811, 0.00623363612254646, 0.712108725756912, 0.7018765669580811] #xyzw 

    hand2camera_mat = Rotation.from_quat(hand2camera_quat).as_matrix()

    flange2opticalFrame = np.eye(4)
    flange2opticalFrame[:3,:3] = hand2camera_mat
    flange2opticalFrame[:3,3] = hand2camera_pos

    current_pose = world2ee_pose @ ee2flange @ flange2opticalFrame

    return current_pose

class gridRegistrator():
    def __init__(self, rbt):
        self.bridge = CvBridge()
        self.poses = None
        self.disposability_grid = None
        self.registration_time = None
        #rbt object, use to get pos of ee when grid is received
        self.rbt = rbt 
        self.acq_matrix = None
        self.acq_pos = None
        self.acq_orn_wxyz = None

    def callback(self, poseArray, disposability_grid_msg):
        if abs(np.sum(self.rbt.dq)) > 0.005:
            return
        self.poses = poseArray.poses
        self.disposability_grid = self.bridge.imgmsg_to_cv2(disposability_grid_msg, "mono8")
        self.registration_time = time.time()
        self.acq_matrix = get_camera_pose(self.rbt, ee_depth=-0.10340)
        self.acq_pos =  self.rbt.model.ee_pos_rel()
        self.acq_orn_wxyz = self.rbt.model.ee_orn_rel()

    def get_acq_coord(self):
        return self.acq_matrix, self.acq_pos, self.acq_orn_wxyz

    def get_poses(self):
        return self.poses
    
    def get_disposability_grid(self):
        return self.disposability_grid
    
    def get_registration_time(self):
        return self.registration_time
    
    def reset(self):
        self.poses = None
        self.disposability_grid = None
        self.registration_time = None
        self.acq_matrix = None
    
    def clear_grid(self):
        self.disposability_grid = np.ones_like(self.disposability_grid) * 255
        self.disposability_grid[0,0] = 0
        self.disposability_grid[-1, -1] = 0

    def set_cell_occupancy(self, cell_idx, occupancy):
        if self.disposability_grid is None:
            print("No disposability grid received yet")
            return False
        else:
            self.disposability_grid[cell_idx] = occupancy
            return True
        
    def get_first_free_cell(self):
        if self.disposability_grid is None or self.poses is None:
            print("No grid received yet")
            return None, None
        self.freeCells = np.argwhere(self.disposability_grid > 0)
        if self.freeCells.size == 0:
            print("No free cell")
            return None, None
        
        poses_reshaped = np.reshape(self.poses, np.shape(self.disposability_grid))
        poseFree = poses_reshaped[self.freeCells[0, 0], self.freeCells[0, 1]]
        posFree = poseFree.position
        posFree_np = self.acq_matrix @ np.array([posFree.x, posFree.y, posFree.z, 1])
        return posFree_np[:3], (self.freeCells[0, 0], self.freeCells[0, 1])
