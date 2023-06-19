import pyrealsense2 as rs
import numpy as np
import cv2
import json
import rospy
from sensor_msgs import point_cloud2
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
import struct
from cv_bridge import CvBridge, CvBridgeError

class RealCameraROS:
    """
    Class for using the realsense camera with rospy
    """
    def __init__(self):
        try:
            rospy.init_node("python_node",anonymous=True)
        except:
            print("rospy already initialized")
        try:
            self.camera_info = rospy.wait_for_message("/camera/color/camera_info", CameraInfo, timeout=0.5)
            self.depth_info = rospy.wait_for_message("/camera/aligned_depth_to_color/camera_info", CameraInfo, timeout=0.5)
        except:
            print("list of found topics:")
            print(rospy.get_published_topics())
            rospy.logerr("Camera not publishing not found")
            return
        
        print("Camera topic found")

    def get_rgb_only(self):
        """Get the rgb image from the camera and convert it to cv2 format

        Returns:
            cv_image: cv2 image
        """
        bridge = CvBridge()
        img_data = rospy.wait_for_message("/camera/color/image_raw", Image, timeout=0.5)
        cv_image = bridge.imgmsg_to_cv2(img_data, "bgr8")

        if (self.camera_info.height, self.camera_info.width) != (self.depth_info.height, self.depth_info.width):
                cv_image = cv2.resize(cv_image, dsize=(self.depth_info.height, self.depth_info.width), interpolation=cv2.INTER_AREA)
        return cv_image

    def get_depth_only(self):
        """Get the depth image from the camera and convert it to cv2 format

        Returns:
            depth_image: cv2 image
        """
        bridge = CvBridge()
        depth_data = rospy.wait_for_message("/camera/aligned_depth_to_color/image_raw", Image, timeout=0.5)
        depth_image = bridge.imgmsg_to_cv2(depth_data, "16UC1") * 0.001
        return depth_image
    
    def get_rgb_depth(self):
        cv_image = self.get_rgb_only()
        depth_image = self.get_depth_only()
        return cv_image, depth_image, 1

    def depth2colormap(self, depth_image = None):
        if depth_image is None:
            depth_image = self.get_depth_only()

        return cv2.applyColorMap(cv2.convertScaleAbs(depth_image.copy(), alpha=0.03), cv2.COLORMAP_JET)

    def getIntrinsic(self):
        return np.array(self.camera_info.K).reshape((3,3)), np.array(self.depth_info.D)
    
    def get_pointcloud(self):
        """Get the pointcloud from the camera convert the data to xyz and rgb format
        
        Returns:
            pc_xyz: pointcloud in xyz format
            pc_rgb: pointcloud in rgb format
        """
        pointcloud_data = rospy.wait_for_message("/camera/depth/color/points", PointCloud2, timeout=0.5)
        pc = list(point_cloud2.read_points(pointcloud_data, field_names=("x", "y", "z", "rgb")))
        pc_xyz = np.array(list(map(lambda x: x[0:3], pc)))
        pc_rgb = list(map(lambda x: x[3], pc))

        # Received data has rgba encripted in one 4 octect word, we format all data in "raw" format
        raw = struct.pack("f"* len(pc_rgb), *pc_rgb)
        #unpack the "raw" data to get the rgba values, a is the alpha channel (not relevant for us)
        rgba = np.array(struct.unpack("BBBB" *len(pc_rgb), raw))

        #reshape to only keep the rgb part
        pc_rgb = np.reshape(rgba, (-1, 4))[:,:3]

        return pc_xyz, pc_rgb

    def get_devices_info(self):
        return self.camera_info, self.depth_info


class RealCamera:
    def __init__(self, width=1280, height=720, framerate=30):
        """Connect to realsense camera
        
        Keyword Argument:
            width {int} -- width of the image (default: {1280})
            height {int} -- height of the image (default: {720})
            framerate {int} -- framerate of the camera (default: {30})
        """
        self._height = height 
        self._width = width
        self._pipeline = rs.pipeline()
        # Configure streams
        self._config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self._pipeline)
        pipeline_profile = self._config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        self._device_product_line = str(device.get_info(rs.camera_info.product_line))
        
        depth_sensor = pipeline_profile.get_device().first_depth_sensor()
        self._depth_scale = depth_sensor.get_depth_scale()

        self._config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        if self._device_product_line == 'L500':
            self._config.enable_stream(rs.stream.color, self._width, self._height, rs.format.bgr8, 30)
        else:
            self._config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        align_to = rs.stream.color
        self._align = rs.align(align_to)
        
    def start(self):
        self._pipeline.start(self._config)
        self._profile = self._pipeline.get_active_profile()
        #Set min acquisition distance to 0
        if self._device_product_line == 'L500':
            self._profile = self._pipeline.get_active_profile()
            sensor = self._profile.get_device().query_sensors()[0]        # Getting the depth sensor's depth scale (see rs-align example for explanation)
            sensor.set_option(rs.option.min_distance, 0)
            sensor.set_option(rs.option.enable_max_usable_range, 2)

    def stop(self):
        self._pipeline.stop()

    def get_rgb_only(self):
        # Wait for a coherent pair of frames: depth and color
        frames = self._pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            print("no frame grabbed")
            return None
        color_image = np.asanyarray(color_frame.get_data()).copy()
        return color_image

    def get_depth_only(self):
        frames = self._pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        depth_image = np.asanyarray(depth_frame.get_data())
        return depth_image
    
    def get_rgb_depth(self):
        # Wait for a coherent pair of frames: depth and color
        frames = self._pipeline.wait_for_frames()

        # Align the depth frame to color frame
        aligned_frames = self._align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            print("no frame grabbed")
            return None

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        depth_colormap_dim = depth_image.shape
        color_colormap_dim = color_image.shape

        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim[:2]:
            color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)

        return color_image, depth_image, self._depth_scale
    
    def depth2colormap(self, depth_image):
        return cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    
    def getIntrinsic(self):
        t = self._pipeline.get_active_profile().get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        intrinsic = np.array([[t.fx, 0., t.ppx], [0., t.fy, t.ppy], [ 0., 0., 1.]])
        distortion = np.array(t.coeffs)
        return intrinsic, distortion

    @staticmethod
    def getDevicesInfo():
        # For my Intel RealSense L515, product_line: L500, usb_type_descriptor: 3.2, serial_number: f0090395
        devicesInfo = []
        context = rs.context()
        for dev in context.query_devices():
            info = []
            info.append(dev.get_info(rs.camera_info.name))
            info.append(dev.get_info(rs.camera_info.product_line))
            info.append(dev.get_info(rs.camera_info.usb_type_descriptor))
            info.append(dev.get_info(rs.camera_info.serial_number))
            devicesInfo.append(info)
        return devicesInfo
