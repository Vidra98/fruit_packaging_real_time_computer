from utils.segmentation import YOLOSegmentation
import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from contact_grasp.srv import segmentationSrv, segmentationSrvResponse
if __name__ == '__main__':
    print("Start")
    rospy.init_node("python_node",anonymous=True)

    #load cv image
    img = cv2.imread("/home/vdrame/bgr.png")

    #convert to ros image
    bridge = CvBridge()

    #convert to ros image
    ros_img = bridge.cv2_to_imgmsg(img, encoding="bgr8")

    #call service
    rospy.wait_for_service("segmentation")

    try:
        segmentation = rospy.ServiceProxy("segmentation", segmentationSrv)
        resp = segmentation(ros_img)
    except rospy.ServiceException as e:
        print(e)

    #convert to cv image
    cv_img = bridge.imgmsg_to_cv2(resp.image, desired_encoding="passthrough")

    #show image
    cv2.imshow("segmented image", cv_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
