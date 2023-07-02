import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
import cv2

def poseCam2World(pos, orn, reference_pose):
    """ Convert pose from camera frame to world frame
    
    Args:
        pos (geometry_msgs.msg.Pose.position): position in camera frame
        orn (geometry_msgs.msg.Pose.orientation): orientation in camera frame
        reference_pose (np.array): 4x4 matrix of pose of camera in world frame"""
    grasps_pos_cam = np.array([pos.x, pos.y, pos.z])
    # grasps_orn_cam_xyzw_raw = np.array([orn.x, orn.y, orn.z, orn.w])
    grasps_orn_cam_xyzw_raw = Rotation.from_quat(np.array([orn.x, orn.y, orn.z, orn.w]))
    rot = np.array([0., 0., 90])
    grasps_orn_eul = grasps_orn_cam_xyzw_raw.as_euler("XYZ", degrees=True) + rot
    grasps_orn_cam_xyzw = Rotation.from_euler("XYZ", grasps_orn_eul, degrees=True).as_quat()
    grasps_mat_cam = quat2mat(grasps_orn_cam_xyzw)

    grasp_pos_world = reference_pose[:3,:3] @ grasps_pos_cam + reference_pose[:3,3]
    grasps_mat_world = reference_pose[:3,:3] @ grasps_mat_cam
    grasps_orn_world_xyzw = mat2quat(grasps_mat_world)

    return grasp_pos_world, grasps_orn_world_xyzw

def convert_quat(q, to="xyzw"):
    """
    Converts quaternion from one convention to another.
    The convention to convert TO is specified as an optional argument.
    If to == 'xyzw', then the input is in 'wxyz' format, and vice-versa.

    Args:
        q (np.array): a 4-dim array corresponding to a quaternion
        to (str): either 'xyzw' or 'wxyz', determining which convention to convert to.
    """
    if to == "xyzw":
        return q[[1, 2, 3, 0]]
    if to == "wxyz":
        return q[[3, 0, 1, 2]]
    raise Exception("convert_quat: choose a valid `to` argument (xyzw or wxyz)")

def quat_multiply(quaternion1, quaternion0):
    """
    Return multiplication of two quaternions (q1 * q0).

    E.g.:
    >>> q = quat_multiply([1, -2, 3, 4], [-5, 6, 7, 8])
    >>> np.allclose(q, [-44, -14, 48, 28])
    True

    Args:
        quaternion1 (np.array): (x,y,z,w) quaternion
        quaternion0 (np.array): (x,y,z,w) quaternion

    Returns:
        np.array: (x,y,z,w) multiplied quaternion
    """
    x0, y0, z0, w0 = quaternion0
    x1, y1, z1, w1 = quaternion1
    return np.array(
        (
            x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
            -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
            x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
            -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
        ),
        dtype=np.float32,
    )

def quat_conjugate(quaternion):
    """
    Return conjugate of quaternion.

    E.g.:
    >>> q0 = random_quaternion()
    >>> q1 = quat_conjugate(q0)
    >>> q1[3] == q0[3] and all(q1[:3] == -q0[:3])
    True

    Args:
        quaternion (np.array): (x,y,z,w) quaternion

    Returns:
        np.array: (x,y,z,w) quaternion conjugate
    """
    return np.array(
        (-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]),
        dtype=np.float32,
    )

def quat_inverse(quaternion):
    """
    Return inverse of quaternion.

    E.g.:
    >>> q0 = random_quaternion()
    >>> q1 = quat_inverse(q0)
    >>> np.allclose(quat_multiply(q0, q1), [0, 0, 0, 1])
    True

    Args:
        quaternion (np.array): (x,y,z,w) quaternion

    Returns:
        np.array: (x,y,z,w) quaternion inverse
    """
    return quat_conjugate(quaternion) / np.dot(quaternion, quaternion)


def quat_distance(quaternion1, quaternion0):
    """
    Returns distance between two quaternions, such that distance * quaternion0 = quaternion1

    Args:
        quaternion1 (np.array): (x,y,z,w) quaternion
        quaternion0 (np.array): (x,y,z,w) quaternion

    Returns:
        np.array: (x,y,z,w) quaternion distance
    """
    return quat_multiply(quaternion1, quat_inverse(quaternion0))

def quat2mat(quaternion):
    """
    Converts given quaternion to matrix.

    Args:
        quaternion (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: 3x3 rotation matrix
    """
    # awkward semantics for use with numba
    inds = np.array([3, 0, 1, 2])
    q = np.asarray(quaternion).copy().astype(np.float32)[inds]

    n = np.dot(q, q)
    if n < np.finfo(float).eps * 4.0:
        return np.identity(3)
    q *= np.sqrt(2.0 / n)
    q2 = np.outer(q, q)
    return np.array(
        [
            [1.0 - q2[2, 2] - q2[3, 3], q2[1, 2] - q2[3, 0], q2[1, 3] + q2[2, 0]],
            [q2[1, 2] + q2[3, 0], 1.0 - q2[1, 1] - q2[3, 3], q2[2, 3] - q2[1, 0]],
            [q2[1, 3] - q2[2, 0], q2[2, 3] + q2[1, 0], 1.0 - q2[1, 1] - q2[2, 2]],
        ]
    )

def pose_inv(pose):
    """
    Computes the inverse of a homogeneous matrix corresponding to the pose of some
    frame B in frame A. The inverse is the pose of frame A in frame B.

    Args:
        pose (np.array): 4x4 matrix for the pose to inverse

    Returns:
        np.array: 4x4 matrix for the inverse pose
    """

    # Note, the inverse of a pose matrix is the following
    # [R t; 0 1]^-1 = [R.T -R.T*t; 0 1]

    # Intuitively, this makes sense.
    # The original pose matrix translates by t, then rotates by R.
    # We just invert the rotation by applying R-1 = R.T, and also translate back.
    # Since we apply translation first before rotation, we need to translate by
    # -t in the original frame, which is -R-1*t in the new frame, and then rotate back by
    # R-1 to align the axis again.

    pose_inv = np.zeros((4, 4))
    pose_inv[:3, :3] = pose[:3, :3].T
    pose_inv[:3, 3] = -pose_inv[:3, :3].dot(pose[:3, 3])
    pose_inv[3, 3] = 1.0
    return pose_inv
   
def mat2quat(rmat):
    """
    Converts given rotation matrix to quaternion.

    Args:
        rmat (np.array): 3x3 rotation matrix

    Returns:
        np.array: (x,y,z,w) float quaternion angles
    """
    M = np.asarray(rmat).astype(np.float32)[:3, :3]

    m00 = M[0, 0]
    m01 = M[0, 1]
    m02 = M[0, 2]
    m10 = M[1, 0]
    m11 = M[1, 1]
    m12 = M[1, 2]
    m20 = M[2, 0]
    m21 = M[2, 1]
    m22 = M[2, 2]
    # symmetric matrix K
    K = np.array(
        [
            [m00 - m11 - m22, np.float32(0.0), np.float32(0.0), np.float32(0.0)],
            [m01 + m10, m11 - m00 - m22, np.float32(0.0), np.float32(0.0)],
            [m02 + m20, m12 + m21, m22 - m00 - m11, np.float32(0.0)],
            [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
        ]
    )
    K /= 3.0
    # quaternion is Eigen vector of K that corresponds to largest eigenvalue
    w, V = np.linalg.eigh(K)
    inds = np.array([3, 0, 1, 2])
    q1 = V[inds, np.argmax(w)]
    if q1[0] < 0.0:
        np.negative(q1, q1)
    inds = np.array([1, 2, 3, 0])
    return q1[inds]

def depth2pc(depth, K, rgb=None, segmap=None, max_distance=1.2):
    """
    Convert depth and intrinsics to point cloud and optionally point cloud color
    :param depth: hxw depth map in m
    :param K: 3x3 Camera Matrix with intrinsics
    :returns: (Nx3 point cloud, point cloud color)
    """

    if segmap is not None:
        mask = np.where((depth > 0) & (depth < max_distance) & (segmap > 0))
    else:
        mask = np.where((depth > 0) & (depth < max_distance))

    x,y = mask[1], mask[0]

    normalized_x = (x.astype(np.float32) - K[0,2])
    normalized_y = (y.astype(np.float32) - K[1,2])

    world_x = normalized_x * depth[y, x] / K[0,0]
    world_y = normalized_y * depth[y, x] / K[1,1]
    world_z = depth[y, x]

    if rgb is not None:
        rgb = rgb[y,x,:]

    pc = np.vstack((world_x, world_y, world_z)).T
    return (pc, rgb)

def regularize_pc(pc, pc_colors, downsampling_method="voxel", n_points=10000, voxel_size=0.005, 
                  outlier_filtering_method="radius", statistical_param_arg=[5, 1.0], radius_param_arg=[12, 0.015]):
    """Regularize point cloud by downsampling and filtering

    Args:
        pc (array): input point cloud
        pc_colors (array): input point cloud colors
        downsampling_method (str, optional): downsampling method in ["voxel", random] . Defaults to "voxel".
        n_points (int, optional): nb of point to downsample to. Defaults to 10000.
        voxel_size (float, optional): voxel size for downsampling. point are averaged in voxel area. Defaults to 0.005.
        filtering_method (str, optional): Filtering method in ["statistical", "radius"]  . Defaults to "radius".
        statistical_param_arg (, optional): [nb_neighbors, std_ratio]. Defaults to [5, 1.0].
        radius_param_arg (, optional): [nb_points, radius]. Defaults to [12, 0.015].
        For more info, refer to http://www.open3d.org/docs/latest/tutorial/Advanced/pointcloud_outlier_removal.html

    Returns:
        pc (array): regularized point cloud
        pc_colors (array): regularized point cloud colors
    """
    pcd = None
    import time
    start = time.time()
    if downsampling_method == "voxel":
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
        pcd.colors = o3d.utility.Vector3dVector(pc_colors)
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        pc = np.asarray(pcd.points)
        pc_colors = np.asarray(pcd.colors)
    elif downsampling_method == "random":
        if pc.shape[0] > n_points:
            idx = np.random.choice(pc.shape[0], n_points, replace=False)
            pc = pc[idx]
            pc_colors = pc_colors[idx]
        else:
            print("Warning: pc has less than {} points".format(n_points))
    else:
        print("Error: method {} not implemented, please chose between [voxel, random]".format(downsampling_method))
    downsampling_time = time.time()
    if outlier_filtering_method is not None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
        pcd.colors = o3d.utility.Vector3dVector(pc_colors)
        if outlier_filtering_method == "statistical":
            cl, ind = pcd.remove_statistical_outlier(nb_neighbors=statistical_param_arg[0], std_ratio=statistical_param_arg[1])
        elif outlier_filtering_method == "radius":
            cl, ind = pcd.remove_radius_outlier(nb_points=radius_param_arg[0], radius=radius_param_arg[1])
        else:
            print("Error: method {} not implemented, please chose between [statistical, radius]".format(outlier_filtering_method))
        pc = np.asarray(cl.points)
        pc_colors = np.asarray(cl.colors)
    filtering_time = time.time()
    print("Downsampling time: {}s, Filtering time: {}s".format(downsampling_time - start, filtering_time - downsampling_time))
    
    return pc, pc_colors

def add_view2pc(pc, pc_colors, main_view_pose, camera_pose, new_pc=None, new_pc_colors=None,
                new_gbr=None, new_depth=None, cam_intrisic=None, regularize=True, voxel_size=0.005):
    """ Add new view to point cloud and fuse it with the previous one,
        Need either the new point cloud or the new rgb, depth and camera intrinsic to function
     
    Args:
        pc (np.array): point cloud of shape (n_points, 3)
        pc_colors (np.array): point cloud colors of shape (n_points, 3)
        main_view_pose (np.array): pose of the pc origin
        camera_pose (np.array): pose of the new view
        new_pc (np.array): point cloud to fused with pc of shape (n_points, 3)
        new_pc_colors (np.array): point cloud colors to fused with pc of shape (n_points, 3)
        new_gbr (np.array): new rgb image of shape (H, W, 3)
        new_depth (np.array): new depth image of shape (H, W)
        cam_intrisic (np.array): camera intrinsic matrix of shape (3, 3)
        regularize (bool): whether to regularize the point cloud
        voxel_size (float): voxel size for regularization

    Returns:
        fused_pc (np.array): fused point cloud of shape (n_points, 3)
        fused_pc_colors (np.array): fused point cloud colors of shape (n_points, 3)"""
    if new_pc is None:
        if new_depth is None or new_gbr is None or cam_intrisic is None:
            print("Error: (new_pc and new_pc_colors) or (new_depth, new_gbr, cam_intrisic) must be provided")
            return pc

        new_pc, new_pc_colors = depth2pc(new_depth, cam_intrisic, new_gbr)

    new2main = pose_inv(main_view_pose) @ camera_pose
    new_pc_in_mainView = (new2main[:3, :3] @ new_pc.T).T + new2main[:3,3]

    if pc is not None:
        fused_pc = np.vstack((pc, new_pc_in_mainView))
        fused_pc_colors = np.vstack((pc_colors, new_pc_colors))
    else:
        fused_pc = new_pc_in_mainView
        fused_pc_colors = new_pc_colors

    if regularize:
        fused_pc, fused_pc_colors = regularize_pc(fused_pc, fused_pc_colors, downsampling_method="voxel", voxel_size=voxel_size,
                                                  outlier_filtering_method="radius", radius_param_arg=[25, 3*voxel_size])
        

    return fused_pc, fused_pc_colors

def correct_angle(quat_xyzw, angle_range):
    """ if angle outside of range, rotate by 180 degrees

    Args:
        quat_xyzw (array): quaternion of the gripper
        angle_range (array): range of the angle

    Returns:
        _type_: _description_
    """
    angle_range = [-140, 40]
    angle_euler = Rotation.from_quat(quat_xyzw).as_euler("XYZ", degrees=True)
    if angle_euler[2] < min(angle_range) or angle_euler[2] > max(angle_range):
        print("Readjuested angle")
        rot_orn=np.array([0, 0, -1, 0])
        quat_xyzw = quat_multiply(quat_xyzw, rot_orn)

    return quat_xyzw

def get_ROI_box(pc, pc_colors, depth, rgb, segmap, intrinsic, border_size = 0.05):

    pc_seg, pc_colors_seg = depth2pc(depth, intrinsic, rgb, segmap=segmap)

    min_bound = np.min(pc_seg, axis=0) - border_size
    max_bound = np.max(pc_seg, axis=0) + border_size

    bounding_point = np.array([[min_bound[0], min_bound[1], min_bound[2], 1],
                            [min_bound[0], min_bound[1], max_bound[2], 1],
                            [min_bound[0], max_bound[1], min_bound[2], 1],
                            [min_bound[0], max_bound[1], max_bound[2], 1],
                            [max_bound[0], min_bound[1], min_bound[2], 1],
                            [max_bound[0], min_bound[1], max_bound[2], 1],
                            [max_bound[0], max_bound[1], min_bound[2], 1],
                            [max_bound[0], max_bound[1], max_bound[2], 1]])
    #crop point cloud
    ROI_mask = np.array((pc[:,0] > min_bound[0]) & (pc[:,0] < max_bound[0]) &
                    (pc[:,1] > min_bound[1]) & (pc[:,1] < max_bound[1]) &
                    (pc[:,2] > min_bound[2]) & (pc[:,2] < max_bound[2]))

    pc_ROI = pc[ROI_mask]
    pc_colors_ROI = pc_colors[ROI_mask]

    return pc_ROI, pc_colors_ROI, bounding_point

def project_pc(pc, pc_colors, intrinsic, extrinsic, image_size=(720, 1280)):
    pc = np.concatenate((pc, np.ones((pc.shape[0], 1))), axis=1)

    T = np.eye(4)
    #invert transformation
    T[:3, 3] = -extrinsic[:3, :3].T @ extrinsic[:3, 3]
    T[:3, :3] = extrinsic[:3, :3].T
    #apply transformation
                    
    pc = T @ pc.T
    uv = intrinsic @ pc[:3, :]
    uv = uv / uv[2, :]
    uv = uv[:2, :].T
    uv = np.round(uv).astype(np.int32)

    #select point in image
    mask = (uv[:, 0] >= 0) & (uv[:, 0] < image_size[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < image_size[0])
    image = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)

    #draw points on image
    image[uv[mask, 1], uv[mask, 0]] = pc_colors[mask][:, [0, 1, 2]]

    #draw circle on image
    for i in range(uv.shape[0]):
        if mask[i]:
            cv2.circle(image, (uv[i, 0], uv[i, 1]), round(max(image_size)/200), pc_colors[i], -1)

    return image

def apply_ROI_box_mask(image, bounding_point, intrinsic, extrinsic, image_size=(720, 1280)):
    T = np.eye(4)
    #invert transformation
    T[:3, 3] = -extrinsic[:3, :3].T @ extrinsic[:3, 3]
    T[:3, :3] = extrinsic[:3, :3].T

    #apply transformation
    bounding_point_uv = T @ bounding_point.T
    bounding_point_uv = intrinsic @ bounding_point_uv[:3, :]
    bounding_point_uv = bounding_point_uv / bounding_point_uv[2, :]
    bounding_point_uv = bounding_point_uv[:2, :].T
    bounding_point_uv = bounding_point_uv.astype(np.int32)
    mask_bounding = (bounding_point_uv[:, 0] >= 0) & (bounding_point_uv[:, 0] < image_size[1]) & (bounding_point_uv[:, 1] >= 0) & (bounding_point_uv[:, 1] < image_size[0])
    max_bound = np.max(bounding_point_uv, axis=0)
    min_bound = np.min(bounding_point_uv, axis=0)
    print(min_bound, max_bound)
    if min_bound[0] < 0:
        min_bound[0] = 0
    if min_bound[1] < 0:
        min_bound[1] = 0
    if max_bound[0] < 0:
        max_bound[0] = 0
    if max_bound[1] < 0:
        max_bound[1] = 0
    print(min_bound, max_bound)
    mask_bounding_ROI = np.ones_like(image)
    mask_bounding_ROI[min_bound[1]:max_bound[1], min_bound[0]:max_bound[0]] = 0

    image[mask_bounding_ROI.astype(np.bool)] = 255

    return image