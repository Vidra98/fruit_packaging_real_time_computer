import numpy as np
import cv2
from scipy.spatial.transform import Rotation
import time
import open3d as o3d
from .transform_utils import depth2pc





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

def get_ROI_box(pc, pc_colors, depth, rgb, segmap, intrinsic, extrinsic, border_size = 0.05):

    if np.all(segmap==0):
        return [], [], None
    pc_seg, pc_colors_seg = depth2pc(depth, intrinsic, rgb, segmap=segmap)
    pc_seg = extrinsic @ np.concatenate((pc_seg, np.ones((pc_seg.shape[0], 1))), axis=1).T
    pc_seg = pc_seg.T[:, :3]    
    
    min_bound = np.min(pc_seg, axis=0) - [border_size, border_size, 0]
    max_bound = np.max(pc_seg, axis=0) + [border_size, border_size, 0]

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
    if min_bound[0] < 0:
        min_bound[0] = 0
    if min_bound[1] < 0:
        min_bound[1] = 0
    if max_bound[0] < 0:
        max_bound[0] = 0
    if max_bound[1] < 0:
        max_bound[1] = 0
    mask_bounding_ROI = np.ones_like(image)
    mask_bounding_ROI[min_bound[1]:max_bound[1], min_bound[0]:max_bound[0]] = 0

    image[mask_bounding_ROI.astype(np.bool)] = 255

    return image

def compute_views_scores(views, pc_fused, pc_colors_fused, intrinsic, bounding_point, image_size=(720, 1280), verbose=False, show=False):
    # compte views score
    next_view = None
    viewBestScore = 0
    times = []
    for view_key in views:
        viewTransform = np.eye(4)
        viewTransform[0:3, 3] = views[view_key]["pos"]
        viewTransform[0:3, 0:3] = Rotation.from_quat(np.array(views[view_key]["orn_wxyz"])[[1, 2, 3, 0]]).as_matrix()

        start = time.time()
        transform = viewTransform
        image = project_pc(pc_fused, pc_colors_fused, intrinsic, transform, image_size=image_size)
        image = apply_ROI_box_mask(image, bounding_point, intrinsic, transform, image_size=image_size)
        image_1C = np.sum(image, axis=2)
        score = np.sum(image_1C == 0)/(image_1C.shape[0] * image_1C.shape[1])
        if score > viewBestScore:
            viewBestScore = score
            next_view = view_key
        times.append(time.time() - start)
        if show:
            cv2.imshow("{}".format(view_key), image)
    
    if show:
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if verbose:
        print("average inference time ", np.mean(times))
        print("best_score is ", viewBestScore)
        print("next view is ", next_view)
    return next_view, viewBestScore