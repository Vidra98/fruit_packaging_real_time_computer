from PyLQR.sim import KDLRobot
from PyLQR.system import PosOrnPlannerSys, PosOrnKeypoint
from PyLQR.solver import BatchILQRCP, BatchILQR, ILQRRecursive
from PyLQR.utils import primitives, PythonCallbackMessage
import numpy as np
from utils.transform_utils import *
import matplotlib.pyplot as plt


class iLQR():
    def __init__(self, rbt):
        #Load URDf for KDL and iLQR solver
        self._control_freq = 20
        
        self._dof = len(rbt.q)
        self._nb_state_var = self._dof
        self._nb_ctrl_var = self._dof
        self._dt = 1/(self._control_freq)

        #PATH_TO_URDF = "/home/vdrame/project/client/panda_description/urdf/panda_arm_robosuite.urdf"
        PATH_TO_URDF = "/home/vdrame/Documents/ros_ws/src/rli_ws_base/py_panda/Tutorials/model.urdf"
        
        BASE_FRAME = "panda_link0"
        TIP_FRAME = "panda_tip"
        #TIP_FRAME = "panda_grasptarget"

        # initial joint config
        q0 = rbt.q
        dq0 = [0]*rbt.dq

        # ##This is min and max value
        # self._qMax = np.array([2.87,   1.75,  2.8973, -0.05,  2.8973,  3.75,   2.8973])
        # self._qMin = np.array([-2.87, -1.75, -2.8973, -3.05, -2.8973, -0.015, -2.8973])

        self._qMax = np.array([1.68,   1.6,  2.65, 0.05,  2.65,  3.5,   2.65])
        self._qMin = np.array([-0.4, -1.6, -2.65, -2.95, -2.65, 0.05, -2.65])
        # self._dqMax = np.array([2., 2., 2., 2., 2.6, 2.6, 2.6])
        self._dqMax = np.array([.5, .5, .5, .5, .6, .6, .6])

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

        if verbose or np.any(np.abs(orn_dif)>0.1) or np.any(np.abs(pos_dif))>0.01:
            print("difference between iLQR model ee and real robot ee")
            print(f"quat (zyzw) :{orn_dif}")
            print(f"pos:{pos_dif}")
        
    def generate_trajectory(self, q0, dq0, keypoints, horizon, cmd_penalties = 1e-5, final_pos = None, final_orn = None):
        #Set robot starting position for retrieving trajectory
        self._rbt_KDL.set_conf(q0, dq0, False)

        # Each control signals have a penalty of 1e-5
        cmd_penalties_list = (np.ones(self._nb_ctrl_var)*cmd_penalties).tolist()

        # It is not mandatory to set the limits, if you do not know them yet or do not want to use them. You can use this constructor:
        #sys = PosOrnPlannerSys(self._rbt_KDL, keypoints, cmd_penalties, horizon, 1, dt)
        sys = PosOrnPlannerSys(self._rbt_KDL, keypoints, cmd_penalties_list, self._qMax, self._qMin, horizon, 1, self._dt)

        u0_t = np.array([0]*(self._nb_ctrl_var-1) + [0])
        u0 = np.tile(u0_t, horizon-1)

        mu = sys.get_mu_vector(False)
        Q = sys.get_Q_matrix(False)

        planner = ILQRRecursive(sys)

        # callback to notify python code of the solver evolution
        # cb = PythonCallbackMessage()
        cb = None
        # solver input :std::vector<VectorXd>& U0, int nb_iter, bool line_search, bool early_stop, CallBackMessage* cb
        jpos, x_pos, U, Ks, ds = planner.solve(
            u0.reshape((-1, self._nb_ctrl_var)), 50, True, True, cb)

        jpos, x_pos = np.asarray(jpos), np.asarray(x_pos)

        if final_orn is not None and final_pos is not None:
            pos_dif = final_pos - x_pos[-1, :3]
            pos_dif = np.linalg.norm(pos_dif)
            orn_dif = quat_distance(convert_quat(final_orn, to="xyzw"), x_pos[-1, 3:])
            if pos_dif > 0.01:
                print("WARNING: final position not reached")
                print(f"pos_dif:{pos_dif}")

        return jpos, x_pos, U, Ks, ds, pos_dif, orn_dif

    def grasping_trajectory(self, q0, dq0, grasps_pos_base, grasps_orn_wxyz, horizon = 120,
                           pos_threshold = 0.01, orn_threshold = 0.1, max_iter = 3):
        Qtarget1 = np.diag([.1,  # Tracking the x position
                            .1,  # Tracking the y position
                            .1,  # Tracking the z position
                            1,  # Tracking orientation around x axis
                            1,  # Tracking orientation around y axis
                            1])  # Tracking orientation around z axis

        rotation_mat_grasps = quat2mat(convert_quat(grasps_orn_wxyz, to="xyzw"))
        target1_pos_base = grasps_pos_base.copy()
        target1_pos_base -= rotation_mat_grasps[:3,2] * 0.1 #rbt.get_ee_pos()[2].copy()
        #target1_pos_base[2] += 0.1
        target1_orn = grasps_orn_wxyz.copy()

        target2_pos_base = grasps_pos_base.copy()
        target2_orn = grasps_orn_wxyz.copy()

        target1_discrete_time = horizon-30 - 1
        keypoint_1 = PosOrnKeypoint(target1_pos_base, target1_orn, Qtarget1, target1_discrete_time)

        Qtarget2 = np.diag([.51, .51, .51, 1, 1, 1])
        target2_discrete_time = horizon - 1
        keypoint_2 = PosOrnKeypoint(target2_pos_base, target2_orn, Qtarget2, target2_discrete_time)

        keypoints = [keypoint_1, keypoint_2]
        #keypoints = [keypoint_1]

        pos_dif = 1000
        iter = 0
        while pos_dif > pos_threshold and iter < max_iter:
            jpos, x_pos, U, Ks, ds, pos_dif, orn_dif = self.generate_trajectory(q0, dq0, keypoints, horizon)
            iter += 1
            horizon = int(1.5*horizon)

        return jpos, x_pos, U, Ks, ds, pos_dif, orn_dif
    
    def dispose_trajectory(self, q0, dq0, grasp_pos, grasp_orn, dispose_pos, dispose_orn, horizon = 120,
                           pos_threshold = 0.01, orn_threshold = 0.1, max_iter = 3):
        """ 
        dispose quat (wxyz)
        """
        Qtarget1 = np.diag([.2, .2, 1, .7, .7, .7])
        Qtarget2 = np.diag([.5, .5, 1, .1, .1, .1])
        Qtarget3 = np.diag([.7, .7, .7, 1, 1, 1])

        target1_orn = grasp_orn.copy()
        rotation1_mat_grasps = quat2mat(convert_quat(target1_orn, to="xyzw"))
        target1_pos_base = grasp_pos.copy()
        target1_pos_base -= rotation1_mat_grasps[:3,2] * 0.15 #rbt.get_ee_pos()[2].copy()
        target1_discrete_time = 20 - 1
        keypoint_1 = PosOrnKeypoint(target1_pos_base, target1_orn, Qtarget1, target1_discrete_time)

        target2_orn =  dispose_orn
        rotation2_mat_grasps = quat2mat(convert_quat(target2_orn, to="xyzw"))
        target2_pos_base = dispose_pos.copy()
        target2_pos_base -= rotation2_mat_grasps[:3,2] * 0.15 #rbt.get_ee_pos()[2].copy()
        target2_discrete_time = horizon - 20 - 1
        keypoint_2 = PosOrnKeypoint(target2_pos_base, target2_orn, Qtarget2, target2_discrete_time)

        target3_orn =  dispose_orn
        target3_pos_base = dispose_pos
        target3_discrete_time = horizon - 1
        keypoint_3 = PosOrnKeypoint(target3_pos_base, target3_orn, Qtarget3, target3_discrete_time)        

        keypoints = [keypoint_1, keypoint_2, keypoint_3]

        pos_dif = 1000
        iter = 0
        while pos_dif > pos_threshold and iter < max_iter:
            jpos, x_pos, U, Ks, ds, pos_dif, orn_dif = self.generate_trajectory(q0, dq0, keypoints, horizon, cmd_penalties = 1e-3)
            iter += 1
            horizon = int(1.5*horizon)

        return jpos, x_pos, U, Ks, ds, pos_dif, orn_dif
    
    def direct_trajectory(self, q0, dq0, target_pos, target_orn_wxyz, horizon = 45,
                           pos_threshold = 0.03, orn_threshold = 0.1, max_iter = 2):
        Qtarget1 = np.diag([.5, .5, .5, 1, 1, 1])

        target1_pos_base = target_pos.copy()
        target1_orn = target_orn_wxyz.copy()

        target1_discrete_time = horizon - 1
        keypoint = PosOrnKeypoint(target1_pos_base, target1_orn, Qtarget1, target1_discrete_time)

        keypoints = [keypoint]

        pos_dif = 1000
        iter = 0
        while pos_dif > pos_threshold and iter < max_iter:
            jpos, x_pos, U, Ks, ds, pos_dif, orn_dif = self.generate_trajectory(q0, dq0, keypoints, horizon)
            iter += 1
            horizon = int(1.5*horizon)

        return jpos, x_pos, U, Ks, ds, pos_dif, orn_dif

    def plot_trajectory(current_pos, target_pos, trajectory, trajectory2 = None):
        # Plot the trajectory
        #fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_title("Planned trajectory")

        ax.scatter3D(current_pos[0], current_pos[1], current_pos[2], c='r', marker='o')
        ax.scatter3D(target_pos[0], target_pos[1], target_pos[2], c='g', marker='o')
        ax.plot3D(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], c='green')

        if trajectory2 is not None:
            ax.plot3D(trajectory2[:, 0], trajectory2[:, 1], trajectory2[:, 2], c='orange', linestyle="dashed")

        # Plot the trajectory
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        #ax.view_init(elev=30, azim=45)
        plt.legend()
        plt.show()