import numpy as np
np.random.seed(0)
import robosuite as suite
from robosuite import load_controller_config
from PyLQR.sim import KDLRobot
from PyLQR.system import PosOrnPlannerSys, PosOrnKeypoint
from PyLQR.solver import BatchILQRCP, BatchILQR, ILQRRecursive
from PyLQR.utils import primitives, PythonCallbackMessage

render = True
controller = 'JOINT_VELOCITY' #TO DO: JOINT_VELOCITY
controller_configs = load_controller_config(default_controller=controller)
# create environment instance
env = suite.make(
    env_name="Wipe",
    robots="Panda",
    controller_configs=controller_configs,
    has_renderer=render,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    horizon=10000000,
    initialization_noise=None,
    control_freq =10,
)
# reset the environment
env.reset()

for i in range(1000):
    action = np.random.randn(env.robots[0].dof) # sample random action
    obs, reward, done, info = env.step(action)  # take action in the environment
    env.render()  # render on display
