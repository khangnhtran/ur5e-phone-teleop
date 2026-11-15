"""Project-only top-down reset using native robot kinematics."""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from robosuite.wrappers.wrapper import Wrapper
from robosuite.wrappers import DataCollectionWrapper

# Columns are local TCP X/Y/Z expressed in world. Proper rotation, det=+1.
TOP_DOWN_R=np.array([[0.,1.,0.],[1.,0.,0.],[0.,0.,-1.]])
# +90deg world yaw from diag(1,-1,-1): finger opening visible in agentview; Z still down.

def base_env(env):
    while isinstance(env,Wrapper):
        env=env.env
    return env

def sync_controller(env):
    env=base_env(env)
    robot=env.robots[0]
    c=robot.controller
    c.update(force=True)
    c.update_initial_joints(np.array(env.sim.data.qpos[c.qpos_index]))
    c.reset_goal()

def apply_top_down(env):
    env=base_env(env)
    sim=env.sim
    robot=env.robots[0]
    c=robot.controller
    assert c.eef_name==sim.model.site_id2name(robot.eef_site_id)
    sid=robot.eef_site_id
    indices=np.array(c.qpos_index,dtype=int)
    seed=sim.data.qpos[indices].copy()
    position=sim.data.site_xpos[sid].copy()
    limits=np.array([sim.model.jnt_range[sim.model.joint_name2id(name)] for name in robot.robot_joints])
    def residual(q):
        sim.data.qpos[indices]=q
        sim.forward()
        rotation=sim.data.site_xmat[sid].reshape(3,3)
        return np.r_[5.*(position-sim.data.site_xpos[sid]),
                     Rotation.from_matrix(TOP_DOWN_R@rotation.T).as_rotvec()]
    result=least_squares(residual,seed,bounds=(limits[:,0]+1e-6,limits[:,1]-1e-6),
                         xtol=1e-12,ftol=1e-12,gtol=1e-12,max_nfev=300)
    error=residual(result.x)
    if np.linalg.norm(error[:3])/5>1e-5 or np.linalg.norm(error[3:])>1e-5:
        sim.data.qpos[indices]=seed
        sim.forward()
        raise RuntimeError("Top-down IK failed; robot reset was not accepted")
    sim.data.qvel[:]=0.
    sim.forward()
    sync_controller(env)
    return dict(tcp_site=c.eef_name,position=sim.data.site_xpos[sid].tolist(),
                R_world_tcp=sim.data.site_xmat[sid].reshape(3,3).tolist(),
                z_world=sim.data.site_xmat[sid].reshape(3,3)[:,2].tolist(),
                qpos=result.x.tolist())

class TopDownDataCollectionWrapper(DataCollectionWrapper):
    def _start_new_episode(self):
        # Apply IK BEFORE the official wrapper captures model/state.
        apply_top_down(self.env)
        super()._start_new_episode()
        # Official XML reload reinstantiates controller; restore its reference.
        sync_controller(self.env)

class TopDownResetWrapper(Wrapper):
    def reset(self):
        self.env.reset()
        apply_top_down(self.env)
        return self.env._get_observations(force_update=True)
