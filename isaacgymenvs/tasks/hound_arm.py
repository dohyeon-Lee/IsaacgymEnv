# Copyright (c) 2021-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgymenvs.utils.torch_jit_utils import get_axis_params, torch_rand_float, quat_rotate, quat_rotate_inverse
from isaacgymenvs.utils.torch_jit_utils import quat_mul, to_torch, tensor_clamp  
from isaacgymenvs.tasks.base.vec_task import VecTask


@torch.jit.script
def axisangle2quat(vec, eps=1e-6):
    """
    Converts scaled axis-angle to quat.
    Args:
        vec (tensor): (..., 3) tensor where final dim is (ax,ay,az) axis-angle exponential coordinates
        eps (float): Stability value below which small values will be mapped to 0

    Returns:
        tensor: (..., 4) tensor where final dim is (x,y,z,w) vec4 float quaternion
    """
    # type: (Tensor, float) -> Tensor
    # store input shape and reshape
    input_shape = vec.shape[:-1]
    vec = vec.reshape(-1, 3)

    # Grab angle
    angle = torch.norm(vec, dim=-1, keepdim=True)

    # Create return array
    quat = torch.zeros(torch.prod(torch.tensor(input_shape)), 4, device=vec.device)
    quat[:, 3] = 1.0

    # Grab indexes where angle is not zero an convert the input to its quaternion form
    idx = angle.reshape(-1) > eps
    quat[idx, :] = torch.cat([
        vec[idx, :] * torch.sin(angle[idx, :] / 2.0) / angle[idx, :],
        torch.cos(angle[idx, :] / 2.0)
    ], dim=-1)

    # Reshape and return output
    quat = quat.reshape(list(input_shape) + [4, ])
    return quat


class Houndarm(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.action_scale = self.cfg["env"]["actionScale"]
        # self.start_position_noise = self.cfg["env"]["startPositionNoise"]
        # self.start_rotation_noise = self.cfg["env"]["startRotationNoise"]
        self.houndarm_position_noise = self.cfg["env"]["houndarmPositionNoise"]
        self.houndarm_rotation_noise = self.cfg["env"]["houndarmRotationNoise"]
        self.houndarm_dof_noise = self.cfg["env"]["houndarmDofNoise"]
        # self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        # command ranges
        self.command_x_range = self.cfg["env"]["randomCommandPositionRanges"]["x"]
        self.command_y_range = self.cfg["env"]["randomCommandPositionRanges"]["y"]
        self.command_z_range = self.cfg["env"]["randomCommandPositionRanges"]["z"]
        # Create dicts to pass to reward function
        self.reward_settings = {
            "r_dist_scale": self.cfg["env"]["distRewardScale"],
            "r_lift_scale": self.cfg["env"]["liftRewardScale"],
            "r_align_scale": self.cfg["env"]["alignRewardScale"],
            "r_stack_scale": self.cfg["env"]["stackRewardScale"],
            "r_vel_scale": self.cfg["env"]["velRewardScale"],
        }

        # Controller type
        self.control_type = self.cfg["env"]["controlType"]
        assert self.control_type in {"osc", "joint_tor"},\
            "Invalid control type specified. Must be one of: {osc, joint_tor}"

        # dimensions
        # obs include: eef_pose (7) + q_gripper (2)
        self.cfg["env"]["numObservations"] = 10 if self.control_type == "osc" else 26
        # actions include: delta EEF if OSC (6) or joint torques (7) + bool gripper (1)
        self.cfg["env"]["numActions"] = 6 if self.control_type == "osc" else 8

        # Values to be filled in at runtime
        self.states = {}                        # will be dict filled with relevant states to use for reward calculation
        self.handles = {}                       # will be dict mapping names to relevant sim handles
        self.num_dofs = None                    # Total number of DOFs per env
        self.actions = None                     # Current actions to be deployed
        self._init_cubeA_state = None           # Initial state of cubeA for the current env
        # self._init_cubeB_state = None           # Initial state of cubeB for the current env
        # self._cubeA_state = None                # Current state of cubeA for the current env
        # self._cubeB_state = None                # Current state of cubeB for the current env
        # self._cubeA_id = None                   # Actor ID corresponding to cubeA for a given env
        # self._cubeB_id = None                   # Actor ID corresponding to cubeB for a given env

        # Tensor placeholders
        self._root_state = None             # State of root body        (n_envs, 13)
        self._dof_state = None  # State of all joints       (n_envs, n_dof)
        self._q = None  # Joint positions           (n_envs, n_dof)
        self._qd = None                     # Joint velocities          (n_envs, n_dof)
        self._rigid_body_state = None  # State of all rigid bodies             (n_envs, n_bodies, 13)
        self._contact_forces = None     # Contact forces in sim
        self._eef_state = None  # end effector state (at grasping point)
        self._eef_lf_state = None  # end effector state (at left fingertip)
        self._eef_rf_state = None  # end effector state (at left fingertip)
        self._j_eef = None  # Jacobian for end effector
        self._mm = None  # Mass matrix
        self._arm_control = None  # Tensor buffer for controlling arm
        self._gripper_control = None  # Tensor buffer for controlling gripper
        self._pos_control = None            # Position actions
        self._effort_control = None         # Torque actions
        self._houndarm_effort_limits = None        # Actuator effort limits for houndarm
        self._global_indices = None         # Unique indices corresponding to all envs in flattened array
        
        
        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.up_axis = "z"
        self.up_axis_idx = 2

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # houndarm defaults
        # self.houndarm_default_dof_pos = to_torch(
        #     [0, 0.1963, 0, -2.6180, 0, 2.9416], device=self.device
        # )
        self.houndarm_default_dof_pos = to_torch(
            [0, 0, 0, 0, 0, 0], device=self.device
        )
        # OSC Gains
        self.kp = to_torch([150.] * 6, device=self.device)
        self.kd = 2 * torch.sqrt(self.kp)
        self.kp_null = to_torch([10.] * 6, device=self.device)
        self.kd_null = 2 * torch.sqrt(self.kp_null)
        #self.cmd_limit = None                   # filled in later

        # Set control limits (osc: operation space control, cmd_limit : x y z r p y )
        self.cmd_limit = to_torch([0.1, 0.1, 0.1, 0.5, 0.5, 0.5], device=self.device).unsqueeze(0) if \
        self.control_type == "osc" else self._houndarm_effort_limits[:6].unsqueeze(0)
        # TODO add command
        self.commands = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands_x = self.commands.view(self.num_envs, 3)[..., 0]
        self.commands_y = self.commands.view(self.num_envs, 3)[..., 1]
        self.commands_z = self.commands.view(self.num_envs, 3)[..., 2]
        # Reset all environments
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

        # Refresh tensors
        self._refresh()

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        houndarm_asset_file = "urdf/open_manipulator_p_gazebo/urdf/open_manipulator_p.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
            houndarm_asset_file = self.cfg["env"]["asset"].get("assetFileNamehoundarm", houndarm_asset_file)

        # load houndarm asset

        asset_options = gymapi.AssetOptions()

        asset_options.replace_cylinder_with_capsule = False
        asset_options.flip_visual_attachments = False 

        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        asset_options.use_mesh_materials = True
        houndarm_asset = self.gym.load_asset(self.sim, asset_root, houndarm_asset_file, asset_options)

        houndarm_dof_stiffness = to_torch([0, 0, 0, 0, 0, 0], dtype=torch.float, device=self.device)
        houndarm_dof_damping = to_torch([0, 0, 0, 0, 0, 0], dtype=torch.float, device=self.device)

        self.num_houndarm_bodies = self.gym.get_asset_rigid_body_count(houndarm_asset)
        self.num_houndarm_dofs = self.gym.get_asset_dof_count(houndarm_asset)

        print("num houndarm bodies: ", self.num_houndarm_bodies)
        print("num houndarm dofs: ", self.num_houndarm_dofs)

        # set houndarm dof properties
        houndarm_dof_props = self.gym.get_asset_dof_properties(houndarm_asset)
        self.houndarm_dof_lower_limits = []
        self.houndarm_dof_upper_limits = []
        self._houndarm_effort_limits = []
        for i in range(self.num_houndarm_dofs):
            houndarm_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS if i > 6 else gymapi.DOF_MODE_EFFORT
            if self.physics_engine == gymapi.SIM_PHYSX:
                houndarm_dof_props['stiffness'][i] = houndarm_dof_stiffness[i]
                houndarm_dof_props['damping'][i] = houndarm_dof_damping[i]
            else:
                houndarm_dof_props['stiffness'][i] = 7000.0
                houndarm_dof_props['damping'][i] = 50.0

            self.houndarm_dof_lower_limits.append(houndarm_dof_props['lower'][i])
            self.houndarm_dof_upper_limits.append(houndarm_dof_props['upper'][i])
            self._houndarm_effort_limits.append(houndarm_dof_props['effort'][i])

        self.houndarm_dof_lower_limits = to_torch(self.houndarm_dof_lower_limits, device=self.device)
        self.houndarm_dof_upper_limits = to_torch(self.houndarm_dof_upper_limits, device=self.device)
        self._houndarm_effort_limits = to_torch(self._houndarm_effort_limits, device=self.device)
        self.houndarm_dof_speed_scales = torch.ones_like(self.houndarm_dof_lower_limits)
        # self.houndarm_dof_speed_scales[[7, 8]] = 0.1
        # houndarm_dof_props['effort'][7] = 200
        # houndarm_dof_props['effort'][8] = 200

        # Define start pose for houndarm # TODO link0's position
        houndarm_start_pose = gymapi.Transform()
        houndarm_start_pose.p = gymapi.Vec3(-0.45, 0.0, 0.0)
        houndarm_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)


        self.houndarms = []
        self.envs = []

        # Create environments
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            houndarm_actor = self.gym.create_actor(env_ptr, houndarm_asset, houndarm_start_pose, "houndarm", i, 0, 0)
            self.gym.set_actor_dof_properties(env_ptr, houndarm_actor, houndarm_dof_props)

            # Store the created env pointers
            self.envs.append(env_ptr)
            self.houndarms.append(houndarm_actor)

        # Setup data
        self.init_data()

    def init_data(self):
        # Setup sim handles
        env_ptr = self.envs[0]
        houndarm_handle = 0
        self.handles = {
            # houndarm
            "hand": self.gym.find_actor_rigid_body_handle(env_ptr, houndarm_handle, "panda_hand"),
            "leftfinger_tip": self.gym.find_actor_rigid_body_handle(env_ptr, houndarm_handle, "panda_leftfinger_tip"),
            "rightfinger_tip": self.gym.find_actor_rigid_body_handle(env_ptr, houndarm_handle, "panda_rightfinger_tip"),
            "grip_site": self.gym.find_actor_rigid_body_handle(env_ptr, houndarm_handle, "panda_grip_site"),
            "endpoint_tip": self.gym.find_actor_rigid_body_handle(env_ptr, houndarm_handle, "end_link"),
            # Cubes
            # "cubeA_body_handle": self.gym.find_actor_rigid_body_handle(self.envs[0], self._cubeA_id, "box"),
            # "cubeB_body_handle": self.gym.find_actor_rigid_body_handle(self.envs[0], self._cubeB_id, "box"),
        }

        # Get total DOFs
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        # Setup tensor buffers
        # print("oooooooooooooooooooooooooooooooooooo")

        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(self.num_envs, -1, 13)
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(self.num_envs, -1, 2)
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(self.num_envs, -1, 13)
        # print(self._rigid_body_state.size())
        self._q = self._dof_state[..., 0]
        self._qd = self._dof_state[..., 1]
        self._eef_state = self._rigid_body_state[:, self.handles["endpoint_tip"], :]
        # self._eef_lf_state = self._rigid_body_state[:, self.handles["leftfinger_tip"], :]
        # self._eef_rf_state = self._rigid_body_state[:, self.handles["rightfinger_tip"], :]
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "houndarm")
        jacobian = gymtorch.wrap_tensor(_jacobian)

        hand_joint_index = self.gym.get_actor_joint_dict(env_ptr, houndarm_handle)['joint6']
        print(jacobian.size())
        
        self._j_eef = jacobian[:, hand_joint_index, :, :6]
        # self._j_eef = jacobian[:, hand_joint_index, :, :7]
        _massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, "houndarm")
        mm = gymtorch.wrap_tensor(_massmatrix)
        self._mm = mm[:, :6, :6]
   
        # self._mm = mm[:, :7, :7]
        # self._cubeA_state = self._root_state[:, self._cubeA_id, :]
        # self._cubeB_state = self._root_state[:, self._cubeB_id, :]

        # Initialize states
        # self.states.update({
        #     "cubeA_size": torch.ones_like(self._eef_state[:, 0]) * self.cubeA_size,
        #     "cubeB_size": torch.ones_like(self._eef_state[:, 0]) * self.cubeB_size,
        # })

        # Initialize actions
        self._pos_control = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self._effort_control = torch.zeros_like(self._pos_control)

        # Initialize control
        self._arm_control = self._effort_control[:, :6]
        # self._arm_control = self._effort_control[:, :7]
        # self._gripper_control = self._pos_control[:, 7:9]

        # Initialize indices
        # self._global_indices = torch.arange(self.num_envs * 5, dtype=torch.int32, device=self.device).view(self.num_envs, -1)
        self._global_indices = torch.arange(self.num_envs, dtype=torch.int32, device=self.device).view(self.num_envs, -1)

    def _update_states(self):
        self.states.update({
            # houndarm
            "q": self._q[:, :],
            "q_gripper": self._q[:, -2:],
            "eef_pos": self._eef_state[:, :3],
            "eef_quat": self._eef_state[:, 3:7],
            "eef_vel": self._eef_state[:, 7:],
            # "eef_lf_pos": self._eef_lf_state[:, :3],
            # "eef_rf_pos": self._eef_rf_state[:, :3],
            "commands": self.commands[:,:],
            # Cubes
            # "cubeA_quat": self._cubeA_state[:, 3:7],
            # "cubeA_pos": self._cubeA_state[:, :3],
            # "cubeA_pos_relative": self._cubeA_state[:, :3] - self._eef_state[:, :3],
            # "cubeB_quat": self._cubeB_state[:, 3:7],
            # "cubeB_pos": self._cubeB_state[:, :3],
            # "cubeA_to_cubeB_pos": self._cubeB_state[:, :3] - self._cubeA_state[:, :3],
        })

    def _refresh(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

        # Refresh states
        self._update_states()

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:] = compute_houndarm_reward(
            self.reset_buf, self.progress_buf, self.actions, self.states, self.reward_settings, self.max_episode_length
        )

    def compute_observations(self):
        self._refresh()
        #obs = ["cubeA_quat", "cubeA_pos", "cubeA_to_cubeB_pos", "eef_pos", "eef_quat"]
        obs = ["eef_pos", "eef_quat", "commands"]
        #obs += ["q_gripper"] if self.control_type == "osc" else ["q"]
        self.obs_buf = torch.cat([self.states[ob] for ob in obs], dim=-1)

        maxs = {ob: torch.max(self.states[ob]).item() for ob in obs}

        return self.obs_buf

    def reset_idx(self, env_ids):
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        # Reset cubes, sampling cube B first, then A
        # if not self._i:
        # self._reset_init_cube_state(cube='B', env_ids=env_ids, check_valid=False)
        # self._reset_init_cube_state(cube='A', env_ids=env_ids, check_valid=True)
        # self._i = True

        # Write these new init states to the sim states
        # self._cubeA_state[env_ids] = self._init_cubeA_state[env_ids]
        # self._cubeB_state[env_ids] = self._init_cubeB_state[env_ids]
        
        # Reset command
        #self.states["commands"]
        self.commands_x[env_ids] = torch_rand_float(self.command_x_range[0], self.command_x_range[1], (len(env_ids), 1), device=self.device).squeeze()
        self.commands_y[env_ids] = torch_rand_float(self.command_y_range[0], self.command_y_range[1], (len(env_ids), 1), device=self.device).squeeze()
        self.commands_z[env_ids] = torch_rand_float(self.command_z_range[0], self.command_z_range[1], (len(env_ids), 1), device=self.device).squeeze()


        # Reset agent
        reset_noise = torch.rand((len(env_ids), 6), device=self.device)
        pos = tensor_clamp(
            self.houndarm_default_dof_pos.unsqueeze(0) +
            self.houndarm_dof_noise * 2.0 * (reset_noise - 0.5),
            self.houndarm_dof_lower_limits.unsqueeze(0), self.houndarm_dof_upper_limits) # TODO limit are in urdf

        # Overwrite gripper init pos (no noise since these are always position controlled)
        #pos[:, -2:] = self.houndarm_default_dof_pos[-2:]

        # Reset the internal obs accordingly
        self._q[env_ids, :] = pos
        self._qd[env_ids, :] = torch.zeros_like(self._qd[env_ids])

        # Set any position control to the current position, and any vel / effort control to be 0
        # NOTE: Task takes care of actually propagating these controls in sim using the SimActions API
        self._pos_control[env_ids, :] = pos
        self._effort_control[env_ids, :] = torch.zeros_like(pos)

        # Deploy updates
        #multi_env_ids_int32 = self._global_indices[env_ids, 0].flatten()
        multi_env_ids_int32 = self._global_indices[env_ids].flatten()
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self._pos_control),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))
        self.gym.set_dof_actuation_force_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self._effort_control),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32),
                                              len(multi_env_ids_int32))

        # Update cube states
        # multi_env_ids_cubes_int32 = self._global_indices[env_ids, -2:].flatten()
        # self.gym.set_actor_root_state_tensor_indexed(
        #     self.sim, gymtorch.unwrap_tensor(self._root_state),
        #     gymtorch.unwrap_tensor(multi_env_ids_cubes_int32), len(multi_env_ids_cubes_int32))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0


    def _compute_osc_torques(self, dpose):
        # Solve for Operational Space Control # Paper: khatib.stanford.edu/publications/pdfs/Khatib_1987_RA.pdf
        # Helpful resource: studywolf.wordpress.com/2013/09/17/robot-control-4-operation-space-control/
        q, qd = self._q[:, :6], self._qd[:, :6]
        #q, qd = self._q[:, :7], self._qd[:, :7]
        
        mm_inv = torch.inverse(self._mm)
        m_eef_inv = self._j_eef @ mm_inv @ torch.transpose(self._j_eef, 1, 2)
        m_eef = torch.inverse(m_eef_inv)

        # Transform our cartesian action `dpose` into joint torques `u`
        u = torch.transpose(self._j_eef, 1, 2) @ m_eef @ (
                self.kp * dpose - self.kd * self.states["eef_vel"]).unsqueeze(-1)

        # Nullspace control torques `u_null` prevents large changes in joint configuration
        # They are added into the nullspace of OSC so that the end effector orientation remains constant
        # roboticsproceedings.org/rss07/p31.pdf
        j_eef_inv = m_eef @ self._j_eef @ mm_inv
        u_null = self.kd_null * -qd + self.kp_null * ((self.houndarm_default_dof_pos[:6] - q + np.pi) % (2 * np.pi) - np.pi)#((self.houndarm_default_dof_pos[:7] - q + np.pi) % (2 * np.pi) - np.pi)
        u_null[:, 6:] *= 0
        # u_null[:, 7:] *= 0
        u_null = self._mm @ u_null.unsqueeze(-1)
        u += (torch.eye(6, device=self.device).unsqueeze(0) - torch.transpose(self._j_eef, 1, 2) @ j_eef_inv) @ u_null
        #u += (torch.eye(7, device=self.device).unsqueeze(0) - torch.transpose(self._j_eef, 1, 2) @ j_eef_inv) @ u_null

        # Clip the values to be within valid effort range
        u = tensor_clamp(u.squeeze(-1),
                         -self._houndarm_effort_limits[:6].unsqueeze(0), self._houndarm_effort_limits[:6].unsqueeze(0))
        # u = tensor_clamp(u.squeeze(-1),
        #                  -self._houndarm_effort_limits[:7].unsqueeze(0), self._houndarm_effort_limits[:7].unsqueeze(0))

        return u

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)

        # Split arm and gripper command
        u_arm = self.actions[:, :]
        #u_arm = self.actions[:, :]
        #print(u_arm)
        # print(u_arm, u_gripper)
        # print(self.cmd_limit, self.action_scale)

        # Control arm (scale value first)
        u_arm = u_arm * self.cmd_limit / self.action_scale
        if self.control_type == "osc":
            u_arm = self._compute_osc_torques(dpose=u_arm)
        self._arm_control[:, :] = u_arm

        # Control gripper
        # u_fingers = torch.zeros_like(self._gripper_control)
        # u_fingers[:, 0] = torch.where(u_gripper >= 0.0, self.houndarm_dof_upper_limits[-2].item(),
        #                               self.houndarm_dof_lower_limits[-2].item())
        # u_fingers[:, 1] = torch.where(u_gripper >= 0.0, self.houndarm_dof_upper_limits[-1].item(),
        #                               self.houndarm_dof_lower_limits[-1].item())
        # Write gripper command to appropriate tensor buffer
        # self._gripper_control[:, :] = u_fingers
        # TODO _effort_control change if _gripper_control,  _arm_control change
        # TODO _pos_control change if_gripper_control, change
        # Deploy actions
        #self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self._pos_control)) ## TODO TODO question
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self._effort_control))

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)

        # debug viz
        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            # Grab relevant states to visualize
            eef_pos = self.states["eef_pos"]
            eef_rot = self.states["eef_quat"]


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_houndarm_reward(
    reset_buf, progress_buf, actions, states, reward_settings, max_episode_length
):
    # type: (Tensor, Tensor, Tensor, Dict[str, Tensor], Dict[str, float], float) -> Tuple[Tensor, Tensor]

    distance_rewards = 1 - torch.tanh(10.0 * (torch.norm(states["eef_pos"] - states["commands"], dim=-1)))  #1 / (1 + torch.norm(states["eef_pos"] - states["commands"], dim=-1))
    velocity_rewards = 0
    
    
    distance_in_reach = (torch.norm(states["eef_pos"] - states["commands"], dim=-1) < 0.02)
    velocity_rewards = 1 - torch.tanh(10.0 * torch.norm(states["eef_vel"], dim=-1))
    rewards = distance_rewards * reward_settings["r_dist_scale"] + velocity_rewards * distance_in_reach * reward_settings["r_vel_scale"]
    rewards = torch.clip(rewards, 0., None)
    
    reset_buf = torch.where((progress_buf >= max_episode_length - 1), torch.ones_like(reset_buf), reset_buf)

    return rewards.detach(), reset_buf
