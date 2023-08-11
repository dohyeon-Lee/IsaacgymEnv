# Copyright (c) 2018-2023, NVIDIA Corporation
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
import os, time

from isaacgym import gymtorch
from isaacgym import gymapi
from .base.vec_task import VecTask

import torch
from typing import Tuple, Dict

from isaacgymenvs.utils.torch_jit_utils import to_torch, quat_mul,quat_rotate, get_axis_params, torch_rand_float, normalize, quat_apply, quat_rotate_inverse, tensor_clamp, quat_from_angle_axis, quat_conjugate
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
def spherical_to_cartesian(r, theta, phi):
    x = r * torch.sin(theta) * torch.cos(phi)
    y = r * torch.sin(theta) * torch.sin(phi)
    z = r * torch.cos(theta)
    return x, y, z
class UsefulHoundPrior(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):

        self.cfg = cfg
        self.height_samples = None
        self.custom_origins = False
        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.init_done = False

        self.TrainingMode = self.cfg["env"]["mode"]
        self.maniControlMode = self.cfg["env"]["ManipulationControlType"] # osc | joint
        self.pushrobot = self.cfg["env"]["learn"]["pushRobots"]

        # normalization
        self.lin_vel_scale = self.cfg["env"]["learn"]["linearVelocityScale"]
        self.ang_vel_scale = self.cfg["env"]["learn"]["angularVelocityScale"]
        self.dof_pos_scale = self.cfg["env"]["learn"]["dofPositionScale"]
        self.dof_vel_scale = self.cfg["env"]["learn"]["dofVelocityScale"]
        self.loco_action_scale = self.cfg["env"]["control"]["locoActionScale"]
        self.mani_action_scale = self.cfg["env"]["control"]["maniActionScale"]

        # command ranges
        self.arm_command_x_range = self.cfg["env"]["randomArmCommandPositionRanges"]["x"]
        self.arm_command_y_range = self.cfg["env"]["randomArmCommandPositionRanges"]["y"]
        self.arm_command_z_range = self.cfg["env"]["randomArmCommandPositionRanges"]["z"]
        self.command_x_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_x"]
        self.command_y_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_y"]
        self.command_yaw_range = self.cfg["env"]["randomCommandVelocityRanges"]["yaw"]
        self.sph_command_x_range = self.cfg["env"]["randomArmCommandPositionRanges"]["sph_l"]
        self.sph_command_y_range = self.cfg["env"]["randomArmCommandPositionRanges"]["sph_p"]
        self.sph_command_z_range = self.cfg["env"]["randomArmCommandPositionRanges"]["sph_y"]
        self.sph_command_T_range = self.cfg["env"]["randomArmCommandPositionRanges"]["sph_T"]

        # reward scales
        self.rew_scales = {}
        if(self.TrainingMode == "Locomotion_vel"):
            self.rew_scales["lin_vel_xy"] = self.cfg["env"]["learn"]["linearVelocityXYRewardScale"] 
            self.rew_scales["ang_vel_z"] = self.cfg["env"]["learn"]["angularVelocityZRewardScale"] 
            self.rew_scales["air_time"] = self.cfg["env"]["learn"]["feetAirTimeRewardScale"]
            self.rew_scales["lin_vel_z"] = self.cfg["env"]["learn"]["linearVelocityZRewardScale"] 
            self.rew_scales["ang_vel_xy"] = self.cfg["env"]["learn"]["angularVelocityXYRewardScale"] 

            self.rew_scales["eef_dist_scale"] = 0
            self.rew_scales["eef_vel_scale"] = 0
            self.rew_scales["rew_imitate"] = 0
            self.rew_scales["ang_vel_rpy"] = 0
            self.rew_scales["lin_vel_xyz"] = 0
            self.rew_scales["basepos"] = 0
        
        else :
            self.rew_scales["lin_vel_xy"] = 0
            self.rew_scales["ang_vel_z"] = 0
            self.rew_scales["air_time"] = 0
            self.rew_scales["lin_vel_z"] = 0
            self.rew_scales["ang_vel_xy"] = 0

            self.rew_scales["eef_dist_scale"] = self.cfg["env"]["learn"]["eefdistRewardScale"]
            self.rew_scales["eef_vel_scale"] = self.cfg["env"]["learn"]["eefvelRewardScale"]
            self.rew_scales["rew_imitate"] = self.cfg["env"]["learn"]["imitateRewardScale"]
            self.rew_scales["ang_vel_rpy"] = self.cfg["env"]["learn"]["angvelrpyRewardScale"]
            self.rew_scales["lin_vel_xyz"] = self.cfg["env"]["learn"]["linvelxyzRewardScale"]
            self.rew_scales["basepos"] = self.cfg["env"]["learn"]["baseposRewardScale"]

        self.rew_scales["orient"] = self.cfg["env"]["learn"]["orientationRewardScale"] 
        self.rew_scales["torque"] = self.cfg["env"]["learn"]["torqueRewardScale"]
        self.rew_scales["loco_joint_acc"] = self.cfg["env"]["learn"]["locojointAccRewardScale"]
        self.rew_scales["base_height"] = self.cfg["env"]["learn"]["baseHeightRewardScale"]
        self.rew_scales["collision"] = self.cfg["env"]["learn"]["kneeCollisionRewardScale"]
        self.rew_scales["stumble"] = self.cfg["env"]["learn"]["feetStumbleRewardScale"]
        self.rew_scales["action_rate"] = self.cfg["env"]["learn"]["actionRateRewardScale"]
        self.rew_scales["mani_joint_acc"] = self.cfg["env"]["learn"]["manijointAccRewardScale"]
        self.rew_scales["mani_dof_vel"] = self.cfg["env"]["learn"]["manijointVelRewardScale"]
        self.rew_scales["loco_dof_vel"] = self.cfg["env"]["learn"]["locojointVelRewardScale"]
        self.rew_scales["termination"] = self.cfg["env"]["learn"]["terminalReward"]
        self.rew_scales["hip"] = self.cfg["env"]["learn"]["hipRewardScale"]

        # base init state
        pos = self.cfg["env"]["baseInitState"]["pos"]
        rot = self.cfg["env"]["baseInitState"]["rot"]
        v_lin = self.cfg["env"]["baseInitState"]["vLinear"]
        v_ang = self.cfg["env"]["baseInitState"]["vAngular"]
        self.base_init_state = pos + rot + v_lin + v_ang

        
        # other
        self.decimation = self.cfg["env"]["control"]["decimation"]
        self.dt = self.decimation * self.cfg["sim"]["dt"]
        self.max_episode_length_s = self.cfg["env"]["learn"]["episodeLength_s"] 
        self.max_episode_length = int(self.max_episode_length_s/ self.dt + 0.5)
        self.push_interval = int(self.cfg["env"]["learn"]["pushInterval_s"] / self.dt + 0.5)
        self.allow_knee_contacts = self.cfg["env"]["learn"]["allowKneeContacts"]
        self.Kp = self.cfg["env"]["control"]["stiffness"]
        self.Kd = self.cfg["env"]["control"]["damping"]

        for key in self.rew_scales.keys():
            self.rew_scales[key] *= self.dt

        #define in _create_envs
        self.total_num_dof = None
        self.hound_num_dof = None
        self.arm_num_dof = None
        self.num_bodies = None
        self.body_names = None
        self.dof_names = None
        self.feet_indices = None
        self.knee_indices = None
        self.shoulder_indices = None
        self.eef_index = None
        self.base_index = None
        self.real_eef_index = None
        self._rigid_body_state = None               # State of all rigid bodies             (n_envs, n_bodies, 13)
        self.base_states = None                     # State of root body                    (n_envs, 13)
        self.eef_state = None                       # end effector state                    (at grasping point)
        self.dof_state = None                       # State of all joints                   (n_envs, n_dof)
        self.dof_pos = None
        self.dof_vel = None
        self.mani_dof_pos = None                    # Joint positions                       (n_envs, n_dof)
        self.mani_dof_vel = None                    # Joint velocities                      (n_envs, n_dof)
        self.loco_dof_pos = None
        self.loco_dof_vel = None
        
        # on base frame
        self.base_lin_vel = None
        self.base_ang_vel = None
        self.eef_lin_vel = None
        self.eef_ang_vel = None
        self.relative_eef_lin_vel = None
        self.relative_eef_ang_vel = None
        self.relative_eef_quat = None
        self.relative_eef_pos = None
        self.mani_effort_limits = None 
        self.mani_dof_lower_limits = None
        self.mani_dof_upper_limits = None
        
        # for OSC control
        self._j_eef = None                          # Jacobian for end effector
        self._mm = None                             # Mass matrix
        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        if self.graphics_device_id != -1:
            p = self.cfg["env"]["viewer"]["pos"]
            lookat = self.cfg["env"]["viewer"]["lookat"]
            cam_pos = gymapi.Vec3(p[0], p[1], p[2])
            cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.feet_air_time = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_loco_dof_vel = torch.zeros_like(self.loco_dof_vel)
        self.last_mani_dof_vel = torch.zeros_like(self.mani_dof_vel)

        # initialize some data used later on
        self.common_step_counter = 0 # for push robot
        self.extras = {} # for Tensorboard log
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg) # noise (bug)
        
        # for locomotion velocity commands 
        self.commands = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.lin_vel_scale, self.lin_vel_scale, self.ang_vel_scale], device=self.device, requires_grad=False,)
        
        # for manipulation position commands
        self.spherical_commands = torch.zeros(self.num_envs, 5, dtype=torch.float, device=self.device, requires_grad=False)
        self.relative_commands_pos = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False) 
        self.commands_sph = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.relative_eef_sph = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False) 
        
        # default joint positions
        self.named_loco_default_joint_angles = self.cfg["env"]["defaultlocoJointAngles"]
        self.mani_default_dof_pos = to_torch([0, 0, 0, 0, 0, 0], device=self.device)
        self.loco_default_dof_pos = torch.zeros_like(self.loco_dof_pos, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(12):
            name = self.dof_names[i]
            angle = self.named_loco_default_joint_angles[name]
            self.loco_default_dof_pos[:, i] = angle
        
        # reward episode sums (for tensorflow log)
        torch_zeros = lambda : torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.episode_sums = {"command_lin_vel_xy": torch_zeros(), "command_ang_vel_z": torch_zeros(), "lin_vel_z": torch_zeros(), "ang_vel_xy": torch_zeros(),
                             "arm_second_vel": torch_zeros(), "mani_joint_acc": torch_zeros(), "loco_joint_vel": torch_zeros(), "mani_joint_vel": torch_zeros(),
                             "imitate_reward": torch_zeros(), "mani_reward": torch_zeros(), "lin_vel_xyz": torch_zeros(), "ang_vel_rpy":torch_zeros(),
                             "orient":torch_zeros(), "torques":torch_zeros(), "loco_joint_acc": torch_zeros(), "stumble": torch_zeros(), "action_rate": torch_zeros(),
                             "base_position":torch_zeros(), "base_height":torch_zeros(), "airTime":torch_zeros(), "hip_cosmetic":torch_zeros()}

        # OSC Gains
        self.arm_kp = to_torch([150.] * 6, device=self.device)
        self.arm_kd = 2 * torch.sqrt(self.arm_kp)
        self.arm_kp_null = to_torch([10.] * 6, device=self.device)
        self.arm_kd_null = 2 * torch.sqrt(self.arm_kp_null)
        # Set control limits (osc: operation space control, cmd_limit : x y z r p y )
        self.arm_cmd_limit = to_torch([0.1, 0.1, 0.1, 0.5, 0.5, 0.5], device=self.device).unsqueeze(0)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        
        self.init_done = True

    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros_like(self.obs_buf[0])
        
        self.add_noise = self.cfg["env"]["learn"]["addNoise"]
        # noise_level = self.cfg["env"]["learn"]["noiseLevel"]
        # noise_vec[:3] = self.cfg["env"]["learn"]["linearVelocityNoise"] * noise_level * self.lin_vel_scale
        # noise_vec[3:6] = self.cfg["env"]["learn"]["angularVelocityNoise"] * noise_level * self.ang_vel_scale
        # noise_vec[6:9] = self.cfg["env"]["learn"]["gravityNoise"] * noise_level
        # noise_vec[9:12] = 0. # commands
        # noise_vec[12:24] = self.cfg["env"]["learn"]["dofPositionNoise"] * noise_level * self.dof_pos_scale
        # noise_vec[24:36] = self.cfg["env"]["learn"]["dofVelocityNoise"] * noise_level * self.dof_vel_scale
        # noise_vec[36:176] = self.cfg["env"]["learn"]["heightMeasurementNoise"] * noise_level * self.height_meas_scale
        # noise_vec[176:188] = 0. # previous actions
        return noise_vec
    
    def create_sim(self):
        self.up_axis_idx = 2 # index of up axis: Y=1, Z=2
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg["env"]["terrain"]["staticFriction"]
        plane_params.dynamic_friction = self.cfg["env"]["terrain"]["dynamicFriction"]
        plane_params.restitution = self.cfg["env"]["terrain"]["restitution"]
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        asset_file = self.cfg["env"]["urdfAsset"]["file"]
        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        if(self.num_actions == 18):
            asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        elif(self.num_actions == 12):
            asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        elif(self.num_actions == 6):
            asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.collapse_fixed_joints = self.cfg["env"]["urdfAsset"]["collapseFixedJoints"]
        asset_options.replace_cylinder_with_capsule = False
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = self.cfg["env"]["urdfAsset"]["fixBaseLink"]
        asset_options.density = 0.001
        asset_options.angular_damping = 0.0
        asset_options.linear_damping = 0.0
        asset_options.armature = 0.0
        asset_options.thickness = 0.01
        asset_options.disable_gravity = False
        usefulhound_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        # prepare friction randomization
        rigid_shape_prop = self.gym.get_asset_rigid_shape_properties(usefulhound_asset) # TODO FIX
        friction_range = self.cfg["env"]["learn"]["frictionRange"]
        num_buckets = 100

        # base start position & orientation
        friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device=self.device)
        self.base_init_state = to_torch(self.base_init_state, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        self.total_num_dof = self.gym.get_asset_dof_count(usefulhound_asset)
        self.hound_num_dof = self.total_num_dof - 6
        self.arm_num_dof = self.total_num_dof - self.hound_num_dof 
        self.num_bodies = self.gym.get_asset_rigid_body_count(usefulhound_asset)
        self.body_names = self.gym.get_asset_rigid_body_names(usefulhound_asset)
        self.dof_names = self.gym.get_asset_dof_names(usefulhound_asset)
        print("body parts num: {}, name: {}".format(len(self.body_names), self.body_names))
        foot_name = self.cfg["env"]["urdfAsset"]["footName"]
        knee_name = self.cfg["env"]["urdfAsset"]["kneeName"]
        shoulder_name = self.cfg["env"]["urdfAsset"]["shoulderName"]

        feet_names = [s for s in self.body_names if foot_name in s]
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        knee_names = [s for s in self.body_names if knee_name in s]
        self.knee_indices = torch.zeros(len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        shoulder_names = [s for s in self.body_names if shoulder_name in s]
        self.shoulder_indices = torch.zeros(len(shoulder_names), dtype=torch.long, device=self.device, requires_grad=False)
        

        dof_props = self.gym.get_asset_dof_properties(usefulhound_asset)
        houndarm_dof_stiffness = to_torch([0, 0, 0, 0, 0, 0], dtype=torch.float, device=self.device)
        houndarm_dof_damping = to_torch([0, 0, 0, 0, 0, 0], dtype=torch.float, device=self.device)

        self.mani_dof_lower_limits = []
        self.mani_dof_upper_limits = []
        self.mani_effort_limits = []

        for _i in range(self.arm_num_dof):
            i = _i + self.hound_num_dof

            if (self.num_actions == 12):
                dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
                # if self.physics_engine == gymapi.SIM_PHYSX:
                #     dof_props['stiffness'][i] = houndarm_dof_stiffness[_i]
                #     dof_props['damping'][i] = houndarm_dof_damping[_i]
                # else:
                dof_props['stiffness'][i] = 7000.0
                dof_props['damping'][i] = 50.0
            else : 
                dof_props['driveMode'][i] = gymapi.DOF_MODE_EFFORT
                if self.physics_engine == gymapi.SIM_PHYSX:
                    dof_props['stiffness'][i] = houndarm_dof_stiffness[_i]
                    dof_props['damping'][i] = houndarm_dof_damping[_i]
                else:
                    dof_props['stiffness'][i] = 7000.0
                    dof_props['damping'][i] = 50.0

            self.mani_dof_lower_limits.append(dof_props['lower'][i])
            self.mani_dof_upper_limits.append(dof_props['upper'][i])
            self.mani_effort_limits.append(dof_props['effort'][i])
        
        if self.TrainingMode == 'Manipulation':
            for i in range(self.hound_num_dof):
                dof_props['stiffness'][i] = 800 
                dof_props['damping'][i] = 20
        
        self.mani_dof_lower_limits = to_torch(self.mani_dof_lower_limits, device=self.device)
        self.mani_dof_upper_limits = to_torch(self.mani_dof_upper_limits, device=self.device)
        self.mani_effort_limits = to_torch(self.mani_effort_limits, device=self.device)
        self.houndarm_dof_speed_scales = torch.ones_like(self.mani_dof_lower_limits)

        # env origins
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        self.terrain_levels = torch.randint(0, self.cfg["env"]["terrain"]["maxInitMapLevel"]+1, (self.num_envs,), device=self.device) # TODO default only 0 
        self.terrain_types = torch.randint(0, self.cfg["env"]["terrain"]["numTerrains"], (self.num_envs,), device=self.device) # TODO torch.randint(0,20,(num_envs,))

        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        self.anymal_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row) # num_per_row : sqrt(num_envs)
            for s in range(len(rigid_shape_prop)):
                rigid_shape_prop[s].friction = friction_buckets[i % num_buckets] # TODO give friction randomly
            self.gym.set_asset_rigid_shape_properties(usefulhound_asset, rigid_shape_prop) 
            anymal_handle = self.gym.create_actor(env_handle, usefulhound_asset, start_pose, "UsefulHound", i, 0, 0)
            self.gym.set_actor_dof_properties(env_handle, anymal_handle, dof_props)
            self.envs.append(env_handle)
            self.anymal_handles.append(anymal_handle)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.anymal_handles[0], feet_names[i])
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.anymal_handles[0], knee_names[i])
        for i in range(len(shoulder_names)):
            self.shoulder_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.anymal_handles[0], shoulder_names[i])

        self.eef_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.anymal_handles[0], "joint6")
        self.base_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.anymal_handles[0], "trunk")
        self.real_eef_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.anymal_handles[0], "end_link")

        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(self.num_envs, -1, 13)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(self.num_envs, -1, 2)
        self.dof_pos = self.dof_state[:, :, 0]
        self.dof_vel = self.dof_state[:, :, 1]
        self.mani_dof_pos = self.dof_state[:, 12:, 0]
        self.mani_dof_vel = self.dof_state[:, 12:, 1]
        self.loco_dof_pos = self.dof_state[:, :12, 0]
        self.loco_dof_vel = self.dof_state[:, :12, 1]
        
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.base_states = gymtorch.wrap_tensor(actor_root_state)
        self.eef_state = self._rigid_body_state[:, self.eef_index, :]
        
        self.base_quat = self.base_states[:, 3:7]
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.base_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.base_states[:, 10:13])
        self.eef_lin_vel = quat_rotate_inverse(self.base_quat, self.eef_state[:, 7:10])
        self.eef_ang_vel = quat_rotate_inverse(self.base_quat, self.eef_state[:, 10:13])
        self.relative_eef_lin_vel = self.eef_lin_vel - self.base_lin_vel
        self.relative_eef_ang_vel = self.eef_ang_vel - self.base_lin_vel
        self.relative_eef_quat = quat_mul(self.base_quat, quat_conjugate(self.eef_state[:,3:7]))
        self.relative_eef_pos = quat_rotate_inverse(self.base_quat, self.eef_state[:,:3] - self.base_states[:,:3])

        # for OSC control
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "UsefulHound")
        jacobian = gymtorch.wrap_tensor(_jacobian)
        hand_joint_index = self.gym.get_actor_joint_dict(self.envs[0], self.anymal_handles[0])['joint6']
        print("jacobian_size: {}".format(jacobian.size())) # torch.Size([4096, 23, 6, 18])
        if(self.cfg["env"]["urdfAsset"]["fixBaseLink"] == True): 
            self._j_eef = jacobian[:, hand_joint_index, :, -6:]
        else:          
            self._j_eef = jacobian[:, hand_joint_index, :, 18:24]
        _massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, "UsefulHound")
        mm = gymtorch.wrap_tensor(_massmatrix)
        self._mm = mm[:, -6:, -6:]

    def check_termination(self):
        self.reset_buf = torch.norm(self.contact_forces[:, self.base_index, :], dim=1) > 1.
        self.reset_buf = self.reset_buf | torch.any(torch.norm(self.contact_forces[:, self.knee_indices, :], dim=2) > 1., dim=1)
        self.reset_buf = self.reset_buf | torch.any(torch.norm(self.contact_forces[:, self.shoulder_indices, :], dim=2) > 1., dim=1)
        self.reset_buf = self.reset_buf | (torch.norm(self.contact_forces[:, self.real_eef_index, :], dim=1) > 1.)
        time_out = self.progress_buf >= self.max_episode_length - 1  # no terminal reward for time-outs
        self.reset_buf = self.reset_buf | time_out

    def compute_observations(self):
        if(self.TrainingMode == "Locomotion_vel"):
            self.obs_buf = torch.cat((  
                                        self.base_lin_vel * self.lin_vel_scale, #3
                                        self.base_ang_vel  * self.ang_vel_scale, #3
                                        self.projected_gravity, # 3
                                        self.loco_dof_pos * self.dof_pos_scale, # 12
                                        self.loco_dof_vel * self.dof_vel_scale, # 12
                                        self.actions[:,:12], # 12
                                        self.commands[:, :3] * self.commands_scale,
                                        ), dim=-1)
            
        elif(self.maniControlMode == 'osc' and self.TrainingMode == "Manipulation"):
            self.obs_buf = torch.cat((  
                                        self.relative_eef_pos, # 3
                                        self.relative_eef_quat, # 4
                                        self.relative_commands_pos #3
                                        ), dim=-1)
        elif(self.maniControlMode == 'joint' and self.TrainingMode == "Manipulation"):
            self.obs_buf = torch.cat((  
                                        self.actions[:], # 6
                                        self.mani_dof_pos * self.dof_pos_scale, # 6
                                        self.mani_dof_vel * self.dof_vel_scale, # 6
                                        self.relative_eef_pos, # 3
                                        self.relative_eef_quat, # 4
                                        self.relative_commands_pos #3
                                        ), dim=-1)

    def compute_reward(self):
        self.relative_eef_sph[:,0], self.relative_eef_sph[:,1], self.relative_eef_sph[:,2] = spherical_to_cartesian(self.relative_eef_pos[:,0], self.relative_eef_pos[:,1], self.relative_eef_pos[:,2])
        self.commands_sph[:,0] = (self.spherical_commands[:,4]/self.spherical_commands[:,3]) * self.relative_eef_sph[:,0] + (1 - self.spherical_commands[:,4]/self.spherical_commands[:,3]) * self.spherical_commands[:,0]
        self.commands_sph[:,1] = (self.spherical_commands[:,4]/self.spherical_commands[:,3]) * self.relative_eef_sph[:,1] + (1 - self.spherical_commands[:,4]/self.spherical_commands[:,3]) * self.spherical_commands[:,1]
        self.commands_sph[:,2] = (self.spherical_commands[:,4]/self.spherical_commands[:,3]) * self.relative_eef_sph[:,2] + (1 - self.spherical_commands[:,4]/self.spherical_commands[:,3]) * self.spherical_commands[:,2]        
        x,y,z = spherical_to_cartesian(self.commands_sph[:,0], self.commands_sph[:,1], self.commands_sph[:,2])
        self.spherical_commands[:,4] += self.dt
        self.relative_commands_pos[:,0] = x
        self.relative_commands_pos[:,1] = y
        self.relative_commands_pos[:,2] = z

        # velocity command following reward (for Locomotion_vel)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2]) # yaw error
        rew_lin_vel_xy = torch.exp(-lin_vel_error/0.25) * self.rew_scales["lin_vel_xy"]
        rew_ang_vel_z = torch.exp(-ang_vel_error/0.25) * self.rew_scales["ang_vel_z"] # yaw error
        
        # arm command following reward
        distance_rewards = torch.exp(-7*torch.norm(self.relative_eef_pos - self.relative_commands_pos, dim=-1)) * self.rew_scales["eef_dist_scale"] 
        velocity_rewards = 0

        distance_in_reach = (torch.norm(self.relative_eef_pos - self.relative_commands_pos, dim=-1) < 0.1)
        velocity_rewards = torch.exp(-torch.norm(torch.cat((self.relative_eef_lin_vel, self.relative_eef_ang_vel), dim=-1), dim=-1)) * distance_in_reach * self.rew_scales["eef_vel_scale"]

        arm_rewards = distance_rewards + velocity_rewards 
        rew_arm_rewards = torch.clip(arm_rewards, 0., None)

        # default dof imitation reward
        rew_imitate = torch.exp(-5*torch.sum(torch.square(self.loco_dof_pos - self.loco_default_dof_pos), dim=1)) * self.rew_scales["rew_imitate"]

        # air time reward
        # contact = torch.norm(contact_forces[:, feet_indices, :], dim=2) > 1.
        contact = self.contact_forces[:, self.feet_indices, 2] > 1. # TODO if feet z force over 1, contact true
        first_contact = (self.feet_air_time > 0.) * contact # TODO if first contact, False. else, True
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) * self.rew_scales["air_time"] # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact

        # other base velocity penalties
        rew_lin_vel_z = torch.square(self.base_lin_vel[:, 2]) * self.rew_scales["lin_vel_z"]
        rew_ang_vel_xy = torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1) * self.rew_scales["ang_vel_xy"]

        # other base velocity penalties (for Manipulation)
        rew_lin_vel_xyz = torch.sum(torch.square(self.base_lin_vel[:, :3]), dim=1) * self.rew_scales["lin_vel_xyz"]
        rew_ang_vel_rpy = torch.sum(torch.square(self.base_ang_vel[:, :3]), dim=1) * self.rew_scales["ang_vel_rpy"]
        
        # base position penalties
        rew_basepos = torch.sum(torch.square(self.base_states[:, :2]), dim=1) * self.rew_scales["basepos"]

        # orientation penalty
        rew_orient = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1) * self.rew_scales["orient"]

        # base height penalty
        rew_base_height = torch.square(self.base_states[:, 2] - 0.52) * self.rew_scales["base_height"] # TODO add target base height to cfg  # 0.52

        # torque penalty
        rew_torque = torch.sum(torch.square(self.torques), dim=1) * self.rew_scales["torque"]

        # joint vel penalty
        rew_mani_dof_vel = torch.sum(torch.square(self.mani_dof_vel), dim=1) * self.rew_scales["mani_dof_vel"]
        rew_loco_dof_vel = torch.sum(torch.square(self.loco_dof_vel), dim=1) * self.rew_scales["loco_dof_vel"]

        # joint acc penalty
        rew_loco_joint_acc = torch.sum(torch.square(self.last_loco_dof_vel - self.loco_dof_vel), dim=1) * self.rew_scales["loco_joint_acc"]
        rew_mani_joint_acc = torch.sum(torch.square(self.last_mani_dof_vel - self.mani_dof_vel), dim=1) * self.rew_scales["mani_joint_acc"]

        # stumbling penalty
        stumble = (torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) > 5.) * (torch.abs(self.contact_forces[:, self.feet_indices, 2]) < 1.)
        rew_stumble = torch.sum(stumble, dim=1) * self.rew_scales["stumble"]

        # action rate penalty
        rew_action_rate = torch.sum(torch.square(self.last_actions - self.actions), dim=1) * self.rew_scales["action_rate"]
        
        # cosmetic penalty for hip motion
        rew_hip = torch.sum(torch.abs(self.loco_dof_pos[:, [0, 3, 6, 9]] - self.loco_default_dof_pos[:, [0, 3, 6, 9]]), dim=1)* self.rew_scales["hip"]

        # collision penalty
        # knee_contact = torch.norm(self.contact_forces[:, self.knee_indices, :], dim=2) > 1.
        # base_contact = torch.norm(self.contact_forces[:, self.shoulder_indices, :], dim=2) > 1.
        # rew_collision = torch.sum(knee_contact, dim=1) * self.rew_scales["collision"] + torch.sum(base_contact, dim=1) * self.rew_scales["collision"]# sum vs any ?

        # # add termination reward
        # self.rew_buf += self.rew_scales["termination"] * self.reset_buf * ~self.timeout_buf
        self.rew_buf = rew_hip + rew_lin_vel_xy + rew_ang_vel_z + rew_lin_vel_z + rew_ang_vel_xy + rew_arm_rewards + rew_mani_joint_acc + rew_mani_dof_vel + rew_torque + rew_loco_dof_vel + rew_lin_vel_xyz + rew_ang_vel_rpy + rew_orient + rew_basepos + rew_base_height + rew_loco_joint_acc + rew_stumble + rew_action_rate + rew_imitate + rew_airTime
        self.rew_buf = torch.clip(self.rew_buf, min=0., max=None)
        # log episode reward sums
        self.episode_sums["hip_cosmetic"] += rew_hip
        self.episode_sums["airTime"] += rew_airTime
        self.episode_sums["command_lin_vel_xy"] += rew_lin_vel_xy
        self.episode_sums["command_ang_vel_z"] += rew_ang_vel_z
        self.episode_sums["lin_vel_z"] += rew_lin_vel_z
        self.episode_sums["ang_vel_xy"] += rew_ang_vel_xy
        self.episode_sums["arm_second_vel"] += velocity_rewards
        self.episode_sums["mani_joint_acc"] += rew_mani_joint_acc
        self.episode_sums["loco_joint_vel"] += rew_loco_dof_vel
        self.episode_sums["mani_joint_vel"] += rew_mani_dof_vel
        self.episode_sums["imitate_reward"] += rew_imitate
        self.episode_sums["mani_reward"] += distance_rewards
        self.episode_sums["lin_vel_xyz"] += rew_lin_vel_xyz
        self.episode_sums["ang_vel_rpy"] += rew_ang_vel_rpy
        self.episode_sums["orient"] += rew_orient
        self.episode_sums["torques"] += rew_torque
        self.episode_sums["loco_joint_acc"] += rew_loco_joint_acc
        self.episode_sums["stumble"] += rew_stumble
        self.episode_sums["action_rate"] += rew_action_rate
        self.episode_sums["base_position"] += rew_basepos
        self.episode_sums["base_height"] += rew_base_height

    def reset_idx(self, env_ids):
        
        hound_positions_offset = torch_rand_float(0.5, 1.5, (len(env_ids), self.hound_num_dof), device=self.device)
        hound_velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.hound_num_dof), device=self.device)
        arm_positions_offset = torch_rand_float(0.5, 1.5, (len(env_ids), self.hound_num_dof), device=self.device)
        arm_velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.arm_num_dof), device=self.device)
        self.loco_dof_pos[env_ids] = self.loco_default_dof_pos[env_ids] * hound_positions_offset
        self.loco_dof_vel[env_ids] = hound_velocities
        self.mani_dof_pos[env_ids] = self.mani_default_dof_pos
        self.mani_dof_vel[env_ids] = arm_velocities
        self.dof_pos[env_ids] = torch.cat([self.loco_dof_pos[env_ids], self.mani_dof_pos[env_ids]], dim=-1) 
        self.dof_vel[env_ids] = torch.cat([self.loco_dof_vel[env_ids], self.mani_dof_vel[env_ids]], dim=-1)
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.base_states[env_ids] = self.base_init_state

        self.spherical_commands[env_ids,0] = torch_rand_float(self.sph_command_x_range[0], self.sph_command_x_range[1], (len(env_ids), 1), device=self.device).squeeze()
        self.spherical_commands[env_ids,1] = torch_rand_float(self.sph_command_y_range[0], self.sph_command_y_range[1], (len(env_ids), 1), device=self.device).squeeze()
        self.spherical_commands[env_ids,2] = torch_rand_float(self.sph_command_z_range[0], self.sph_command_z_range[1], (len(env_ids), 1), device=self.device).squeeze()
        self.spherical_commands[env_ids,3] = torch_rand_float(self.sph_command_T_range[0], self.sph_command_T_range[1], (len(env_ids), 1), device=self.device).squeeze()
        self.spherical_commands[env_ids,4] = 0.
        self.commands[env_ids, 0] = torch_rand_float(self.command_x_range[0], self.command_x_range[1], (len(env_ids), 1), device=self.device).squeeze()
        self.commands[env_ids, 1] = torch_rand_float(self.command_y_range[0], self.command_y_range[1], (len(env_ids), 1), device=self.device).squeeze()
        self.commands[env_ids, 3] = torch_rand_float(self.command_yaw_range[0], self.command_yaw_range[1], (len(env_ids), 1), device=self.device).squeeze()
        self.commands[env_ids] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.25).unsqueeze(1) # set small commands to zero

        reset_noise = torch.rand((len(env_ids), 6), device=self.device)
        self.mani_dof_noise = self.cfg["env"]["houndarmDofNoise"]
        pos = tensor_clamp(
            self.mani_default_dof_pos.unsqueeze(0) +
            self.mani_dof_noise * 2.0 * (reset_noise - 0.5),
            self.mani_dof_lower_limits.unsqueeze(0), self.mani_dof_upper_limits) # TODO limit are in urdf
        self.mani_dof_pos[env_ids, :] = pos

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.base_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        
        # sum hound pos and arm pos & effort
        total_pos = torch.cat([self.loco_dof_pos, self.mani_dof_pos], axis=1) 
        total_effort = torch.zeros_like(total_pos)

        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(total_pos),
                                                        gymtorch.unwrap_tensor(env_ids_int32),
                                                        len(env_ids_int32))
        
        self.gym.set_dof_actuation_force_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(total_effort),
                                                        gymtorch.unwrap_tensor(env_ids_int32),
                                                        len(env_ids_int32))
        self.last_actions[env_ids] = 0.
        self.last_loco_dof_vel[env_ids] = 0.
        self.last_mani_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())


    def push_robots(self):
        self.base_states[:, 7:9] = torch_rand_float(-1., 1., (self.num_envs, 2), device=self.device) # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.base_states))
    
    def _compute_osc_torques(self, dpose):
        # Solve for Operational Space Control # Paper: khatib.stanford.edu/publications/pdfs/Khatib_1987_RA.pdf
        # Helpful resource: studywolf.wordpress.com/2013/09/17/robot-control-4-operation-space-control/
        q, qd = self.mani_dof_pos[:, :6], self.mani_dof_vel[:, :6]
    
        mm_inv = torch.inverse(self._mm)
        m_eef_inv = self._j_eef @ mm_inv @ torch.transpose(self._j_eef, 1, 2)
        m_eef = torch.inverse(m_eef_inv)

        # Transform our cartesian action `dpose` into joint torques `u`
        vel = torch.cat((self.relative_eef_lin_vel, self.relative_eef_ang_vel), dim=-1)
        u = torch.transpose(self._j_eef, 1, 2) @ m_eef @ (
                self.arm_kp * dpose - self.arm_kd * (vel)).unsqueeze(-1)

        # Nullspace control torques `u_null` prevents large changes in joint configuration
        # They are added into the nullspace of OSC so that the end effector orientation remains constant
        # roboticsproceedings.org/rss07/p31.pdf
        j_eef_inv = m_eef @ self._j_eef @ mm_inv
        u_null = self.arm_kd_null * -qd + self.arm_kp_null * ((self.mani_default_dof_pos[:6] - q + np.pi) % (2 * np.pi) - np.pi)
        u_null[:, 6:] *= 0
        u_null = self._mm @ u_null.unsqueeze(-1)
        u += (torch.eye(6, device=self.device).unsqueeze(0) - torch.transpose(self._j_eef, 1, 2) @ j_eef_inv) @ u_null
        # Clip the values to be within valid effort range
        u = tensor_clamp(u.squeeze(-1),
                         -self.mani_effort_limits[:6].unsqueeze(0), self.mani_effort_limits[:6].unsqueeze(0))

        return u

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)     
        
        for i in range(self.decimation):
            if(self.num_actions == 18 and self.maniControlMode == 'osc'):      
                mani_input_torque = self.actions[:,12:]
                mani_input_torque = mani_input_torque * self.arm_cmd_limit / self.mani_action_scale
                mani_input_torque = self._compute_osc_torques(dpose=mani_input_torque)
                torques = torch.clip(self.Kp*(self.loco_action_scale*self.actions[:,:12] + self.loco_default_dof_pos - self.loco_dof_pos) - self.Kd*self.loco_dof_vel, -80., 80.)
                torques = torch.cat([torques, mani_input_torque], axis=1) 
                self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(torques))  
                self.torques = torques.view(self.torques.shape)

            elif(self.num_actions == 6 and self.maniControlMode == 'osc'):
                mani_input_torque = self.actions[:,:]
                mani_input_torque = mani_input_torque * self.arm_cmd_limit / self.mani_action_scale
                mani_input_torque = self._compute_osc_torques(dpose=mani_input_torque)
                torques = torch.zeros(self.num_envs, 12, device=self.device)
                torques = torch.cat([torques, mani_input_torque], axis=1) 
                self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(torques))  
                position_input = torch.cat([self.loco_default_dof_pos,self.mani_dof_pos], axis=1)
                self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(position_input))
                self.torques = torques[:,12:]
            
            elif(self.num_actions == 6 and self.maniControlMode == 'joint'):
                mani_input_torque = self.actions[:,:]
                mani_input_torque = mani_input_torque * self.arm_cmd_limit / self.mani_action_scale
                # mani_input_torque = tensor_clamp(mani_input_torque.squeeze(-1), -self.mani_effort_limits[:6].unsqueeze(0), self.mani_effort_limits[:6].unsqueeze(0))
                torques = torch.zeros(self.num_envs, 12, device=self.device)
                torques = torch.cat([torques, mani_input_torque], axis=1) 
                self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(torques))  
                position_input = torch.cat([self.loco_default_dof_pos,self.mani_dof_pos], axis=1)
                self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(position_input))
                self.torques = torques[:,12:]

            elif(self.num_actions == 12 and self.maniControlMode == 'osc'):
                mani_input_torque = torch.zeros(self.num_envs, 6, device=self.device)
                torques = torch.clip(self.Kp*(self.loco_action_scale*self.actions[:,:12] + self.loco_default_dof_pos - self.loco_dof_pos) - self.Kd*self.loco_dof_vel, -80., 80.)
                torques = torch.cat([torques, mani_input_torque], axis=1) 
                self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(torques)) 
                mani_zero_pos = torch.zeros(self.num_envs, 6, device=self.device)
                position_input = torch.cat([self.loco_dof_pos,mani_zero_pos], axis=1)
                self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(position_input))
                self.torques = torques[:,:12]
            
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)

    def post_physics_step(self):
        # self.gym.refresh_dof_state_tensor(self.sim) # done in step
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)
        self.progress_buf += 1
        self.randomize_buf += 1

        if(self.pushrobot == True):
            self.common_step_counter += 1
            if self.common_step_counter % self.push_interval == 0:
                self.push_robots()

        # prepare quantities
        self.base_quat = self.base_states[:, 3:7]
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.base_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.base_states[:, 10:13])
        self.eef_lin_vel = quat_rotate_inverse(self.base_quat, self.eef_state[:, 7:10])
        self.eef_ang_vel = quat_rotate_inverse(self.base_quat, self.eef_state[:, 10:13])
        self.relative_eef_lin_vel = self.eef_lin_vel - self.base_lin_vel
        self.relative_eef_ang_vel = self.eef_ang_vel - self.base_lin_vel

        self.relative_eef_pos = quat_rotate_inverse(self.base_quat, self.eef_state[:,:3] - self.base_states[:,:3])
        self.relative_eef_quat = quat_mul(self.base_quat, quat_conjugate(self.eef_state[:,3:7]))
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        
        # for Locomotion_vel
        forward = quat_apply(self.base_quat, self.forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0])
        self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        if (self.add_noise):
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
        
        self.last_actions[:] = self.actions[:]
        self.last_loco_dof_vel[:] = self.loco_dof_vel[:]
        self.last_mani_dof_vel[:] = self.mani_dof_vel[:]  

   

from isaacgym.terrain_utils import *

@torch.jit.script
def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)

@torch.jit.script
def wrap_to_pi(angles):
    angles %= 2*np.pi
    angles -= 2*np.pi * (angles > np.pi)
    return angles
