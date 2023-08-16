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
class UsefulHoundImitate(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):

        self.cfg = cfg
        self.height_samples = None
        self.custom_origins = False
        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.init_done = False

        # normalization
        self.lin_vel_scale = self.cfg["env"]["learn"]["linearVelocityScale"]
        self.ang_vel_scale = self.cfg["env"]["learn"]["angularVelocityScale"]
        self.dof_pos_scale = self.cfg["env"]["learn"]["dofPositionScale"]
        self.dof_vel_scale = self.cfg["env"]["learn"]["dofVelocityScale"]
        self.height_meas_scale = self.cfg["env"]["learn"]["heightMeasurementScale"]
        self.action_scale = self.cfg["env"]["control"]["actionScale"]

        ###########################################################################
        ## for arm ################################################################
        self.arm_action_scale = self.cfg["env"]["control"]["houndarmactionScale"]
        self.houndarm_dof_noise = self.cfg["env"]["houndarmDofNoise"]
        self.arm_reward_settings = {
            "r_dist_scale": self.cfg["env"]["learn"]["distRewardScale"],
            "r_vel_scale": self.cfg["env"]["learn"]["velRewardScale"],
        }
        
        self.arm_control_type = self.cfg["env"]["houndarmcontrolType"]

        # command ranges
        self.arm_command_x_range = self.cfg["env"]["randomArmCommandPositionRanges"]["x"]
        self.arm_command_y_range = self.cfg["env"]["randomArmCommandPositionRanges"]["y"]
        self.arm_command_z_range = self.cfg["env"]["randomArmCommandPositionRanges"]["z"]

        self.sph_command_x_range = self.cfg["env"]["randomArmCommandPositionRanges"]["sph_l"]
        self.sph_command_y_range = self.cfg["env"]["randomArmCommandPositionRanges"]["sph_p"]
        self.sph_command_z_range = self.cfg["env"]["randomArmCommandPositionRanges"]["sph_y"]
        self.sph_command_T_range = self.cfg["env"]["randomArmCommandPositionRanges"]["sph_T"]
        # Tensor placeholders
        self._root_state = None             # State of root body        (n_envs, 13)
        self._dof_state = None  # State of all joints       (n_envs, n_dof)
        self._q = None  # Joint positions           (n_envs, n_dof)
        self._qd = None                     # Joint velocities          (n_envs, n_dof)
        self._rigid_body_state = None  # State of all rigid bodies             (n_envs, n_bodies, 13)
        self._contact_forces = None     # Contact forces in sim
        self._eef_state = None  # end effector state (at grasping point)

        self.eef_lin_vel = None
        self._eef_lf_state = None  # end effector state (at left fingertip)
        self._eef_rf_state = None  # end effector state (at left fingertip)
        self._j_eef = None  # Jacobian for end effector
        self._mm = None  # Mass matrix
        # self._arm_control = None  # Tensor buffer for controlling arm
        self._gripper_control = None  # Tensor buffer for controlling gripper
        self._pos_control = None            # Position actions
        # self._effort_control = None         # Torque actions
        self._houndarm_effort_limits = None        # Actuator effort limits for houndarm
        self._global_indices = None         # Unique indices corresponding to all envs in flattened array
       
        ###########################################################################
        ###########################################################################

        # reward scales
        self.rew_scales = {}
        self.rew_scales["termination"] = self.cfg["env"]["learn"]["terminalReward"] 
        self.rew_scales["lin_vel_xy"] = self.cfg["env"]["learn"]["linearVelocityXYRewardScale"] 
        self.rew_scales["lin_vel_z"] = self.cfg["env"]["learn"]["linearVelocityZRewardScale"] 
        self.rew_scales["ang_vel_z"] = self.cfg["env"]["learn"]["angularVelocityZRewardScale"] 
        self.rew_scales["ang_vel_xy"] = self.cfg["env"]["learn"]["angularVelocityXYRewardScale"] 
        self.rew_scales["orient"] = self.cfg["env"]["learn"]["orientationRewardScale"] 
        self.rew_scales["torque"] = self.cfg["env"]["learn"]["torqueRewardScale"]
        self.rew_scales["joint_acc"] = self.cfg["env"]["learn"]["jointAccRewardScale"]
        self.rew_scales["base_height"] = self.cfg["env"]["learn"]["baseHeightRewardScale"]
        self.rew_scales["air_time"] = self.cfg["env"]["learn"]["feetAirTimeRewardScale"]
        self.rew_scales["collision"] = self.cfg["env"]["learn"]["kneeCollisionRewardScale"]
        self.rew_scales["stumble"] = self.cfg["env"]["learn"]["feetStumbleRewardScale"]
        self.rew_scales["action_rate"] = self.cfg["env"]["learn"]["actionRateRewardScale"]
        self.rew_scales["hip"] = self.cfg["env"]["learn"]["hipRewardScale"]
        self.rew_scales["arm_joint_acc"] = self.cfg["env"]["learn"]["armjointAccRewardScale"]
        self.rew_scales["arm_dof_vel"] = self.cfg["env"]["learn"]["armjointVelRewardScale"]
        self.rew_scales["hound_dof_vel"] = self.cfg["env"]["learn"]["houndjointVelRewardScale"]
        #command ranges
        self.command_x_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_x"]
        self.command_y_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_y"]
        self.command_yaw_range = self.cfg["env"]["randomCommandVelocityRanges"]["yaw"]
        
        # base init state
        pos = self.cfg["env"]["baseInitState"]["pos"]
        rot = self.cfg["env"]["baseInitState"]["rot"]
        v_lin = self.cfg["env"]["baseInitState"]["vLinear"]
        v_ang = self.cfg["env"]["baseInitState"]["vAngular"]
        self.base_init_state = pos + rot + v_lin + v_ang

        # default joint positions
        self.named_hound_default_joint_angles = self.cfg["env"]["defaultJointAngles"]

        # other
        self.decimation = self.cfg["env"]["control"]["decimation"]
        self.dt = self.decimation * self.cfg["sim"]["dt"]
        self.max_episode_length_s = self.cfg["env"]["learn"]["episodeLength_s"] 
        self.max_episode_length = int(self.max_episode_length_s/ self.dt + 0.5)
        self.push_interval = int(self.cfg["env"]["learn"]["pushInterval_s"] / self.dt + 0.5)
        self.allow_knee_contacts = self.cfg["env"]["learn"]["allowKneeContacts"]
        self.Kp = self.cfg["env"]["control"]["stiffness"]
        self.Kd = self.cfg["env"]["control"]["damping"]
        self.curriculum = self.cfg["env"]["terrain"]["curriculum"]

        for key in self.rew_scales.keys():
            self.rew_scales[key] *= self.dt

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        if self.graphics_device_id != -1:
            p = self.cfg["env"]["viewer"]["pos"]
            lookat = self.cfg["env"]["viewer"]["lookat"]
            cam_pos = gymapi.Vec3(p[0], p[1], p[2])
            cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.total_num_dof, 2)[:, :, 0] 
        self.dof_vel = self.dof_state.view(self.num_envs, self.total_num_dof, 2)[:, :, 1] 
        self.hound_dof_pos = self.dof_state.view(self.num_envs, self.total_num_dof, 2)[:, 0:12, 0] # FIX
        self.hound_dof_vel = self.dof_state.view(self.num_envs, self.total_num_dof, 2)[:, 0:12, 1] # FIX
        self.arm_dof_pos = self.dof_state.view(self.num_envs, self.total_num_dof, 2)[:, 12:, 0] # FIX
        self.arm_dof_vel = self.dof_state.view(self.num_envs, self.total_num_dof, 2)[:, 12:, 1] # FIX
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.commands = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.lin_vel_scale, self.lin_vel_scale, self.ang_vel_scale], device=self.device, requires_grad=False,)
        self.relative_commands_pos = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False) 
        self.commands_sph = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.feet_air_time = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_hound_dof_vel = torch.zeros_like(self.hound_dof_vel)
        self.last_arm_dof_vel = torch.zeros_like(self.arm_dof_vel)
        self.height_points = self.init_height_points()
        self.measured_heights = None
        self.relative_eef_sph = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False) 

        # joint positions offsets
        self.hound_default_dof_pos = torch.zeros_like(self.hound_dof_pos, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(12):
            name = self.dof_names[i]
            angle = self.named_hound_default_joint_angles[name]
            self.hound_default_dof_pos[:, i] = angle
        # reward episode sums
        torch_zeros = lambda : torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        # self.episode_sums = {"lin_vel_xy": torch_zeros(), "lin_vel_z": torch_zeros(), "ang_vel_z": torch_zeros(), "ang_vel_xy": torch_zeros(),
        #                      "orient": torch_zeros(), "torques": torch_zeros(), "joint_acc": torch_zeros(), "base_height": torch_zeros(),
        #                      "air_time": torch_zeros(), "collision": torch_zeros(), "stumble": torch_zeros(), "action_rate": torch_zeros(), "hip": torch_zeros()}
        self.episode_sums = {"imitate_reward": torch_zeros(), "lin_vel": torch_zeros(), "ang_vel": torch_zeros(), "arm_reward": torch_zeros(),
                             "orient": torch_zeros(), "torques": torch_zeros(), "joint_acc": torch_zeros(), "base_height": torch_zeros(),
                             "stumble": torch_zeros(), "action_rate": torch_zeros(), "base_position": torch_zeros(), "arm_joint_acc":torch_zeros(),
                             "arm_joint_vel":torch_zeros(), "hound_joint_vel":torch_zeros(), "arm_second_vel": torch_zeros(), "terminate": torch_zeros(), "terminate_check": torch_zeros()}
        
        ###########################################################################
        ## for arm ################################################################
        self.houndarm_default_dof_pos = to_torch(
            [0, 0, 0, 0, 0, 0], device=self.device
        )
        # OSC Gains
        self.arm_kp = to_torch([150.] * 6, device=self.device)
        self.arm_kd = 2 * torch.sqrt(self.arm_kp)
        self.arm_kp_null = to_torch([10.] * 6, device=self.device)
        self.arm_kd_null = 2 * torch.sqrt(self.arm_kp_null)
        #self.cmd_limit = None                   # filled in later

        # Set control limits (osc: operation space control, cmd_limit : x y z r p y )
        self.arm_cmd_limit = to_torch([0.1, 0.1, 0.1, 0.5, 0.5, 0.5], device=self.device).unsqueeze(0)

        # TODO add command
        self.arm_commands = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.spherical_commands = torch.zeros(self.num_envs, 5, dtype=torch.float, device=self.device, requires_grad=False)
        self.state_arm_commands = torch.zeros_like(self.arm_commands)
        ###########################################################################
        ###########################################################################
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        
        self.init_done = True

    def create_sim(self):
        self.up_axis_idx = 2 # index of up axis: Y=1, Z=2
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        terrain_type = self.cfg["env"]["terrain"]["terrainType"] 
        if terrain_type=='plane':
            self._create_ground_plane()
        elif terrain_type=='trimesh':
            self._create_trimesh()
            self.custom_origins = True
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg["env"]["learn"]["addNoise"]
        noise_level = self.cfg["env"]["learn"]["noiseLevel"]
        noise_vec[:3] = self.cfg["env"]["learn"]["linearVelocityNoise"] * noise_level * self.lin_vel_scale
        noise_vec[3:6] = self.cfg["env"]["learn"]["angularVelocityNoise"] * noise_level * self.ang_vel_scale
        noise_vec[6:9] = self.cfg["env"]["learn"]["gravityNoise"] * noise_level
        noise_vec[9:12] = 0. # commands
        noise_vec[12:24] = self.cfg["env"]["learn"]["dofPositionNoise"] * noise_level * self.dof_pos_scale
        noise_vec[24:36] = self.cfg["env"]["learn"]["dofVelocityNoise"] * noise_level * self.dof_vel_scale
        noise_vec[36:176] = self.cfg["env"]["learn"]["heightMeasurementNoise"] * noise_level * self.height_meas_scale
        noise_vec[176:188] = 0. # previous actions
        return noise_vec

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg["env"]["terrain"]["staticFriction"]
        plane_params.dynamic_friction = self.cfg["env"]["terrain"]["dynamicFriction"]
        plane_params.restitution = self.cfg["env"]["terrain"]["restitution"]
        self.gym.add_ground(self.sim, plane_params)

    def _create_trimesh(self):
        self.terrain = Terrain(self.cfg["env"]["terrain"], num_robots=self.num_envs)
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]
        tm_params.transform.p.x = -self.terrain.border_size 
        tm_params.transform.p.y = -self.terrain.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg["env"]["terrain"]["staticFriction"]
        tm_params.dynamic_friction = self.cfg["env"]["terrain"]["dynamicFriction"]
        tm_params.restitution = self.cfg["env"]["terrain"]["restitution"]

        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)   
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_envs(self, num_envs, spacing, num_per_row):
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        asset_file = self.cfg["env"]["urdfAsset"]["file"]
        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        asset_options.collapse_fixed_joints = self.cfg["env"]["urdfAsset"]["collapseFixedJoints"] # false
        asset_options.replace_cylinder_with_capsule = False
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = self.cfg["env"]["urdfAsset"]["fixBaseLink"]
        asset_options.density = 0.001
        asset_options.angular_damping = 0.0
        asset_options.linear_damping = 0.0
        asset_options.armature = 0.0
        asset_options.thickness = 0.01
        asset_options.disable_gravity = False
     
        anymal_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        
        self.total_num_dof = self.gym.get_asset_dof_count(anymal_asset) # FIX
        self.hound_num_dof = self.total_num_dof - 6 # FIX
        self.arm_num_dof = self.total_num_dof - self.hound_num_dof 

        self.num_bodies = self.gym.get_asset_rigid_body_count(anymal_asset)

        # prepare friction randomization
        rigid_shape_prop = self.gym.get_asset_rigid_shape_properties(anymal_asset) # TODO FIX
        friction_range = self.cfg["env"]["learn"]["frictionRange"]
        num_buckets = 100
        friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device=self.device)

        self.base_init_state = to_torch(self.base_init_state, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
   

        body_names = self.gym.get_asset_rigid_body_names(anymal_asset)
        self.dof_names = self.gym.get_asset_dof_names(anymal_asset)
        print("body parts num: {}, name: {}".format(len(body_names), body_names))
        foot_name = self.cfg["env"]["urdfAsset"]["footName"]
        knee_name = self.cfg["env"]["urdfAsset"]["kneeName"]
        base_name = self.cfg["env"]["urdfAsset"]["baseName"]
        endpoint_name = self.cfg["env"]["urdfAsset"]["endpointName"]

        feet_names = [s for s in body_names if foot_name in s]
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        knee_names = [s for s in body_names if knee_name in s]
        self.knee_indices = torch.zeros(len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        base_names = [s for s in body_names if base_name in s]
        self.base_indices = torch.zeros(len(base_names), dtype=torch.long, device=self.device, requires_grad=False)
        # eef_names = [s for s in body_names if endpoint_name in s]
        # self.eef_indices = torch.zeros(len(eef_names), dtype=torch.long, device=self.device, requires_grad=False)
        self.base_index = 0

        dof_props = self.gym.get_asset_dof_properties(anymal_asset)

        ######################################################################
        ## for arm ###########################################################
        houndarm_dof_stiffness = to_torch([0, 0, 0, 0, 0, 0], dtype=torch.float, device=self.device)
        houndarm_dof_damping = to_torch([0, 0, 0, 0, 0, 0], dtype=torch.float, device=self.device)

        self.houndarm_dof_lower_limits = []
        self.houndarm_dof_upper_limits = []
        self._houndarm_effort_limits = []

        for _i in range(self.arm_num_dof):
            i = _i + self.hound_num_dof
            dof_props['driveMode'][i] = gymapi.DOF_MODE_EFFORT
            if self.physics_engine == gymapi.SIM_PHYSX:
                dof_props['stiffness'][i] = houndarm_dof_stiffness[_i]
                dof_props['damping'][i] = houndarm_dof_damping[_i]
            else:
                dof_props['stiffness'][i] = 7000.0
                dof_props['damping'][i] = 50.0

            self.houndarm_dof_lower_limits.append(dof_props['lower'][i])
            self.houndarm_dof_upper_limits.append(dof_props['upper'][i])
            self._houndarm_effort_limits.append(dof_props['effort'][i])
        # for i in range(self.hound_num_dof):
        #     dof_props['stiffness'][i] = 5* 10**20 #7000.0
        #     dof_props['damping'][i] = 500.0
        self.houndarm_dof_lower_limits = to_torch(self.houndarm_dof_lower_limits, device=self.device)
        self.houndarm_dof_upper_limits = to_torch(self.houndarm_dof_upper_limits, device=self.device)
        self._houndarm_effort_limits = to_torch(self._houndarm_effort_limits, device=self.device)
        self.houndarm_dof_speed_scales = torch.ones_like(self.houndarm_dof_lower_limits)
        #######################################################################
        #######################################################################

        # env origins
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        if not self.curriculum: self.cfg["env"]["terrain"]["maxInitMapLevel"] = self.cfg["env"]["terrain"]["numLevels"] - 1 # TODO default true, so this if not work
        self.terrain_levels = torch.randint(0, self.cfg["env"]["terrain"]["maxInitMapLevel"]+1, (self.num_envs,), device=self.device) # TODO default only 0 
        self.terrain_types = torch.randint(0, self.cfg["env"]["terrain"]["numTerrains"], (self.num_envs,), device=self.device) # TODO torch.randint(0,20,(num_envs,))
        if self.custom_origins: # TODO if plane: false
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            spacing = 0.

        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        self.anymal_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row) # num_per_row : sqrt(num_envs)
            if self.custom_origins: # TODO if plane: false
                self.env_origins[i] = self.terrain_origins[self.terrain_levels[i], self.terrain_types[i]]
                pos = self.env_origins[i].clone()
                pos[:2] += torch_rand_float(-1., 1., (2, 1), device=self.device).squeeze(1)
                start_pose.p = gymapi.Vec3(*pos)

            for s in range(len(rigid_shape_prop)):
                rigid_shape_prop[s].friction = friction_buckets[i % num_buckets] # TODO give friction randomly
            self.gym.set_asset_rigid_shape_properties(anymal_asset, rigid_shape_prop) 
            anymal_handle = self.gym.create_actor(env_handle, anymal_asset, start_pose, "UsefulHound", i, 0, 0)
            self.gym.set_actor_dof_properties(env_handle, anymal_handle, dof_props)
            self.envs.append(env_handle)
            self.anymal_handles.append(anymal_handle)

        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.anymal_handles[0], feet_names[i])
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.anymal_handles[0], knee_names[i])
        for i in range(len(base_names)):
            self.base_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.anymal_handles[0], base_names[i])
        # for i in range(len(eef_names)):
        #     self.eef_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.anymal_handles[0], eef_names[i])
        self.eef_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.anymal_handles[0], "joint6")
        self.base_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.anymal_handles[0], "trunk")
        self.real_eef_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.anymal_handles[0], "end_link")

        ##################################################################################
        ## for arm #######################################################################
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(self.num_envs, -1, 13)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(self.num_envs, -1, 2)
        self._q = self.dof_state[:, 12:, 0]
        self._qd = self.dof_state[:, 12:, 1]
        self._root_state = self._rigid_body_state[:, self.base_index, :]
        self._eef_state = self._rigid_body_state[:, self.eef_index, :]
        
        self.base_quat = self._root_state[:, 3:7]
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self._root_state[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self._root_state[:, 10:13])
        
        
        self.eef_lin_vel = quat_rotate_inverse(self.base_quat, self._eef_state[:, 7:10])
        self.eef_ang_vel = quat_rotate_inverse(self.base_quat, self._eef_state[:, 10:13])
        self.relative_eef_lin_vel = self.eef_lin_vel - self.base_lin_vel
        self.relative_eef_ang_vel = self.eef_ang_vel - self.base_lin_vel
        self.relative_eef_quat = quat_mul(self.base_quat, quat_conjugate(self._eef_state[:,3:7]))
        self.relative_eef_pos = quat_rotate_inverse(self.base_quat, self._eef_state[:,:3] - self._root_state[:,:3])

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
        # print("massmatrix size: {}".format(mm.size()))
        self._mm = mm[:, -6:, -6:]
        
        # Initialize actions
        self._pos_control = torch.zeros((self.num_envs, self.arm_num_dof), dtype=torch.float, device=self.device)
        # self._effort_control = torch.zeros_like(self._pos_control)
        # Initialize control
        # self._arm_control = self._effort_control[:, :6]
        # Initialize indices (if there is other object in gym, it will be useful)
        self._global_indices = torch.arange(self.num_envs, dtype=torch.int32, device=self.device).view(self.num_envs, -1)
        ##################################################################################
        ##################################################################################

    def check_termination(self):
        # print("sssssssssssssssssssssssssssssssssssssssssss")
        # print(self.contact_forces.size())
        # test=torch.norm(self.contact_forces[:, self.real_eef_index, :], dim=1)
        # print(test.size())
        # print(self.reset_buf.size())
        self.reset_buf = torch.norm(self.contact_forces[:, self.base_index, :], dim=1) > 1.
        self.reset_buf = self.reset_buf | torch.any(torch.norm(self.contact_forces[:, self.knee_indices, :], dim=2) > 1., dim=1)
        self.reset_buf = self.reset_buf | torch.any(torch.norm(self.contact_forces[:, self.base_indices, :], dim=2) > 1., dim=1)
        self.reset_buf = self.reset_buf | (torch.norm(self.contact_forces[:, self.real_eef_index, :], dim=1) > 1.)
        
        #self.reset_buf = self.reset_buf | torch.any(self.contact_forces[:, self.feet_indices, 2] < 1., dim=1)

        # self.reset_buf = self.reset_buf | torch.any(torch.norm(self.contact_forces[:, self.calf_indices, :], dim=2) > 1., dim=1)
        time_out = self.progress_buf >= self.max_episode_length - 1  # no terminal reward for time-outs
        #contact = torch.norm(contact_forces[:, feet_indices, :], dim=2) > 1.
        # contact = self.contact_forces[:, self.feet_indices, 2] > 1.

        self.reset_buf = self.reset_buf | time_out

        # self.reset_buf = torch.norm(self.contact_forces[:, self.base_index, :], dim=1) > 1.
        # if not self.allow_knee_contacts:
        #     knee_contact = torch.norm(self.contact_forces[:, self.knee_indices, :], dim=2) > 1.
        #     self.reset_buf |= torch.any(knee_contact, dim=1)

        # self.reset_buf = torch.where(self.progress_buf >= self.max_episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf)

    def compute_observations(self):
        # self.measured_heights = self.get_heights()
        # heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.height_meas_scale
        # self.relative_commands_pos = quat_rotate_inverse(self.base_quat, self.arm_commands[:, :3] - self._root_state[:, :3])
        self.obs_buf = torch.cat((  
                                    self.base_lin_vel * self.lin_vel_scale, #3
                                    self.base_ang_vel  * self.ang_vel_scale, #3
                                    self.projected_gravity, # 3
                                    # self.dof_pos * self.dof_pos_scale, # 18
                                    # self.dof_vel * self.dof_vel_scale, # 18
                                    self.hound_dof_pos * self.dof_pos_scale, # 12
                                    self.hound_dof_vel * self.dof_vel_scale, # 12
                                    self.actions[:,:12], # 12

                                    self.relative_eef_pos, # 3
                                    self.relative_eef_quat, # 4
                                    self.relative_commands_pos #3
                                    ), dim=-1)
        # self.obs_buf = torch.cat((  self._eef_state[:, :3],  # eef_pos
        #                             self._eef_state[:, 3:7], # eef_quat
        #                             self.arm_commands[:,:]   # commands
        #                             ), dim=-1)

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
        # self.relative_commands_pos = quat_rotate_inverse(self.base_quat, self.arm_commands[:, :3] - self._root_state[:, :3])
        #distance_rewards = 1 - torch.tanh(10.0 * (torch.norm(self.relative_eef_pos - self.relative_commands_pos, dim=-1)))  #1 / (1 + torch.norm(states["eef_pos"] - states["commands"], dim=-1))
        distance_rewards = torch.exp(-7*torch.norm(self.relative_eef_pos - self.relative_commands_pos, dim=-1)) * self.arm_reward_settings["r_dist_scale"] 
        velocity_rewards = 0
        # print("relative_commands_pos: {}".format(self.relative_commands_pos[0,:]))
        # print("     relative_eef_pos: {}".format(self.relative_eef_pos[0,:]))
        distance_in_reach = (torch.norm(self.relative_eef_pos - self.relative_commands_pos, dim=-1) < 0.1)
        velocity_rewards = torch.exp(-torch.norm(torch.cat((self.relative_eef_lin_vel, self.relative_eef_ang_vel), dim=-1), dim=-1)) * distance_in_reach * self.arm_reward_settings["r_vel_scale"] * 0
        check = distance_in_reach*1
        check_check = 1
        # check=torch.cat((self.relative_eef_lin_vel, self.relative_eef_ang_vel), dim=-1)
        # print("vel_struct: {}".format(check.size()))
        arm_rewards = distance_rewards + velocity_rewards 
        arm_rewards = torch.clip(arm_rewards, 0., None)

        # other base velocity penalties
        rew_lin_vel = torch.sum(torch.square(self.base_lin_vel[:, :3]), dim=1) * -1 #self.rew_scales["lin_vel_z"]
        rew_ang_vel = torch.sum(torch.square(self.base_ang_vel[:, :3]), dim=1) * -1 #self.rew_scales["ang_vel_xy"]
        
        # base position penalties
        rew_basepos = torch.sum(torch.square(self.root_states[:, :2]), dim=1) * -1

        # orientation penalty
        rew_orient = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1) * self.rew_scales["orient"]

        # base height penalty
        rew_base_height = torch.square(self.root_states[:, 2] - 0.52) * self.rew_scales["base_height"] # TODO add target base height to cfg  # 0.52

        # torque penalty
        rew_torque = torch.sum(torch.square(self.torques), dim=1) * self.rew_scales["torque"]

        # joint vel penalty
        rew_arm_dof_vel = torch.sum(torch.square(self.arm_dof_vel), dim=1) * self.rew_scales["arm_dof_vel"]
        rew_hound_dof_vel = torch.sum(torch.square(self.hound_dof_vel), dim=1) * self.rew_scales["hound_dof_vel"]

        # joint acc penalty
        rew_joint_acc = torch.sum(torch.square(self.last_hound_dof_vel - self.hound_dof_vel), dim=1) * self.rew_scales["joint_acc"]
        rew_arm_joint_acc = torch.sum(torch.square(self.last_arm_dof_vel - self.arm_dof_vel), dim=1) * self.rew_scales["arm_joint_acc"]

        # collision penalty
        # knee_contact = torch.norm(self.contact_forces[:, self.knee_indices, :], dim=2) > 1.
        # base_contact = torch.norm(self.contact_forces[:, self.base_indices, :], dim=2) > 1.
        # rew_collision = torch.sum(knee_contact, dim=1) * self.rew_scales["collision"] + torch.sum(base_contact, dim=1) * self.rew_scales["collision"]# sum vs any ?

        # stumbling penalty
        stumble = (torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) > 5.) * (torch.abs(self.contact_forces[:, self.feet_indices, 2]) < 1.)
        rew_stumble = torch.sum(stumble, dim=1) * self.rew_scales["stumble"]

        # action rate penalty
        rew_action_rate = torch.sum(torch.square(self.last_actions - self.actions), dim=1) * self.rew_scales["action_rate"]

        # default dof imitation reward
        rew_imitate = torch.exp(-5*torch.sum(torch.square(self.hound_dof_pos - self.hound_default_dof_pos), dim=1)) * 0.2


        # # air time reward
        # # contact = torch.norm(contact_forces[:, feet_indices, :], dim=2) > 1.
        # contact = self.contact_forces[:, self.feet_indices, 2] > 1. # TODO if feet z force over 1, contact true
        
        # first_contact = (self.feet_air_time > 0.) * contact # TODO if first contact, False. else, True
        # self.feet_air_time += self.dt
        # rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) * self.rew_scales["air_time"] # reward only on first contact with the ground
        # rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        # self.feet_air_time *= ~contact

        # # cosmetic penalty for hip motion
        # rew_hip = torch.sum(torch.abs(self.hound_dof_pos[:, [0, 3, 6, 9]] - self.hound_default_dof_pos[:, [0, 3, 6, 9]]), dim=1)* self.rew_scales["hip"]

        # # total reward
        # self.rew_buf = rew_lin_vel_xy + rew_ang_vel_z + rew_lin_vel_z + rew_ang_vel_xy + rew_orient + rew_base_height +\
        #             rew_torque + rew_joint_acc + rew_collision + rew_action_rate + rew_airTime + rew_hip + rew_stumble
        # self.rew_buf = torch.clip(self.rew_buf, min=0., max=None)

        # # add termination reward
        # self.rew_buf += self.rew_scales["termination"] * self.reset_buf * ~self.timeout_buf
        self.rew_buf = arm_rewards + rew_arm_joint_acc + rew_arm_dof_vel + rew_torque + rew_hound_dof_vel + rew_lin_vel + rew_ang_vel + rew_orient + rew_basepos + rew_base_height + rew_joint_acc + rew_stumble + rew_action_rate + rew_imitate
        self.rew_buf = torch.clip(self.rew_buf, min=0., max=None)
        # log episode reward sums
        # self.episode_sums["lin_vel_xy"] += rew_lin_vel_xy
        # self.episode_sums["ang_vel_z"] += rew_ang_vel_z
        # self.episode_sums["lin_vel_z"] += rew_lin_vel_z
        # self.episode_sums["ang_vel_xy"] += rew_ang_vel_xy
        self.episode_sums["terminate_check"] += check_check
        self.episode_sums["terminate"] += check
        self.episode_sums["arm_second_vel"] += velocity_rewards
        self.episode_sums["arm_joint_acc"] += rew_arm_joint_acc
        self.episode_sums["hound_joint_vel"] += rew_hound_dof_vel
        self.episode_sums["arm_joint_vel"] += rew_arm_dof_vel
        self.episode_sums["imitate_reward"] += rew_imitate
        self.episode_sums["arm_reward"] += distance_rewards
        self.episode_sums["lin_vel"] += rew_lin_vel
        self.episode_sums["ang_vel"] += rew_ang_vel
        self.episode_sums["orient"] += rew_orient
        self.episode_sums["torques"] += rew_torque
        self.episode_sums["joint_acc"] += rew_joint_acc
        # self.episode_sums["collision"] += rew_collision
        self.episode_sums["stumble"] += rew_stumble
        self.episode_sums["action_rate"] += rew_action_rate
        # self.episode_sums["air_time"] += rew_airTime
        self.episode_sums["base_position"] += rew_basepos
        self.episode_sums["base_height"] += rew_base_height
        # self.episode_sums["hip"] += rew_hip

    def reset_idx(self, env_ids):
        
        hound_positions_offset = torch_rand_float(0.5, 1.5, (len(env_ids), self.hound_num_dof), device=self.device)
        hound_velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.hound_num_dof), device=self.device)
        arm_positions_offset = torch_rand_float(0.5, 1.5, (len(env_ids), self.hound_num_dof), device=self.device)
        arm_velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.arm_num_dof), device=self.device)
        self.hound_dof_pos[env_ids] = self.hound_default_dof_pos[env_ids] * hound_positions_offset
        self.hound_dof_vel[env_ids] = hound_velocities
        self.arm_dof_pos[env_ids] = self.houndarm_default_dof_pos
        self.arm_dof_vel[env_ids] = arm_velocities
        self.dof_pos[env_ids] = torch.cat([self.hound_dof_pos[env_ids], self.arm_dof_pos[env_ids]], dim=-1) 
        self.dof_vel[env_ids] = torch.cat([self.hound_dof_vel[env_ids], self.arm_dof_vel[env_ids]], dim=-1)
        # print("dof_vel: {}".format(self.dof_vel[0,:]))
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        if self.custom_origins:
            self.update_terrain_level(env_ids)
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-0.5, 0.5, (len(env_ids), 2), device=self.device)
        else:
            self.root_states[env_ids] = self.base_init_state

        

        
        ################################################################
        ## for arm######################################################
        self.arm_commands[env_ids,0] = torch_rand_float(self.arm_command_x_range[0], self.arm_command_x_range[1], (len(env_ids), 1), device=self.device).squeeze()
        self.arm_commands[env_ids,1] = torch_rand_float(self.arm_command_y_range[0], self.arm_command_y_range[1], (len(env_ids), 1), device=self.device).squeeze()
        self.arm_commands[env_ids,2] = torch_rand_float(self.arm_command_z_range[0], self.arm_command_z_range[1], (len(env_ids), 1), device=self.device).squeeze()
        self.spherical_commands[env_ids,0] = torch_rand_float(self.sph_command_x_range[0], self.sph_command_x_range[1], (len(env_ids), 1), device=self.device).squeeze()
        self.spherical_commands[env_ids,1] = torch_rand_float(self.sph_command_y_range[0], self.sph_command_y_range[1], (len(env_ids), 1), device=self.device).squeeze()
        self.spherical_commands[env_ids,2] = torch_rand_float(self.sph_command_z_range[0], self.sph_command_z_range[1], (len(env_ids), 1), device=self.device).squeeze()
        self.spherical_commands[env_ids,3] = torch_rand_float(self.sph_command_T_range[0], self.sph_command_T_range[1], (len(env_ids), 1), device=self.device).squeeze()
        self.spherical_commands[env_ids,4] = 0.
        reset_noise = torch.rand((len(env_ids), 6), device=self.device)
        pos = tensor_clamp(
            self.houndarm_default_dof_pos.unsqueeze(0) +
            self.houndarm_dof_noise * 2.0 * (reset_noise - 0.5),
            self.houndarm_dof_lower_limits.unsqueeze(0), self.houndarm_dof_upper_limits) # TODO limit are in urdf
        self._q[env_ids, :] = pos
        self._qd[env_ids, :] = torch.zeros_like(self._qd[env_ids])
        self._pos_control[env_ids, :] = pos
        # self._effort_control[env_ids, :] = torch.zeros_like(pos)
        ################################################################
        ################################################################
        # self.commands[env_ids, 0] = torch_rand_float(self.command_x_range[0], self.command_x_range[1], (len(env_ids), 1), device=self.device).squeeze()
        # self.commands[env_ids, 1] = torch_rand_float(self.command_y_range[0], self.command_y_range[1], (len(env_ids), 1), device=self.device).squeeze()
        # self.commands[env_ids, 3] = torch_rand_float(self.command_yaw_range[0], self.command_yaw_range[1], (len(env_ids), 1), device=self.device).squeeze()
        # self.commands[env_ids] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.25).unsqueeze(1) # set small commands to zero

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        
        # sum hound pos and arm pos & effort
        total_pos = torch.cat([self.hound_dof_pos, self._pos_control], axis=1) 
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
        self.last_hound_dof_vel[env_ids] = 0.
        self.last_arm_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())

    def update_terrain_level(self, env_ids):
        if not self.init_done or not self.curriculum:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        self.terrain_levels[env_ids] -= 1 * (distance < torch.norm(self.commands[env_ids, :2])*self.max_episode_length_s*0.25)
        self.terrain_levels[env_ids] += 1 * (distance > self.terrain.env_length / 2)
        self.terrain_levels[env_ids] = torch.clip(self.terrain_levels[env_ids], 0) % self.terrain.env_rows
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]

    def push_robots(self):
        self.root_states[:, 7:9] = torch_rand_float(-1., 1., (self.num_envs, 2), device=self.device) # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
    
    ######################################################################
    ## for arm ###########################################################
    def _compute_osc_torques(self, dpose):
        # Solve for Operational Space Control # Paper: khatib.stanford.edu/publications/pdfs/Khatib_1987_RA.pdf
        # Helpful resource: studywolf.wordpress.com/2013/09/17/robot-control-4-operation-space-control/
        q, qd = self._q[:, :6], self._qd[:, :6]
        #q, qd = self._q[:, :7], self._qd[:, :7]
    
        mm_inv = torch.inverse(self._mm)
        m_eef_inv = self._j_eef @ mm_inv @ torch.transpose(self._j_eef, 1, 2)
        m_eef = torch.inverse(m_eef_inv)

        # Transform our cartesian action `dpose` into joint torques `u`
        vel = torch.cat((self.relative_eef_lin_vel, self.relative_eef_ang_vel), dim=-1)        
        # print("check relative velocity: {}".format(vel[0,:]))
        # u = torch.transpose(self._j_eef, 1, 2) @ m_eef @ (
        #         self.arm_kp * dpose - self.arm_kd * (self._eef_state[:,7:] - self._root_state[:,7:])).unsqueeze(-1)
        u = torch.transpose(self._j_eef, 1, 2) @ m_eef @ (
                self.arm_kp * dpose - self.arm_kd * (vel)).unsqueeze(-1)

        # Nullspace control torques `u_null` prevents large changes in joint configuration
        # They are added into the nullspace of OSC so that the end effector orientation remains constant
        # roboticsproceedings.org/rss07/p31.pdf
        j_eef_inv = m_eef @ self._j_eef @ mm_inv
        u_null = self.arm_kd_null * -qd + self.arm_kp_null * ((self.houndarm_default_dof_pos[:6] - q + np.pi) % (2 * np.pi) - np.pi)#((self.houndarm_default_dof_pos[:7] - q + np.pi) % (2 * np.pi) - np.pi)
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
    ######################################################################
    ######################################################################

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)     
        
        for i in range(self.decimation):
            ######################################################################
            ## for arm ###########################################################
            # Control arm (scale value first)
            u_arm = self.actions[:,12:]
            u_arm = u_arm * self.arm_cmd_limit / self.arm_action_scale
            u_arm = self._compute_osc_torques(dpose=u_arm)
            # self._arm_control[:, :] = u_arm # self._arm_control don't use
            #self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self._effort_control))
            ######################################################################
            ######################################################################
            torque_leg = torch.zeros(self.num_envs, 12, device=self.device)
            torques = torch.clip(self.Kp*(self.action_scale*self.actions[:,:12] + self.hound_default_dof_pos - self.hound_dof_pos) - self.Kd*self.hound_dof_vel,
                                 -80., 80.)
            ######################################################################
            ## for arm ###########################################################
            torques_arm = u_arm #torch.zeros(self.num_envs,6, device=self.device) 
            torques = torch.cat([torques, torques_arm], axis=1) 
            ######################################################################
            ######################################################################
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(torques))                    
            # pos_arm = torch.zeros_like(u_arm)
            # position_input = torch.cat([self.hound_default_dof_pos,pos_arm], axis=1) 
            # self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(position_input))
            # self.torques = torques[:,0:12] # FIX
            self.torques = torques.view(self.torques.shape)
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
        # self.common_step_counter += 1
        # if self.common_step_counter % self.push_interval == 0:
        #     self.push_robots()

        # prepare quantities
        self.base_quat = self.root_states[:, 3:7]
        # self.relative_commands_pos = quat_rotate_inverse(self.base_quat, self.arm_commands[:, :3] - self._root_state[:, :3])
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.eef_lin_vel = quat_rotate_inverse(self.base_quat, self._eef_state[:, 7:10])
        self.eef_ang_vel = quat_rotate_inverse(self.base_quat, self._eef_state[:, 10:13])
        self.relative_eef_lin_vel = self.eef_lin_vel - self.base_lin_vel
        self.relative_eef_ang_vel = self.eef_ang_vel - self.base_lin_vel

        self.relative_eef_pos = quat_rotate_inverse(self.base_quat, self._eef_state[:,:3] - self._root_state[:,:3])
        self.relative_eef_quat = quat_mul(self.base_quat, quat_conjugate(self._eef_state[:,3:7]))
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        forward = quat_apply(self.base_quat, self.forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0])
        # self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
        
        self.last_actions[:] = self.actions[:]
        self.last_hound_dof_vel[:] = self.hound_dof_vel[:]
        self.last_arm_dof_vel[:] = self.arm_dof_vel[:]  
        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            # draw height lines
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
            for i in range(self.num_envs):
                base_pos = (self.root_states[i, :3]).cpu().numpy()
                heights = self.measured_heights[i].cpu().numpy()
                height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
                for j in range(heights.shape[0]):
                    x = height_points[j, 0] + base_pos[0]
                    y = height_points[j, 1] + base_pos[1]
                    z = heights[j]
                    sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                    gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 

    def init_height_points(self):
        # 1mx1.6m rectangle (without center line)
        y = 0.1 * torch.tensor([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5], device=self.device, requires_grad=False) # 10-50cm on each side
        x = 0.1 * torch.tensor([-8, -7, -6, -5, -4, -3, -2, 2, 3, 4, 5, 6, 7, 8], device=self.device, requires_grad=False) # 20-80cm on each side
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def get_heights(self, env_ids=None):
        if self.cfg["env"]["terrain"]["terrainType"] == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg["env"]["terrain"]["terrainType"] == 'none':
            raise NameError("Can't measure height with terrain type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)
 
        points += self.terrain.border_size
        points = (points/self.terrain.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]

        heights2 = self.height_samples[px+1, py+1]
        heights = torch.min(heights1, heights2)

        return heights.view(self.num_envs, -1) * self.terrain.vertical_scale


# terrain generator
from isaacgym.terrain_utils import *
class Terrain:
    def __init__(self, cfg, num_robots) -> None:

        self.type = cfg["terrainType"]
        if self.type in ["none", 'plane']:
            return
        self.horizontal_scale = 0.1
        self.vertical_scale = 0.005
        self.border_size = 20
        self.num_per_env = 2
        self.env_length = cfg["mapLength"]
        self.env_width = cfg["mapWidth"]
        self.proportions = [np.sum(cfg["terrainProportions"][:i+1]) for i in range(len(cfg["terrainProportions"]))]

        self.env_rows = cfg["numLevels"]
        self.env_cols = cfg["numTerrains"]
        self.num_maps = self.env_rows * self.env_cols
        self.num_per_env = int(num_robots / self.num_maps)
        self.env_origins = np.zeros((self.env_rows, self.env_cols, 3))

        self.width_per_env_pixels = int(self.env_width / self.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / self.horizontal_scale)

        self.border = int(self.border_size/self.horizontal_scale)
        self.tot_cols = int(self.env_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(self.env_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)
        if cfg["curriculum"]: # TODO True
            self.curiculum(num_robots, num_terrains=self.env_cols, num_levels=self.env_rows)
        else:
            self.randomized_terrain()   
        self.heightsamples = self.height_field_raw
        self.vertices, self.triangles = convert_heightfield_to_trimesh(self.height_field_raw, self.horizontal_scale, self.vertical_scale, cfg["slopeTreshold"])
    
    def randomized_terrain(self):
        for k in range(self.num_maps):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.env_rows, self.env_cols))

            # Heightfield coordinate system from now on
            start_x = self.border + i * self.length_per_env_pixels
            end_x = self.border + (i + 1) * self.length_per_env_pixels
            start_y = self.border + j * self.width_per_env_pixels
            end_y = self.border + (j + 1) * self.width_per_env_pixels

            terrain = SubTerrain("terrain",
                              width=self.width_per_env_pixels,
                              length=self.width_per_env_pixels,
                              vertical_scale=self.vertical_scale,
                              horizontal_scale=self.horizontal_scale)
            choice = np.random.uniform(0, 1)
            if choice < 0.1:
                if np.random.choice([0, 1]):
                    pyramid_sloped_terrain(terrain, np.random.choice([-0.3, -0.2, 0, 0.2, 0.3]))
                    random_uniform_terrain(terrain, min_height=-0.1, max_height=0.1, step=0.05, downsampled_scale=0.2)
                else:
                    pyramid_sloped_terrain(terrain, np.random.choice([-0.3, -0.2, 0, 0.2, 0.3]))
            elif choice < 0.6:
                # step_height = np.random.choice([-0.18, -0.15, -0.1, -0.05, 0.05, 0.1, 0.15, 0.18])
                step_height = np.random.choice([-0.15, 0.15])
                pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
            elif choice < 1.:
                discrete_obstacles_terrain(terrain, 0.15, 1., 2., 40, platform_size=3.)

            self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

            env_origin_x = (i + 0.5) * self.env_length
            env_origin_y = (j + 0.5) * self.env_width
            x1 = int((self.env_length/2. - 1) / self.horizontal_scale)
            x2 = int((self.env_length/2. + 1) / self.horizontal_scale)
            y1 = int((self.env_width/2. - 1) / self.horizontal_scale)
            y2 = int((self.env_width/2. + 1) / self.horizontal_scale)
            env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*self.vertical_scale
            self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

    def curiculum(self, num_robots, num_terrains, num_levels):
        num_robots_per_map = int(num_robots / num_terrains)
        left_over = num_robots % num_terrains
        idx = 0
        for j in range(num_terrains):
            for i in range(num_levels):
                terrain = SubTerrain("terrain",
                                    width=self.width_per_env_pixels,
                                    length=self.width_per_env_pixels,
                                    vertical_scale=self.vertical_scale,
                                    horizontal_scale=self.horizontal_scale)
                difficulty = i / num_levels
                choice = j / num_terrains

                slope = difficulty * 0.4
                step_height = 0.05 + 0.175 * difficulty
                discrete_obstacles_height = 0.025 + difficulty * 0.15
                stepping_stones_size = 2 - 1.8 * difficulty
                if choice < self.proportions[0]:
                    if choice < 0.05:
                        slope *= -1
                    pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
                elif choice < self.proportions[1]:
                    if choice < 0.15:
                        slope *= -1
                    pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
                    random_uniform_terrain(terrain, min_height=-0.1, max_height=0.1, step=0.025, downsampled_scale=0.2)
                elif choice < self.proportions[3]:
                    if choice<self.proportions[2]:
                        step_height *= -1
                    pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
                elif choice < self.proportions[4]:
                    discrete_obstacles_terrain(terrain, discrete_obstacles_height, 1., 2., 40, platform_size=3.)
                else:
                    stepping_stones_terrain(terrain, stone_size=stepping_stones_size, stone_distance=0.1, max_height=0., platform_size=3.)

                # Heightfield coordinate system
                start_x = self.border + i * self.length_per_env_pixels
                end_x = self.border + (i + 1) * self.length_per_env_pixels
                start_y = self.border + j * self.width_per_env_pixels
                end_y = self.border + (j + 1) * self.width_per_env_pixels
                self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

                robots_in_map = num_robots_per_map
                if j < left_over:
                    robots_in_map +=1

                env_origin_x = (i + 0.5) * self.env_length
                env_origin_y = (j + 0.5) * self.env_width
                x1 = int((self.env_length/2. - 1) / self.horizontal_scale)
                x2 = int((self.env_length/2. + 1) / self.horizontal_scale)
                y1 = int((self.env_width/2. - 1) / self.horizontal_scale)
                y2 = int((self.env_width/2. + 1) / self.horizontal_scale)
                env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*self.vertical_scale
                self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]


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