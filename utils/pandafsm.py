# Copyright (c) 2020 NVIDIA Corporation

# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
# OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.
"""Class to represent grasp process as finite state machine."""

import copy

import numpy as np
from isaacgym import gymapi
from isaacgym import gymtorch
from scipy.signal import butter, filtfilt

from utils import panda_fk
from utils import tet_based_metrics


NUM_JOINTS = 16
GRIPPER_TIP_Z_OFFSET = 0.112
GRIPPER_TIP_Y_OFFSET = 0.00444222
DEBUG = False


def butter_lowpass_filter(data):
    """Low-pass filter the dynamics data."""
    fs = 20
    cutoff = 0.05
    nyq = 0.5 * fs
    order = 1
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


class PandaFsm:
    """FSM for control of Panda hand for the grasp tests."""

    def __init__(self, gym_handle, sim_handle, env_handles, franka_handle,
                 platform_handle, state, object_cof, f_errs,
                 grasp_transform, obj_name, env_id, hand_origin, viewer,
                 envs_per_row, env_dim, youngs, density, directions, mode):
        """Initialize attributes of grasp evaluation FSM.

        Args:
            gym_handle (gymapi.Gym): Gym object.
            sim_handle (gymapi.Sim): Simulation object.
            env_handles (list of gymapi.Env): List of all environments.
            franka_handle (int): Handle of Franka panda hand actor.
            platform_handle (int): Handle of support plane actor.
            state (str): Name of initial FSM state.
            object_cof (float): Coefficient of friction.
            f_errs (np.ndarray): Array to track most recent window of contact force errors.
            grasp_transform (isaacgym.gymapi.Transform): Initial pose of Franka panda hand.
            obj_name (str): Name of object to be grasped.
            env_id (int): Index of environment from env_handles.
            hand_origin (gymapi.Transform): Pose of the hand origin (at its base).
            viewer (gymapi.Viewer): Graphical display object.
            envs_per_row (int): Number of environments to be placed in a row.
            env_dim (float): Size of each environment.
            youngs (str): Elastic modulus of the object, eg '3e5'.
            density (str): Density of the object, eg. '1000'.
            directions (np.ndarray): Array of directions to be evaluated in this env.
            mode (str): Name of grasp test {'pickup', 'reorient', 'shake', 'twist'}.
        """
        self.mode = mode
        self.started = False
        self.state = state

        # Simulation handles
        self.gym_handle = gym_handle
        self.sim_handle = sim_handle
        self.env_handles = env_handles
        self.env_id = env_id
        self.env_handle = self.env_handles[self.env_id]
        self.viewer = viewer

        # Sim params
        self.sim_params = gymapi.SimParams()
        self.sim_params = self.gym_handle.get_sim_params(self.sim_handle)
        self.envs_per_row = envs_per_row
        self.env_dim = env_dim
        self.env_x_offset = 2. * self.env_dim * (self.env_id %
                                                 self.envs_per_row)
        self.env_z_offset = 2. * self.env_dim * int(
            self.env_id / self.envs_per_row)

        # Actors
        self.franka_handle = franka_handle
        self.platform_handle = platform_handle
        num_franka_bodies = self.gym_handle.get_actor_rigid_body_count(
            self.env_handle, self.franka_handle)
        num_platform_bodies = self.gym_handle.get_actor_rigid_body_count(
            self.env_handle, self.platform_handle)
        total_num_bodies = num_franka_bodies + num_platform_bodies
        self.finger_indices = [
            total_num_bodies * self.env_id + num_franka_bodies - 2,
            total_num_bodies * self.env_id + num_franka_bodies - 1
        ]
        self.hand_indices = range(
            total_num_bodies * self.env_id,
            total_num_bodies * self.env_id + num_franka_bodies)
        self.platform_indices = [
            total_num_bodies * self.env_id + num_franka_bodies + 1
        ]

        # Object material and mesh values
        self.obj_name = obj_name
        self.object_cof = object_cof
        self.particle_state_tensor = gymtorch.wrap_tensor(
            self.gym_handle.acquire_particle_state_tensor(self.sim_handle))
        self.previous_particle_state_tensor = None
        self.state_tensor_length = 0
        self.youngs = float(youngs)
        self.density = float(density)

        # Contacts and force control
        self.contacts = np.array([])
        self.particles_contacting_gripper = np.zeros(2)
        self.FOS = 1 + np.log10(self.youngs) / 10.
        self.initial_desired_force = 0.0
        self.corrected_desired_force = 0.0
        self.F_history = []
        self.F_max_window_size = 300
        self.filtered_forces = []
        self.moving_average = []
        self.f_errs = f_errs

        # Gripper positions
        self.gripper_positions_under_gravity = np.zeros(2)
        self.squeeze_min_gripper_width = 0.0

        # 16 vector directions for reorient and acceleration tests
        self.directions = np.column_stack(
            (directions[:, 1], directions[:, 2], directions[:, 0]))
        self.direction = self.directions[0]

        # Linear and angular acceleration testing
        self.shake_vel, self.rot_vel, self.travel_speed = 0.0, 0.0, 0.0

        # Franka Panda hand kinematics
        self.grasp_transform = grasp_transform
        self.franka_dof_states = None
        self.hand_origin = hand_origin
        self.mid_finger_origin = np.array([
            self.hand_origin.p.x, self.hand_origin.p.y,
            self.hand_origin.p.z + GRIPPER_TIP_Z_OFFSET, 1
        ])
        self.mid_finger_position_transformed = np.zeros(3)
        self.left_finger_position = np.array([
            self.hand_origin.p.x, self.hand_origin.p.y + GRIPPER_TIP_Y_OFFSET,
            self.hand_origin.p.z + GRIPPER_TIP_Z_OFFSET, 1
        ])
        self.right_finger_position = np.array([
            self.hand_origin.p.x, self.hand_origin.p.y - GRIPPER_TIP_Y_OFFSET,
            self.hand_origin.p.z + GRIPPER_TIP_Z_OFFSET, 1
        ])
        self.mid_finger_position = np.array([
            self.hand_origin.p.x, self.hand_origin.p.y,
            self.hand_origin.p.z + GRIPPER_TIP_Z_OFFSET, 1
        ])
        self.left_normal = self.grasp_transform.transform_vector(
            gymapi.Vec3(1.0, 0., 0.))
        self.right_normal = self.grasp_transform.transform_vector(
            gymapi.Vec3(-1.0, 0., 0.))

        # Franka Panda hand control outputs
        self.vel_des = np.zeros(NUM_JOINTS)
        self.pos_des = np.zeros(NUM_JOINTS)
        self.torque_des = np.zeros(NUM_JOINTS)
        self.running_torque = [0, -0.1, -0.1]

        # FSM: Close state
        self.close_fails = 0
        self.left_has_contacted = False
        self.right_has_contacted = False
        self.franka_positions_at_contact = np.zeros(NUM_JOINTS)
        self.desired_closing_gripper_pos = [0.0, 0.0]
        self.grippers_pre_squeeze = [-1, -1]

        # FSM: Squeeze state
        self.squeeze_counter = 0
        self.squeeze_lost_contact_counter = 0
        self.squeeze_intensity = 0
        self.squeezing_close_fails = 0
        self.squeezing_no_grasp = 0
        self.squeezed_until_force = False

        # FSM: Hang state
        self.reached_hang = False
        self.hang_stresses = []
        self.hang_separations = []

        # FSM: Pickup
        self.inferred_rot_force = False

        # FSM: Reorient
        self.reorient_stress_history = []

        # FSM: Accelerations
        self.reached_twist_location = False
        self.shake_counter, self.twist_counter, self.twist_travel_counter = 0, 0, 0
        self.reached_shake_speed = False
        self.reached_twist_speed = False

        # Success flags
        self.pickup_success = False
        self.timed_out = False

        # Counters
        self.full_counter = 0
        self.inferred_rot_force_counter = 0
        self.hang_counter = 0
        self.reorient_counter = 0
        self.open_counter = 0
        self.close_soft_counter = 0

    def init_metrics_and_features(self):
        """Initialize attributes to store metrics and features."""
        # Tet structure
        num_nodes = self.state_tensor_length
        (tet_particles,
         tet_stresses) = self.gym_handle.get_sim_tetrahedra(self.sim_handle)
        num_envs = self.gym_handle.get_env_count(self.sim_handle)
        num_tets_per_env = int(len(tet_stresses) / num_envs)
        self.undeformed_mesh = np.zeros((num_nodes, 3))

        # Pre contact
        self.pre_contact_stresses = np.zeros((num_envs, num_tets_per_env))
        self.pre_contact_se = 0.0

        # Metrics at target squeeze force
        self.stresses_at_force = np.zeros(num_tets_per_env)
        self.se_at_force = 0.0
        self.positions_at_force = np.zeros((num_nodes, 3))
        self.gripper_force_at_force = 0.0
        self.gripper_distance_at_force = 0.0

        # Metrics after pickup
        self.stresses_under_gravity = np.zeros(num_tets_per_env)
        self.se_under_gravity = 0.0
        self.positions_under_gravity = np.zeros((num_nodes, 3))
        self.gripper_force_under_gravity = 0.0

        # Reorientation metrics
        self.reorientation_meshes = np.zeros((4, num_nodes, 3))
        self.reorientation_stresses = np.zeros((4, num_tets_per_env))

        # Linear and angular accelerations at fail
        self.shake_fail_acc, self.twist_fail_acc = 0, 0

        # Geometry metrics
        self.pure_distances = np.zeros(2)
        self.perp_distances = np.zeros(2)
        self.edge_distances = np.zeros(2)
        self.num_gripper_contacts = np.zeros(2)

    def get_force_based_velocity(self, F_des, F_curr):
        """Get desired torques for force-based sequences of grasping process."""
        total_F_curr = self.moving_average[
            -1]  # Use the smoothed value instead of raw readings
        if np.sum(F_curr[1:]) == 0.0 and np.all(
                self.particles_contacting_gripper == 0):
            total_F_curr = 0

        total_F_err = np.sum(F_des[1:]) - total_F_curr

        # Compute error values for state transitions
        F_curr_mag = (np.abs(F_curr[1]) + np.abs(F_curr[2])) / 2.0

        S = 500
        self.running_torque[1] -= min(total_F_err / S, 3 / S)
        self.running_torque[2] -= min(total_F_err / S, 3 / S)
        self.running_torque[1] = min(-0.05, self.running_torque[1])
        self.running_torque[2] = min(-0.05, self.running_torque[2])

        torque_des = self.running_torque
        if DEBUG:
            print(torque_des[-2:], total_F_curr, self.desired_force)

        return torque_des, F_curr_mag, total_F_err

    def get_grasp_F_curr(self, body_index, debug=False):
        """Get current forces acting on fingerpads, as sum of nodal forces."""
        net_hor_force_left = 0.0
        net_hor_force_right = 0.0
        left_contacts = []
        left_force_mags = []
        left_barys = []
        right_contacts = []
        for contact in self.contacts:
            curr_body_index = contact[4]
            # If the rigid body (identified by body_index) is in contact
            if curr_body_index in body_index:
                curr_force_dir = contact[6].view(
                    (np.float32, len(
                        contact[6].dtype.names)))  # Cast from structured array
                curr_force_mag = contact[7]
                normal_to_gripper = self.grasp_transform.transform_vector(
                    gymapi.Vec3(1., 0., 0.))

                if True:
                    normal_component = curr_force_mag * normal_to_gripper.dot(
                        gymapi.Vec3(curr_force_dir[0], curr_force_dir[1],
                                    curr_force_dir[2]))
                    if curr_body_index == body_index[0]:
                        net_hor_force_left += np.abs(normal_component)
                        left_contacts.append(contact[2])
                        left_force_mags.append(curr_force_mag)
                        left_barys.append(contact[3])
                    elif curr_body_index == body_index[1]:
                        net_hor_force_right += np.abs(normal_component)
                        right_contacts.append(contact[2])

        F_curr = np.array([-1.0, net_hor_force_left,
                           net_hor_force_right])  # -1 denotes unused
        F_curr_dot = 0.0
        return F_curr, F_curr_dot

    def get_node_indices_contacting_body(self, body_name):
        """Get indices of the mesh nodes that are contacting any part of the hand."""
        all_contact_indices = []
        for contact in self.contacts:
            in_same_environment = contact[2][0] in range(
                self.env_id * self.state_tensor_length,
                (self.env_id + 1) * self.state_tensor_length)

            if body_name == "hand":
                contact_with_body = contact[4] in self.hand_indices
            elif body_name == "platform":
                contact_with_body = contact[4] in self.platform_indices
            if in_same_environment and contact_with_body:
                all_contact_indices.append(contact[2][0])

        return all_contact_indices

    def object_contacting_platform(self):
        """Return True if the soft object is in contact with the platform."""
        for contact in self.contacts:
            if contact[4] in self.platform_indices:
                return True
        return False

    def get_node_indices_contacting_fingers(self):
        """Get indices of the mesh nodes that are contacting the fingers."""
        all_contact_indices = []
        left_contact_indices = []
        right_contact_indices = []
        for contact in self.contacts:
            in_same_environment = contact[2][0] in range(
                self.env_id * self.state_tensor_length,
                (self.env_id + 1) * self.state_tensor_length)
            contact_with_fingers = contact[4] in self.finger_indices
            if in_same_environment and contact_with_fingers:
                all_contact_indices.append(contact[2][0])
                if contact[4] == self.finger_indices[0]:
                    left_contact_indices.append(contact[2][0])
                else:
                    right_contact_indices.append(contact[2][0])
        return left_contact_indices, right_contact_indices, all_contact_indices

    def get_contact_geometry_features(self):
        """Calculate features based on object contact geometry."""
        _, _, _, object_centroid = tet_based_metrics.get_tet_based_metrics(
            self.gym_handle, self.sim_handle, self.env_handles, self.env_id,
            self.particle_state_tensor, self.youngs)
        self.object_centroid = object_centroid
        left_indices, right_indices, finger_indices = self.get_node_indices_contacting_fingers(
        )

        left_contact_locations = np.copy(
            self.particle_state_tensor.numpy()[left_indices, :3])
        right_contact_locations = np.copy(
            self.particle_state_tensor.numpy()[right_indices, :3])

        left_contact_centroid = left_contact_locations.mean(axis=0)
        right_contact_centroid = right_contact_locations.mean(axis=0)

        gripper_normal = np.array(
            [self.left_normal.x, self.left_normal.y, self.left_normal.z])

        self.pure_distances = np.array([
            np.linalg.norm(left_contact_centroid - object_centroid),
            np.linalg.norm(right_contact_centroid - object_centroid)
        ])

        # Get perpendicular distances
        left_perp_vec = (object_centroid - left_contact_centroid) - np.dot(
            (object_centroid - left_contact_centroid),
            gripper_normal) * gripper_normal
        right_perp_vec = (object_centroid - right_contact_centroid) - np.dot(
            (object_centroid - right_contact_centroid),
            gripper_normal) * gripper_normal

        self.perp_distances = np.array(
            [np.linalg.norm(left_perp_vec),
             np.linalg.norm(right_perp_vec)])
        self.num_gripper_contacts = np.array(
            [len(left_indices), len(right_indices)])

        # Get distance to edges of fingers
        left_fk_map = panda_fk.get_fk(self.franka_dof_states['pos'],
                                      self.hand_origin,
                                      mode="left")
        right_fk_map = panda_fk.get_fk(self.franka_dof_states['pos'],
                                       self.hand_origin,
                                       mode="right")

        left_finger_position = left_fk_map.dot(self.left_finger_position)[:3]
        right_finger_position = right_fk_map.dot(
            self.right_finger_position)[:3]

        edge_vector = left_fk_map.dot(np.array([1, 0, 0, 0]))[:3]

        left_edge_perp_vec = (
            left_contact_centroid - left_finger_position) - np.dot(
                (left_contact_centroid - left_finger_position),
                edge_vector) * edge_vector
        right_edge_perp_vec = (
            right_contact_centroid - right_finger_position) - np.dot(
                (right_contact_centroid - right_finger_position),
                edge_vector) * edge_vector

        self.edge_distances = np.array([
            np.linalg.norm(left_edge_perp_vec),
            np.linalg.norm(right_edge_perp_vec)
        ])

    def infer_rot_force(self, F_curr):
        """Calculate required squeezing force to counteract rotational slip."""
        _, _, _, object_centroid = tet_based_metrics.get_tet_based_metrics(
            self.gym_handle, self.sim_handle, self.env_handles, self.env_id,
            self.particle_state_tensor, self.youngs)
        _, _, finger_indices = self.get_node_indices_contacting_fingers()

        # Get centroid
        contact_locations = np.copy(
            self.particle_state_tensor.numpy()[finger_indices, :3])
        contact_centroid = contact_locations.mean(axis=0)

        # Get projected COF onto gripper side plane
        side_plane_normal = np.array(
            [self.left_normal.x, self.left_normal.y, self.left_normal.z])
        COF_proj = object_centroid - np.dot(
            side_plane_normal,
            (object_centroid - contact_centroid)) * side_plane_normal

        # Choose the moment axis to be between the two points of moment
        line_to_COF = (COF_proj - contact_centroid
                       ) / np.linalg.norm(COF_proj - contact_centroid)

        # Find the closest and farthest ends of the gripper to contact centroids
        comps = []
        for p in range(contact_locations.shape[0]):
            contact_point = contact_locations[p, :]
            comp = np.dot((contact_point - contact_centroid), line_to_COF)
            comps.append(comp)
        min_point = np.min(
            comps
        ) * line_to_COF + contact_centroid
        max_point = np.max(
            comps
        ) * line_to_COF + contact_centroid

        # Make the min point halfway between what it actually is to the center of the gripper
        min_point = 0.5 * (min_point + contact_centroid)
        max_point = 0.5 * (max_point + contact_centroid)

        len_R = np.linalg.norm(COF_proj - max_point)
        len_r = np.linalg.norm(max_point - min_point)
        F2 = len_R * self.mg / len_r
        F1 = self.mg + F2

        print("Desired gripper force to overcome rot",
              self.FOS * (F1 + F2) / self.object_cof)
        print("Current desired force without FOS", self.mg / self.object_cof)

        return min(self.FOS * (F1 + F2) / self.object_cof,
                   10 * self.desired_force)

    def particles_between_gripper(self):
        """Return number of contacts with grippers and number of nodes between grippers."""
        franka_dof_states = self.franka_dof_states
        fk_map_left = panda_fk.get_fk(franka_dof_states['pos'],
                                      self.hand_origin,
                                      mode="left")
        fk_map_right = panda_fk.get_fk(franka_dof_states['pos'],
                                       self.hand_origin,
                                       mode="right")

        left_finger_position = np.array([
            self.hand_origin.p.x,
            self.hand_origin.p.y + GRIPPER_TIP_Y_OFFSET,
            self.hand_origin.p.z + GRIPPER_TIP_Z_OFFSET,
            1
        ])
        new_left_finger_position = fk_map_left.dot(left_finger_position)

        right_finger_position = np.array([
            self.hand_origin.p.x, self.hand_origin.p.y - GRIPPER_TIP_Y_OFFSET,
            self.hand_origin.p.z + GRIPPER_TIP_Z_OFFSET, 1
        ])
        new_right_finger_position = fk_map_right.dot(right_finger_position)

        left_normal = self.grasp_transform.transform_vector(
            gymapi.Vec3(0, -0.03, 0.))
        right_normal = self.grasp_transform.transform_vector(
            gymapi.Vec3(0, 0.03, 0.))

        # Check if the object is in contact with the gripper
        num_contacts_with_finger = np.zeros(len(self.finger_indices))
        for contact in self.contacts:
            curr_body_index = contact[4]
            for i in range(len(self.finger_indices)):
                if curr_body_index == self.finger_indices[i]:
                    num_contacts_with_finger[i] += 1

        # Check if the object is in between the gripper
        state_tensor = self.particle_state_tensor.numpy()[
            self.env_id * self.state_tensor_length:(self.env_id + 1)
            * self.state_tensor_length, :]
        num_nodes_between_fingers = 0
        for n in range(state_tensor.shape[0]):
            pos = gymapi.Vec3(state_tensor[n][0].item(),
                              state_tensor[n][1].item(),
                              state_tensor[n][2].item())

            pos.x -= self.env_x_offset
            pos.z -= self.env_z_offset

            left_project = pos - gymapi.Vec3(new_left_finger_position[0],
                                             new_left_finger_position[1],
                                             new_left_finger_position[2])
            right_project = pos - gymapi.Vec3(new_right_finger_position[0],
                                              new_right_finger_position[1],
                                              new_right_finger_position[2])
            if left_normal.dot(left_project) >= 0 and right_normal.dot(
                    right_project) >= 0:
                num_nodes_between_fingers += 1

        return num_contacts_with_finger, num_nodes_between_fingers

    def update_previous_particle_state_tensor(self):
        """Save copy of previous timestep's particle state tensor."""
        if self.particle_state_tensor is not None:
            self.previous_particle_state_tensor = np.copy(
                self.particle_state_tensor.numpy())

    def save_full_state(self):
        """Save current state."""
        self.saved_platform_state = np.copy(
            self.gym_handle.get_actor_rigid_body_states(
                self.env_handle, self.platform_handle, gymapi.STATE_ALL))
        self.saved_franka_state = np.copy(
            self.gym_handle.get_actor_rigid_body_states(
                self.env_handle, self.franka_handle, gymapi.STATE_ALL))
        self.saved_object_state = copy.deepcopy(self.particle_state_tensor)
        self.saved_fsm_state = self.state

    def reset_saved_state(self):
        """Revert to previously saved state."""
        self.gym_handle.set_actor_rigid_body_states(self.env_handle,
                                                    self.platform_handle,
                                                    self.saved_platform_state,
                                                    gymapi.STATE_ALL)
        self.gym_handle.set_actor_rigid_body_states(self.env_handle,
                                                    self.franka_handle,
                                                    self.saved_franka_state,
                                                    gymapi.STATE_ALL)

        self.gym_handle.set_particle_state_tensor(
            self.sim_handle, gymtorch.unwrap_tensor(self.saved_object_state))
        print("Reverting back to state", self.saved_fsm_state)
        self.inferred_rot_force_counter
        self.state = self.saved_fsm_state
        self.squeeze_counter = 0

    def run_state_machine(self):
        """Run state machine for running grasp tests."""
        self.full_counter += 1

        # Get hand states, soft contacts
        self.franka_dof_states = self.gym_handle.get_actor_dof_states(
            self.env_handle, self.franka_handle, gymapi.STATE_ALL)
        if self.started:
            self.contacts = self.gym_handle.get_soft_contacts(self.sim_handle)

        # Process finger grasp forces with LP filter and moving average
        F_curr, F_curr_dot = self.get_grasp_F_curr(self.finger_indices)
        self.F_history.append(np.sum(F_curr[1:]))
        window = self.F_history[-self.F_max_window_size:]
        filtered_force, avg_of_filter = 0.0, 0.0
        if len(window) > 10:
            filtered_force = butter_lowpass_filter(window)[-1]
        self.filtered_forces.append(filtered_force)
        if len(self.F_history) > 0:
            avg_of_filter = np.mean(self.filtered_forces[-30:])
        self.moving_average.append(avg_of_filter)

        # Get num_contacts, gripper_separation
        try:
            particles_contacting_gripper, _ = self.particles_between_gripper()
            self.particles_contacting_gripper = particles_contacting_gripper
        except BaseException:
            particles_contacting_gripper = self.particles_contacting_gripper

        curr_separation = np.sum([
            self.franka_dof_states['pos'][-3:][1],
            self.franka_dof_states['pos'][-3:][2]
        ])

        ############################################################################
        # OPEN STATE: Hand is initialized in a state where the fingers are open
        ############################################################################
        if self.state == 'open':
            self.started = True
            self.open_counter += 1

            # Get length of state tensor
            if self.particle_state_tensor is not None and self.state_tensor_length == 0:
                self.state_tensor_length = int(
                    self.particle_state_tensor.shape[0]
                    / self.gym_handle.get_env_count(self.sim_handle))
                print("State tensor length", self.state_tensor_length)

            self.init_metrics_and_features()

            # Save current untouched mesh
            self.undeformed_mesh = np.copy(
                self.particle_state_tensor.numpy()
                [self.env_id * self.state_tensor_length:(self.env_id + 1)
                    * self.state_tensor_length, :][:, :3])

            # Get pre-contact stresses, SE, and weight
            self.pre_contact_stresses, self.pre_contact_se, \
                object_volume, _ = tet_based_metrics.get_tet_based_metrics(
                    self.gym_handle, self.sim_handle, self.env_handles,
                    self.env_id, self.particle_state_tensor, self.youngs)
            self.pre_contact_stresses = self.pre_contact_stresses[
                self.env_id]
            self.mg = 9.81 * object_volume * self.density
            self.desired_force = self.FOS * 9.81 \
                * object_volume * self.density / self.object_cof
            self.initial_desired_force = self.desired_force

            # If hand starts in contact with object, end test
            if len(self.get_node_indices_contacting_body("hand")) > 0:
                print(self.env_id, "in collision")
                self.state = 'done'

            # Save state, then transition to close
            self.save_full_state()

            # Transition when mesh stresses aren't zero (takes a couple of iterations)
            if not np.all(self.pre_contact_stresses == 0):
                self.state = "close"

        ############################################################################
        # CLOSE STATE: Fingers close rapidly until contact with object is made
        ############################################################################
        if self.state == 'close':

            closing_speeds = np.zeros(NUM_JOINTS)
            closing_speeds[-2:] = np.array([-0.5, -0.5])
            self.vel_des = np.copy(closing_speeds)

            # Close fingers only until initial contact is made
            if np.abs(F_curr[1]) > 0.0:
                self.desired_closing_gripper_pos[0] = self.franka_dof_states[
                    'pos'][-3:][1]
                self.vel_des[-3:][1] = 0
                self.left_has_contacted = True

            if np.abs(F_curr[2]) > 0.0:
                self.desired_closing_gripper_pos[1] = self.franka_dof_states[
                    'pos'][-3:][2]
                self.vel_des[-3:][2] = 0
                self.right_has_contacted = True

            if (self.left_has_contacted and self.right_has_contacted):
                self.vel_des = np.copy(closing_speeds)

            in_contact = np.abs(F_curr[1]) > 0.005 and np.abs(
                F_curr[2]) > 0.005

            # Catch failure case where object is not between fingers
            if not in_contact and (
                    self.franka_dof_states['pos'][-3:][1] < 0.0001
                    and self.franka_dof_states['pos'][-3:][2] < 0.0001):
                print(self.env_id,
                      "Failed: Grippers closed without contacting object.")
                self.state = 'done'

            # Check for state transition
            if in_contact:
                self.franka_positions_at_contact = np.copy(
                    self.franka_dof_states['pos'])
                self.reset_saved_state()
                self.state = 'start_closer'

        ############################################################################
        # START_CLOSER STATE: Reset back to open position, fingers close rapidly until
        # close to making contact with the object.
        ############################################################################
        elif self.state == 'start_closer':
            """Close grippers until grippers are close to contacting object."""

            if self.franka_dof_states['pos'][
                    -2] < self.franka_positions_at_contact[-2] + 0.007:
                self.vel_des[-2] = 0.0

            if self.franka_dof_states['pos'][
                    -1] < self.franka_positions_at_contact[-1] + 0.007:
                self.vel_des[-1] = 0.0

            if np.all(self.franka_dof_states['pos'][-2:]
                      < self.franka_positions_at_contact[-2:] + 0.009):
                self.state = 'close_soft'
                self.save_full_state()

        ############################################################################
        # CLOSE_SOFT STATE: Fingers close slowly until contact with object is made
        ############################################################################
        if self.state == 'close_soft':

            # Closing speed decreases every time initial contact forces are too high
            first_speed = 0.25 / np.log10(self.youngs)
            closing_speeds = np.zeros(NUM_JOINTS)
            closing_speeds[-2:] = np.array([
                -first_speed / (self.close_fails + 1),
                -first_speed / (self.close_fails + 1)
            ])
            self.vel_des = np.copy(closing_speeds)

            if np.sum(F_curr[1:]) > 300 * np.log10(self.youngs) / 4:
                self.close_fails += 1
                self.left_has_contacted, self.right_has_contacted = False, False
                print("Forces too high during close_soft, resetting state",
                      np.sum(F_curr[1:]))
                self.reset_saved_state()

            in_contact = np.abs(F_curr[1]) > 0.005 and np.abs(
                F_curr[2]) > 0.005

            if in_contact:
                self.grippers_pre_squeeze = [
                    self.franka_dof_states['pos'][-3:][1],
                    self.franka_dof_states['pos'][-3:][2]
                ]
                self.state = "squeeze"

        ############################################################################
        # SQUEEZE STATE: Fingers squeeze until desired force is applied
        ############################################################################
        if self.state == "squeeze":
            self.squeeze_counter += 1

            if self.squeeze_counter > 300:
                self.squeeze_intensity += 1
                self.squeeze_counter = 0

            # Torque controller to achieved desired squeezing force
            F_des = np.array(
                [0.0, self.desired_force / 2.0, self.desired_force / 2.0])

            torque_des_force, F_curr_mag, F_err = self.get_force_based_velocity(
                F_des, F_curr)
            self.torque_des[-2:] = torque_des_force[1:]
            self.f_errs = np.hstack((self.f_errs[1:], F_err))

            # Increase squeezing force to counteract rotational slip
            if self.mode == "reorient" and np.all(np.abs(self.f_errs) < 0.3 * self.desired_force) \
                    and not self.squeezed_until_force and not self.inferred_rot_force \
                    and self.squeeze_counter > 30:
                req_rot_force = self.infer_rot_force(F_curr)
                if req_rot_force > self.desired_force:
                    print("Change desired force from %s to %s" %
                          (self.desired_force, req_rot_force))
                    self.desired_force = req_rot_force
                self.corrected_desired_force = self.desired_force
                self.inferred_rot_force = True

            # DETECT FAILURE CASES DURING SQUEEZING

            # 1. Detect whether squeezing forces too high, try again
            force_too_high = False
            if np.log10(self.youngs) > 9 and avg_of_filter > 50:
                force_too_high = True
            elif np.log10(self.youngs) <= 9 and (avg_of_filter > 4 * self.desired_force) \
                    and avg_of_filter > 10:
                force_too_high = True
            if force_too_high:
                print("Squeezing force too high, reset")
                self.squeezing_close_fails += 1
                if self.squeezing_close_fails > 4:
                    self.state = "done"
                else:
                    self.reset_saved_state()

            # 2. Detect whether contact has been lost for a while
            if np.all(particles_contacting_gripper == 0.0):
                self.squeeze_lost_contact_counter += 1
            else:
                self.squeeze_lost_contact_counter = 0

            if self.squeeze_lost_contact_counter > 100:
                print("Lost contact during squeezing, reset")
                self.squeezing_close_fails += 1
                self.squeeze_lost_contact_counter = 0
                if self.squeezing_close_fails > 4:
                    self.state = "done"
                else:
                    self.reset_saved_state()

            # 3. Detect whether object is no longer between grippers
            if self.franka_dof_states['pos'][-3:][
                    1] < 0.0001 and self.franka_dof_states['pos'][-3:][
                        2] < 0.0001:
                print("Can't close that tightly during squeezing, reset")
                self.squeezing_close_fails += 1
                self.squeezing_no_grasp += 1
                if self.squeezing_close_fails > 4 or self.squeezing_no_grasp > 2:
                    self.state = "done"
                else:
                    self.reset_saved_state()

            # 4. Detect whether grippers have exceeded joint limits
            # (occurs when there are spikes in force readings-> spikes in torque responses)
            if self.franka_dof_states['pos'][-3:][
                    1] > 0.04 or self.franka_dof_states['pos'][-3:][2] > 0.04:
                print(self.env_id, "Grippers exceeded joint limits, reset")
                self.squeezing_close_fails += 1
                if self.squeezing_close_fails > 4:
                    self.state = "done"
                else:
                    self.reset_saved_state()

            ############################################################
            # DETECT STATE TRANSITION WHEN SQUEEZING FORCES ARE MET
            ############################################################
            if self.inferred_rot_force:
                self.inferred_rot_force_counter += 1
            squeeze_guard = (self.mode != 'reorient') or (
                self.inferred_rot_force_counter > 30)

            # If desired squeezing forces is met
            if np.all(
                    np.abs(self.f_errs) < 0.05 * self.desired_force
            ) and not self.squeezed_until_force and squeeze_guard and np.all(
                    particles_contacting_gripper > 0):

                if self.mode == "reorient":
                    assert (self.inferred_rot_force)
                self.squeezed_until_force = True

                # Record measurements once force is reached
                self.positions_at_force = np.copy(
                    self.particle_state_tensor.numpy()
                    [self.env_id * self.state_tensor_length:(self.env_id + 1)
                     * self.state_tensor_length, :])[:, :3]
                self.gripper_force_at_force = np.sum(F_curr[1:])
                self.gripper_distance_at_force = np.sum(
                    self.grippers_pre_squeeze) - curr_separation
                self.get_contact_geometry_features()
                stresses_at_force, self.se_at_force, _, _ = tet_based_metrics.get_tet_based_metrics(
                    self.gym_handle, self.sim_handle, self.env_handles,
                    self.env_id, self.particle_state_tensor, self.youngs)
                self.stresses_at_force = stresses_at_force[self.env_id]

                self.f_errs = np.ones(10)
                curr_joint_positions = self.gym_handle.get_actor_dof_states(
                    self.env_handle, self.platform_handle, gymapi.STATE_ALL)

                # Record location of gripper fingers
                mid_fk_map = panda_fk.get_fk(self.franka_dof_states['pos'],
                                             self.hand_origin,
                                             mode="mid")
                self.mid_finger_position_transformed = mid_fk_map.dot(
                    self.mid_finger_position)[:3]

                print("Platform lower")
                self.state = "hang"

        if self.state == "hang":
            curr_gripper_width = np.sum(self.franka_dof_states['pos'][-2:])
            if self.squeeze_min_gripper_width == 0.0:
                self.squeeze_min_gripper_width = curr_gripper_width
            else:
                if curr_gripper_width < self.squeeze_min_gripper_width:
                    self.squeeze_min_gripper_width = curr_gripper_width

            self.vel_des = np.zeros(NUM_JOINTS)
            self.hang_separations.append(curr_separation)

            # Add PID controller to achieve desired force
            F_des = np.array(
                [0.0, self.desired_force / 2.0, self.desired_force / 2.0])
            torque_des_force, F_curr_mag, F_err = self.get_force_based_velocity(
                F_des, F_curr)
            self.torque_des[-2:] = torque_des_force[1:]

            self.f_errs = np.hstack((self.f_errs[1:], F_err))

            curr_joint_positions = self.gym_handle.get_actor_dof_states(
                self.env_handle, self.platform_handle, gymapi.STATE_ALL)

            object_on_platform = self.object_contacting_platform()

            if np.all(np.abs(self.f_errs[-10:]) < 0.05
                      * self.desired_force) or np.all(self.f_errs[-10:] < 0.0):
                self.gym_handle.set_actor_dof_velocity_targets(
                    self.env_handle, self.platform_handle, [-0.08])
            else:
                self.gym_handle.set_actor_dof_velocity_targets(
                    self.env_handle, self.platform_handle, [0.0])

            if not object_on_platform:
                self.hang_counter += 1
                curr_stresses = tet_based_metrics.get_stresses_only(
                    self.gym_handle, self.sim_handle, self.env_handles,
                    self.env_id, self.particle_state_tensor)[self.env_id]
                self.hang_stresses.append(curr_stresses)

            if not object_on_platform and self.hang_counter > 50:
                self.pickup_success = True
                # Save current hanging mesh
                self.positions_under_gravity = np.copy(
                    self.particle_state_tensor.numpy()
                    [self.env_id * self.state_tensor_length:(self.env_id + 1)
                        * self.state_tensor_length, :][:, :3])

                # Get strain energies and stresses
                _, self.se_under_gravity, _, _ = tet_based_metrics.get_tet_based_metrics(
                    self.gym_handle, self.sim_handle, self.env_handles,
                    self.env_id, self.particle_state_tensor, self.youngs)
                self.stresses_under_gravity = np.mean(self.hang_stresses,
                                                      axis=0)

                # Get gripper force
                self.gripper_force_under_gravity = np.sum(F_curr[1:])
                print(self.env_id, "Force under gravity",
                      self.gripper_force_under_gravity)

                # Move the platform far away
                curr_joint_positions['pos'][0] = -0.4
                self.gym_handle.set_actor_dof_states(self.env_handle,
                                                     self.platform_handle,
                                                     curr_joint_positions,
                                                     gymapi.STATE_ALL)

                # Freeze gripper joints on both sides
                dof_props = self.gym_handle.get_actor_dof_properties(
                    self.env_handle, self.franka_handle)
                dof_props['lower'][
                    -1] = self.franka_dof_states['pos'][-1] - 1e-6
                dof_props['lower'][
                    -2] = self.franka_dof_states['pos'][-2] - 1e-6
                dof_props['upper'][
                    -1] = self.franka_dof_states['pos'][-1] + 1e-6
                dof_props['upper'][
                    -2] = self.franka_dof_states['pos'][-2] + 1e-6
                self.gym_handle.set_actor_dof_properties(
                    self.env_handle, self.franka_handle, dof_props)

                self.pickup_success = True
                if self.mode == "pickup":
                    self.state = "done"
                else:
                    self.state = self.mode
                print(self.env_id, "Pickup success:", self.pickup_success)

                # Set the desired squeezing gripper width
                curr_gripper_positions = np.copy(
                    self.franka_dof_states['pos'][-2:])
                des_gripper_width = np.mean(self.hang_separations[-30:])
                self.gripper_positions_under_gravity = np.copy(
                    self.franka_dof_states['pos'][-2:]) - 0.5 * (
                        np.sum(curr_gripper_positions) - des_gripper_width)

            elif (curr_joint_positions['pos'][0] <= -0.2
                  and object_on_platform) or np.all(
                      particles_contacting_gripper == 0.0):
                self.pickup_success = False
                print(self.env_id, "Pickup success:", self.pickup_success)
                self.state = "done"

        if self.state == 'reorient':

            curr_stresses = tet_based_metrics.get_stresses_only(
                self.gym_handle, self.sim_handle, self.env_handles,
                self.env_id, self.particle_state_tensor)[self.env_id]
            self.reorient_stress_history.append(curr_stresses)
            self.reorient_stress_history = self.reorient_stress_history[-20:]
            # Rotate according to rotation vectors
            rotation_axis = self.direction

            # Components of each revolute joint in rotation direction
            rot_vel = 3
            # See what angle has been rotated by
            self.reorient_counter += 1
            angle_covered = self.sim_params.dt * self.reorient_counter * rot_vel

            # Use Jacobian to rotate about centroid
            J = panda_fk.jacobian(self.franka_dof_states['pos'], self.hand_origin)
            w_des = rotation_axis
            d = self.object_centroid
            v = np.cross(d, w_des)
            V_st = np.concatenate((v, w_des))
            joint_vels = np.linalg.inv(J) @ V_st
            self.vel_des = np.zeros(NUM_JOINTS)
            self.vel_des[:6] = rot_vel * joint_vels
            self.vel_des[-2:] = np.array([0.0, 0.0])

            increments = np.linspace(np.pi / 4, np.pi, 4)
            for ind, ang in enumerate(increments):
                if angle_covered >= ang and np.all(
                        self.reorientation_meshes[ind]) == 0.0:
                    print(self.env_id, "Visited", ind)
                    self.reorientation_meshes[ind] = np.copy(
                        self.particle_state_tensor.numpy()
                        [self.env_id
                         * self.state_tensor_length:(self.env_id + 1)
                         * self.state_tensor_length, :][:, :3])

                    # Take a windowed average of reorientation_stresses
                    self.reorientation_stresses[ind] = np.mean(
                        self.reorient_stress_history, axis=0)

                    if ind >= len(increments) - 1:
                        self.state = "done"

            # If one side loses contact, go to done
            if np.all(particles_contacting_gripper == 0.0):
                print(self.env_id, "Lost contact during reorient",
                      particles_contacting_gripper)
                self.state = "done"

        if self.state == "shake":
            shake_direction = self.direction

            self.vel_des = np.zeros(NUM_JOINTS)
            max_shake_acc = 50
            Dt = self.sim_params.dt * self.shake_counter
            jerk = 100
            a = jerk * Dt
            a = min(max_shake_acc, jerk * Dt)
            self.shake_vel += a * self.sim_params.dt
            self.shake_counter += 1

            self.vel_des[:3] = self.shake_vel * shake_direction

            if a >= max_shake_acc:
                self.shake_fail_acc = max_shake_acc
                self.state = "done"

            curr_contacts = particles_contacting_gripper

            if np.any(curr_contacts == 0):
                self.shake_fail_acc = a
                self.state = "done"

        if self.state == "twist":

            # Move to the location
            curr_location = self.franka_dof_states['pos'][10:13]
            travel_location = np.array([0, 0, -GRIPPER_TIP_Z_OFFSET])

            if not self.reached_twist_location:
                self.twist_travel_counter += 1
                Dt = self.sim_params.dt * self.twist_travel_counter
                travel_location = np.array([0, 0, -GRIPPER_TIP_Z_OFFSET])

                acc = 3
                if (curr_location[-1] - travel_location[-1]) > (GRIPPER_TIP_Z_OFFSET / 2.):
                    self.travel_speed += acc * self.sim_params.dt
                else:
                    acc = -acc
                    self.travel_speed += acc * self.sim_params.dt
                self.travel_speed = min(self.travel_speed, 0.5)
                self.travel_speed = max(self.travel_speed, 0.1)
                travel_vel = self.travel_speed * travel_location / np.linalg.norm(
                    travel_location)

                self.vel_des = np.zeros(NUM_JOINTS)
                self.vel_des[10:13] = travel_vel
                self.vel_des = np.asarray(self.vel_des, dtype=np.float32)
                self.gym_handle.set_actor_dof_velocity_targets(
                    self.env_handle, self.franka_handle, self.vel_des)
            if np.linalg.norm(travel_location - curr_location
                              ) < 0.01 and not self.reached_twist_location:
                self.vel_des = np.zeros(NUM_JOINTS)
                self.reached_twist_location = True

                # Lock locations of sliding joints
                dof_props = self.gym_handle.get_actor_dof_properties(
                    self.env_handle, self.franka_handle)
                dof_props['driveMode'][10] = gymapi.DOF_MODE_POS
                dof_props['driveMode'][11] = gymapi.DOF_MODE_POS
                dof_props['driveMode'][12] = gymapi.DOF_MODE_POS
                dof_props['driveMode'][13] = gymapi.DOF_MODE_POS
                dof_props['driveMode'][-2] = gymapi.DOF_MODE_POS
                dof_props['driveMode'][-1] = gymapi.DOF_MODE_POS

                dof_props['lower'][
                    -1] = self.franka_dof_states['pos'][-1] - 1e-6
                dof_props['lower'][
                    -2] = self.franka_dof_states['pos'][-2] - 1e-6
                dof_props['upper'][
                    -1] = self.franka_dof_states['pos'][-1] + 1e-6
                dof_props['upper'][
                    -2] = self.franka_dof_states['pos'][-2] + 1e-6

                self.gym_handle.set_actor_dof_properties(
                    self.env_handle, self.franka_handle, dof_props)
                self.pos_des = np.zeros(NUM_JOINTS)
                self.pos_des[12] = self.franka_dof_states['pos'][12]
                self.pos_des[-2] = self.franka_dof_states['pos'][-2]
                self.pos_des[-1] = self.franka_dof_states['pos'][-1]
                self.pos_des = np.asarray(self.pos_des, dtype=np.float32)
                self.gym_handle.set_actor_dof_position_targets(
                    self.env_handle, self.franka_handle, self.pos_des)

            if self.reached_twist_location:
                self.twist_counter += 1
                max_rot_acc = 1250
                Dt = self.sim_params.dt * self.twist_counter
                jerk = 2500
                a = min(max_rot_acc, jerk * Dt)

                self.rot_vel += a * self.sim_params.dt
                self.vel_des = np.zeros(NUM_JOINTS)
                self.vel_des[6] = self.rot_vel
                self.vel_des = np.asarray(self.vel_des, dtype=np.float32)
                self.gym_handle.set_actor_dof_velocity_targets(
                    self.env_handle, self.franka_handle, self.vel_des)
                if a >= max_rot_acc:
                    self.twist_fail_acc = max_rot_acc
                    self.state = "done"

                curr_contacts = particles_contacting_gripper

                if np.any(curr_contacts == 0):
                    print("Lost contact")
                    self.twist_fail_acc = a
                    self.state = "done"

        if self.state == "done":
            print(self.shake_fail_acc, self.twist_fail_acc)
            self.vel_des = np.zeros(NUM_JOINTS)

        # Apply desired velocity targets
        self.vel_des = np.asarray(self.vel_des, dtype=np.float32)
        self.pos_des = np.asarray(self.pos_des, dtype=np.float32)
        self.torque_des = np.asarray(self.torque_des, dtype=np.float32)

        # States that require velocity control
        if self.state in ["close", "close_soft"]:
            dof_props = self.gym_handle.get_actor_dof_properties(
                self.env_handle, self.franka_handle)
            dof_props['driveMode'][-1] = gymapi.DOF_MODE_VEL
            dof_props['driveMode'][-2] = gymapi.DOF_MODE_VEL
            self.gym_handle.set_actor_dof_properties(self.env_handle,
                                                     self.franka_handle,
                                                     dof_props)

        # States that require torque control
        if self.state in ["squeeze", "hang"]:

            # Change mode of the fingers to torque control
            dof_props = self.gym_handle.get_actor_dof_properties(
                self.env_handle, self.franka_handle)
            dof_props['driveMode'][-1] = gymapi.DOF_MODE_EFFORT
            dof_props['driveMode'][-2] = gymapi.DOF_MODE_EFFORT
            self.gym_handle.set_actor_dof_properties(self.env_handle,
                                                     self.franka_handle,
                                                     dof_props)

            self.gym_handle.apply_actor_dof_efforts(self.env_handle,
                                                    self.franka_handle,
                                                    self.torque_des)
            self.gym_handle.set_actor_dof_velocity_targets(
                self.env_handle, self.franka_handle, self.vel_des)

            if self.state == "hang":
                # Lock position of grippers from opening further
                dof_props = self.gym_handle.get_actor_dof_properties(
                    self.env_handle, self.franka_handle)
                dof_props['upper'][
                    -1] = self.franka_dof_states['pos'][-1] + 1e-6
                dof_props['upper'][
                    -2] = self.franka_dof_states['pos'][-2] + 1e-6
                self.gym_handle.set_actor_dof_properties(
                    self.env_handle, self.franka_handle, dof_props)

        self.gym_handle.set_actor_dof_velocity_targets(self.env_handle,
                                                       self.franka_handle,
                                                       self.vel_des)

        # Tune the robot joint damping
        dof_props = self.gym_handle.get_actor_dof_properties(
            self.env_handle, self.franka_handle)
        dof_props['damping'][-2:] = np.repeat(100., 2)

        self.gym_handle.set_actor_dof_properties(self.env_handle,
                                                 self.franka_handle, dof_props)

        return
