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
"""Helper functions to calculate and write metrics and features."""

import os

import h5py
import numpy as np
from isaacgym import gymapi
from scipy.spatial.transform import Rotation as R


def get_franka_rpy(trimesh_grasp_quat):
    """Return RPY angles for Panda joints based on the grasp pose in the Z-up convention."""
    neg_rot_x = gymapi.Quat(0.7071068, 0, 0, -0.7071068)
    rot_z = gymapi.Quat(0, 0, 0.7071068, 0.7071068)
    desired_transform = neg_rot_x * trimesh_grasp_quat * rot_z
    r = R.from_quat([
        desired_transform.x, desired_transform.y, desired_transform.z,
        desired_transform.w
    ])
    return desired_transform, r.as_euler('ZYX')


def get_global_deformation_metrics(undeformed_mesh, deformed_mesh, get_field=False):
    """Get the mean and max deformation of the nodes over the entire mesh.

    Involves separating the pure deformation field from the raw displacement field.
    """
    num_nodes = undeformed_mesh.shape[0]
    undeformed_positions = undeformed_mesh[:, :3]
    deformed_positions = deformed_mesh[:, :3]

    centered_undeformed_positions = undeformed_positions - np.mean(undeformed_positions, axis=0)
    centered_deformed_positions = deformed_positions - np.mean(undeformed_positions, axis=0)

    # Extract deformations by rigid body motion
    axis_angle, t = rigid_body_motion(centered_undeformed_positions, centered_deformed_positions)
    rot = R.from_rotvec(axis_angle)
    aligned_deformed_positions = centered_deformed_positions
    for i in range(num_nodes):
        aligned_deformed_positions[i, :] = np.linalg.inv(
            rot.as_matrix()) @ (centered_deformed_positions[i, :] - t)
    centered_deformed_positions = aligned_deformed_positions

    def_field = centered_deformed_positions - centered_undeformed_positions
    deformation_norms = np.linalg.norm(def_field, axis=1)

    if get_field:
        return def_field
    return np.mean(deformation_norms), np.max(deformation_norms), np.median(deformation_norms)


def rigid_body_motion(P, Q):
    """Return best-fit rigid body motion from point sets P to Q."""
    P = np.transpose(P)  # Previous positions
    Q = np.transpose(Q)  # Current positions
    n = P.shape[1]

    # Center everything in the middle
    origin_offset = np.vstack(P.mean(axis=1))
    P = P - origin_offset
    Q = Q - origin_offset

    # Compute the weight centroids of both point sets
    P_mean = P.mean(axis=1)
    Q_mean = Q.mean(axis=1)

    # Compute the centered vectors
    X = P - np.matrix(P_mean).T
    Y = Q - np.matrix(Q_mean).T
    W = np.diag(np.ones(n) / n)

    # Compute the covariance matrix
    S = X @ W @ Y.T

    # Get the SVD, S factors as U @ np.diag(Sig) @ Vh
    U, Sig, Vh = np.linalg.svd(S, full_matrices=True)

    # Optimal rotation and translation
    d = np.linalg.det(Vh.T @ U.T)
    Rot = Vh.T @ np.diag([1, 1, d]) @ U.T
    t = Q_mean - Rot @ P_mean

    Rot_scipy = R.from_matrix(Rot)
    axis_angle = Rot_scipy.as_rotvec()

    return axis_angle, np.asarray(t)[0]


def write_metrics_to_h5(mode, grasp_ind, oris, num_grasp_poses,
                        num_directions, h5_file_path, panda_fsms, num_dp):
    """Write metrics and features to h5 result file."""
    h5_file_name = h5_file_path
    print("Trying to write to", h5_file_name)

    if not os.path.exists(h5_file_name):
        print("Writing to new file", h5_file_name)
        os.makedirs(os.path.dirname(h5_file_name), exist_ok=True)
        with h5py.File(h5_file_name, "w") as hf:

            hf.create_dataset("grasp_index", (num_grasp_poses,), maxshape=(None,))
            hf.create_dataset("pickup_success", (num_grasp_poses,), maxshape=(None,))
            hf.create_dataset("timed_out",
                              (num_grasp_poses, num_directions),
                              maxshape=(None,
                                        num_directions))
            # Stress-based metrics
            hf.create_dataset("pre_contact_stresses",
                              (num_grasp_poses, ) + panda_fsms[0].pre_contact_stresses.shape,
                              maxshape=(None, ) + panda_fsms[0].pre_contact_stresses.shape)
            hf.create_dataset("stresses_at_force",
                              (num_grasp_poses, ) + panda_fsms[0].pre_contact_stresses.shape,
                              maxshape=(None, ) + panda_fsms[0].pre_contact_stresses.shape)
            hf.create_dataset("stresses_under_gravity",
                              (num_grasp_poses, ) + panda_fsms[0].pre_contact_stresses.shape,
                              maxshape=(None, ) + panda_fsms[0].pre_contact_stresses.shape)

            # Strain energy-based metrics
            hf.create_dataset("pre_contact_se", (num_grasp_poses,),
                              maxshape=(None,))
            hf.create_dataset("se_at_force", (num_grasp_poses,),
                              maxshape=(None,))
            hf.create_dataset("se_under_gravity", (num_grasp_poses,),
                              maxshape=(None,))

            # Position-based metrics
            hf.create_dataset("pre_contact_positions",
                              (num_grasp_poses, panda_fsms[0].state_tensor_length, 3),
                              maxshape=(None,
                                        panda_fsms[0].state_tensor_length, 3))
            hf.create_dataset("positions_at_force",
                              (num_grasp_poses, panda_fsms[0].state_tensor_length, 3),
                              maxshape=(None,
                                        panda_fsms[0].state_tensor_length, 3))
            hf.create_dataset("positions_under_gravity",
                              (num_grasp_poses, panda_fsms[0].state_tensor_length, 3),
                              maxshape=(None,
                                        panda_fsms[0].state_tensor_length, 3))

            # Contact-based metrics
            hf.create_dataset("left_contacted_nodes_under_gravity",
                              (num_grasp_poses, panda_fsms[0].state_tensor_length),
                              maxshape=(None,
                                        panda_fsms[0].state_tensor_length))
            hf.create_dataset("right_contacted_nodes_under_gravity",
                              (num_grasp_poses, panda_fsms[0].state_tensor_length),
                              maxshape=(None,
                                        panda_fsms[0].state_tensor_length))
            hf.create_dataset("left_contacted_nodes_under_gravity_initial",
                              (num_grasp_poses, panda_fsms[0].state_tensor_length),
                              maxshape=(None,
                                        panda_fsms[0].state_tensor_length))
            hf.create_dataset("right_contacted_nodes_under_gravity_initial",
                              (num_grasp_poses, panda_fsms[0].state_tensor_length),
                              maxshape=(None,
                                        panda_fsms[0].state_tensor_length))

            # Squeezing in no-gravity metrics
            hf.create_dataset("stacked_left_node_contacts",
                              (num_grasp_poses, num_dp, panda_fsms[0].state_tensor_length, 6),
                              maxshape=(None, num_dp,
                                        panda_fsms[0].state_tensor_length, 6))
            hf.create_dataset("stacked_right_node_contacts",
                              (num_grasp_poses, num_dp, panda_fsms[0].state_tensor_length, 6),
                              maxshape=(None, num_dp,
                                        panda_fsms[0].state_tensor_length, 6))
            hf.create_dataset("stacked_left_gripper_contact_points",
                              (num_grasp_poses, num_dp, panda_fsms[0].state_tensor_length, 3),
                              maxshape=(None, num_dp,
                                        panda_fsms[0].state_tensor_length, 3))
            hf.create_dataset("stacked_right_gripper_contact_points",
                              (num_grasp_poses, num_dp, panda_fsms[0].state_tensor_length, 3),
                              maxshape=(None, num_dp,
                                        panda_fsms[0].state_tensor_length, 3))
            hf.create_dataset("stacked_gripper_positions",
                              (num_grasp_poses, num_dp, 2),
                              maxshape=(None, num_dp, 2))
            hf.create_dataset("stacked_positions",
                              (num_grasp_poses, num_dp, panda_fsms[0].state_tensor_length, 3),
                              maxshape=(None, num_dp,
                                        panda_fsms[0].state_tensor_length, 3))
            hf.create_dataset("stacked_stresses",
                              (num_grasp_poses, num_dp,) + panda_fsms[0].pre_contact_stresses.shape,
                              maxshape=(None, num_dp,) + panda_fsms[0].pre_contact_stresses.shape)
            hf.create_dataset("stacked_forces",
                              (num_grasp_poses, num_dp,),
                              maxshape=(None, num_dp,))
            hf.create_dataset("stacked_forces_on_nodes",
                              (num_grasp_poses, num_dp, panda_fsms[0].state_tensor_length),
                              maxshape=(None, num_dp, panda_fsms[0].state_tensor_length))

            hf.create_dataset("squeeze_no_gravity_max_force", (num_grasp_poses,),
                              maxshape=(None,))
            hf.create_dataset("squeeze_no_gravity_failed_to_increase", (num_grasp_poses,),
                              maxshape=(None,))
            hf.create_dataset("squeeze_no_gravity_lost_contact", (num_grasp_poses,),
                              maxshape=(None,))

            # Gripper-based metrics
            hf.create_dataset("gripper_distance_at_force", (num_grasp_poses,),
                              maxshape=(None,))
            hf.create_dataset("gripper_force_at_force", (num_grasp_poses,),
                              maxshape=(None,))
            hf.create_dataset("gripper_force_under_gravity", (num_grasp_poses,),
                              maxshape=(None,))
            hf.create_dataset("gripper_positions_under_gravity", (num_grasp_poses, 2),
                              maxshape=(None, 2))

            # Desired force setpoints
            hf.create_dataset("initial_desired_force", (num_grasp_poses,),
                              maxshape=(None,))
            hf.create_dataset("corrected_desired_force", (num_grasp_poses, num_directions),
                              maxshape=(None, num_directions))

            # Directions for all the tests
            hf.create_dataset("directions",
                              (num_grasp_poses, num_directions, 3),
                              maxshape=(None,
                                        num_directions, 3))

            # Reorientation meshes
            hf.create_dataset("reorientation_meshes",
                              (num_grasp_poses, num_directions, 4,
                               panda_fsms[0].state_tensor_length, 3),
                              maxshape=(None,
                                        num_directions, 4, panda_fsms[0].state_tensor_length, 3))
            hf.create_dataset("reorientation_stresses",
                              (num_grasp_poses, num_directions, 4,
                               panda_fsms[0].pre_contact_stresses.shape[0]),
                              maxshape=(None, num_directions, 4,
                                        panda_fsms[0].pre_contact_stresses.shape[0]))

            # Stability metrics
            hf.create_dataset("lin_acc_fail_accs",
                              (num_grasp_poses, num_directions),
                              maxshape=(None,
                                        num_directions))

            hf.create_dataset("ang_acc_fail_accs",
                              (num_grasp_poses, num_directions),
                              maxshape=(None,
                                        num_directions))

            # Geometry metrics
            hf.create_dataset("pure_distances",
                              (num_grasp_poses, 2),
                              maxshape=(None,
                                        2))
            hf.create_dataset("perp_distances",
                              (num_grasp_poses, 2),
                              maxshape=(None,
                                        2))
            hf.create_dataset("edge_distances",
                              (num_grasp_poses, 2),
                              maxshape=(None,
                                        2))
            hf.create_dataset("num_gripper_contacts",
                              (num_grasp_poses, 2),
                              maxshape=(None,
                                        2))

    try:
        with h5py.File(h5_file_name, 'a') as hf:
            for i, panda_fsm in enumerate(panda_fsms):
                if mode in ["reorient", "lin_acc", "ang_acc"]:
                    grasp_index = grasp_ind
                    ori_index = i + oris[0]
                else:
                    grasp_index = i + grasp_ind
                    ori_index = 0  # no orientations involved in other modes

                pickup_success_dset = hf['pickup_success']
                pickup_success_dset[grasp_index] = panda_fsm.pickup_success

                timed_out_dset = hf['timed_out']
                timed_out_dset[grasp_index, ori_index] = panda_fsm.timed_out

                pre_contact_stresses_dset = hf['pre_contact_stresses']
                pre_contact_stresses_dset[grasp_index, :] = panda_fsm.pre_contact_stresses

                stresses_at_force_dset = hf['stresses_at_force']
                stresses_at_force_dset[grasp_index, :] = panda_fsm.stresses_at_force

                stresses_under_gravity_dset = hf['stresses_under_gravity']
                stresses_under_gravity_dset[grasp_index, :] = panda_fsm.stresses_under_gravity

                pre_contact_se_dset = hf['pre_contact_se']
                pre_contact_se_dset[
                    grasp_index] = panda_fsm.pre_contact_se

                se_at_force_dset = hf['se_at_force']
                se_at_force_dset[
                    grasp_index] = panda_fsm.se_at_force

                se_under_gravity_dset = hf['se_under_gravity']
                se_under_gravity_dset[
                    grasp_index] = panda_fsm.se_under_gravity

                pre_contact_positions_dset = hf['pre_contact_positions']
                pre_contact_positions_dset[grasp_index, :, :] = panda_fsm.undeformed_mesh

                positions_at_force_dset = hf['positions_at_force']
                positions_at_force_dset[grasp_index, :, :] = panda_fsm.positions_at_force

                positions_under_gravity_dset = hf['positions_under_gravity']
                if np.all(positions_under_gravity_dset[grasp_index] == 0):
                    positions_under_gravity_dset[
                        grasp_index, :, :] = panda_fsm.positions_under_gravity

                left_contacted_nodes_under_gravity_dset = hf['left_contacted_nodes_under_gravity']
                left_contacted_nodes_under_gravity_dset[grasp_index,
                                                        :] = panda_fsm.left_gripper_node_contacts

                right_contacted_nodes_under_gravity_dset = hf['right_contacted_nodes_under_gravity']
                right_contacted_nodes_under_gravity_dset[grasp_index,
                                                         :] = panda_fsm.right_gripper_node_contacts

                l_nodes_gravity_init_dset = hf['left_contacted_nodes_under_gravity_initial']
                l_nodes_gravity_init_dset[grasp_index,
                                          :] = panda_fsm.left_gripper_node_contacts_initial

                r_nodes_gravity_init_dset = hf['right_contacted_nodes_under_gravity_initial']
                r_nodes_gravity_init_dset[grasp_index,
                                          :] = panda_fsm.right_gripper_node_contacts_initial

                # Squeezing under no gravity

                stacked_left_node_contacts_dset = hf['stacked_left_node_contacts']
                stacked_left_node_contacts_dset[grasp_index, :,
                                                :, :] = panda_fsm.stacked_left_node_contacts

                stacked_right_node_contacts_dset = hf['stacked_right_node_contacts']
                stacked_right_node_contacts_dset[grasp_index, :,
                                                 :, :] = panda_fsm.stacked_right_node_contacts

                stkd_l_gripper_contact_dset = hf['stacked_left_gripper_contact_points']
                stkd_l_gripper_contact_dset[grasp_index, :,
                                            :, :] = panda_fsm.stacked_left_gripper_contact_points

                stkd_r_gripper_contact_dset = hf['stacked_right_gripper_contact_points']
                stkd_r_gripper_contact_dset[grasp_index, :,
                                            :, :] = panda_fsm.stacked_right_gripper_contact_points

                stacked_gripper_positions_dset = hf['stacked_gripper_positions']
                stacked_gripper_positions_dset[grasp_index, :,
                                               :] = panda_fsm.stacked_gripper_positions

                stacked_stresses_dset = hf['stacked_stresses']
                stacked_stresses_dset[grasp_index, :, :] = panda_fsm.stacked_stresses

                stacked_positions_dset = hf['stacked_positions']
                stacked_positions_dset[grasp_index, :, :, :] = panda_fsm.stacked_positions

                stacked_forces_dset = hf['stacked_forces']
                stacked_forces_dset[grasp_index, :] = panda_fsm.stacked_forces

                stacked_forces_on_nodes_dset = hf['stacked_forces_on_nodes']
                stacked_forces_on_nodes_dset[grasp_index, :, :] = panda_fsm.stacked_forces_on_nodes

                sng_max_f_dset = hf['squeeze_no_gravity_max_force']
                sng_max_f_dset[grasp_index] = panda_fsm.squeeze_no_gravity_max_force

                sng_incr_fail_fset = hf['squeeze_no_gravity_failed_to_increase']
                sng_incr_fail_fset[grasp_index] = panda_fsm.squeeze_no_gravity_failed_to_increase

                sng_lost_contact_dset = hf['squeeze_no_gravity_lost_contact']
                sng_lost_contact_dset[grasp_index] = panda_fsm.squeeze_no_gravity_lost_contact

                ######################

                gripper_distance_at_force_dset = hf['gripper_distance_at_force']
                gripper_distance_at_force_dset[
                    grasp_index] = panda_fsm.gripper_distance_at_force

                gripper_force_at_force_dset = hf['gripper_force_at_force']
                gripper_force_at_force_dset[
                    grasp_index] = panda_fsm.gripper_force_at_force

                gripper_force_under_gravity_dset = hf['gripper_force_under_gravity']
                gripper_force_under_gravity_dset[
                    grasp_index] = panda_fsm.gripper_force_under_gravity

                gripper_positions_under_gravity_dset = hf['gripper_positions_under_gravity']
                gripper_positions_under_gravity_dset[
                    grasp_index] = panda_fsm.gripper_positions_under_gravity

                initial_desired_force_dset = hf['initial_desired_force']
                initial_desired_force_dset[grasp_index] = panda_fsm.initial_desired_force

                corrected_desired_force_dset = hf['corrected_desired_force']
                corrected_desired_force_dset[grasp_index,
                                             ori_index] = panda_fsm.corrected_desired_force

                pure_distances_dset = hf['pure_distances']
                pure_distances_dset[
                    grasp_index] = panda_fsm.pure_distances

                perp_distances_dset = hf['perp_distances']
                if np.all(perp_distances_dset[grasp_index] == np.zeros(2)):
                    perp_distances_dset[
                        grasp_index] = panda_fsm.perp_distances

                edge_distances_dset = hf['edge_distances']
                if np.all(edge_distances_dset[grasp_index] == np.zeros(2)):
                    edge_distances_dset[
                        grasp_index] = panda_fsm.edge_distances

                num_gripper_contacts_dset = hf['num_gripper_contacts']
                if np.all(num_gripper_contacts_dset[grasp_index] == np.zeros(2)):
                    num_gripper_contacts_dset[
                        grasp_index] = panda_fsm.num_gripper_contacts

                # Reorientation datasets
                reorientation_quats_dset = hf['directions']
                reorientation_quats_dset[grasp_index, ori_index, :] = panda_fsm.directions[0]

                if mode == "reorient":
                    reorientation_meshes_dset = hf['reorientation_meshes']
                    reorientation_meshes_dset[grasp_index, ori_index,
                                              :, :, :] = panda_fsm.reorientation_meshes

                    reorientation_stresses_dset = hf['reorientation_stresses']
                    reorientation_stresses_dset[grasp_index, ori_index,
                                                :, :] = panda_fsm.reorientation_stresses

                elif mode == "lin_acc":
                    lin_acc_fail_accs_dset = hf['lin_acc_fail_accs']
                    lin_acc_fail_accs_dset[grasp_index, ori_index] = panda_fsm.lin_acc_fail_acc

                elif mode == "ang_acc":
                    ang_acc_fail_accs_dset = hf['ang_acc_fail_accs']
                    ang_acc_fail_accs_dset[grasp_index, ori_index] = panda_fsm.ang_acc_fail_acc

        hf.close()

    except BaseException:
        print("Couldn't record data")
        pass
