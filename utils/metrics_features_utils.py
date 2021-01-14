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


def get_desired_rpy(reorient_quat, grasp_quat):
    """Return RPY angles for Panda joints based on the grasp pose in the Z-up convention."""
    neg_rot_x = gymapi.Quat(0.7071068, 0, 0, -0.7071068)
    rot_z = gymapi.Quat(0, 0, 0.7071068, 0.7071068)
    desired_transform = neg_rot_x * reorient_quat * \
        grasp_quat * rot_z  # Changed here to test reorient code
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


def write_metrics_to_h5(args, num_grasp_poses, num_directions, h5_file_path, panda_fsms):
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
            hf.create_dataset("shake_fail_accs",
                              (num_grasp_poses, num_directions),
                              maxshape=(None,
                                        num_directions))

            hf.create_dataset("twist_fail_accs",
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

    with h5py.File(h5_file_name, 'a') as hf:
        for i, panda_fsm in enumerate(panda_fsms):
            if args.mode in ["reorient", "shake", "twist"]:
                grasp_index = args.grasp_ind
                ori_index = i + args.ori_start
            else:
                grasp_index = i + args.grasp_ind
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
            corrected_desired_force_dset[grasp_index, ori_index] = panda_fsm.corrected_desired_force

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

            if args.mode == "reorient":
                reorientation_meshes_dset = hf['reorientation_meshes']
                reorientation_meshes_dset[grasp_index, ori_index,
                                          :, :, :] = panda_fsm.reorientation_meshes

                reorientation_stresses_dset = hf['reorientation_stresses']
                reorientation_stresses_dset[grasp_index, ori_index,
                                            :, :] = panda_fsm.reorientation_stresses

            elif args.mode == "shake":
                shake_fail_accs_dset = hf['shake_fail_accs']
                shake_fail_accs_dset[grasp_index, ori_index] = panda_fsm.shake_fail_acc

            elif args.mode == "twist":
                twist_fail_accs_dset = hf['twist_fail_accs']
                twist_fail_accs_dset[grasp_index, ori_index] = panda_fsm.twist_fail_acc

    hf.close()
