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


"""Perform a grasp evaluation on a soft object using a Panda hand.

Usage example:
python3 run_grasp_evaluation.py --object=rectangle --grasp_ind=3 --youngs=2e5 --density=1000
    --ori_start=10 --ori_end=10 --mode=reorient --write
"""

import argparse
import os
import timeit
import xml.etree.ElementTree as ET

import h5py
import numpy as np
from isaacgym import gymapi
from scipy.spatial.transform import Rotation as R

from utils import pandafsm
from utils import uniform_sphere
from utils import metrics_features_utils


ASSETS_DIR = "examples/"
RESULTS_DIR = "results/"
RESULTS_STORAGE_TAG = "_local"
PLATFORM_HEIGHT = 1.0


def set_object_parameters(asset_file_object, **kwargs):
    """Write object parameters into URDF file (TODO: This function is now a Gym method)."""
    try:
        tree = ET.parse(asset_file_object)
        root = tree.getroot()
        for key, value in kwargs.items():
            for attribute in root.iter(key):
                attribute.set('value', str(value))
        tree.write(asset_file_object)
        return True
    except BaseException:
        return False


def get_height_of_objects(tet_file):
    """Return the height of the test object."""
    mesh_lines = list(open(tet_file, "r"))
    mesh_lines = [line.strip('\n') for line in mesh_lines]
    zs = []
    for ml in mesh_lines:
        sp = ml.split(" ")
        if sp[0] == 'v':
            zs.append(float(sp[3]))
    return 2 * abs(min(zs))


def create_sim(gym, use_viewer, args):
    """Set sim parameters and create a Sim object."""
    # Set simulation parameters
    sim_type = gymapi.SIM_FLEX  # Can also use SIM_PHYSX
    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 1500
    sim_params.substeps = 1
    sim_params.gravity = gymapi.Vec3(0.0, -9.81, 0.0)
    if args.mode in ["shake", "twist"]:
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, 0.0)

    # Set stress visualization parameters
    sim_params.stress_visualization = True
    sim_params.stress_visualization_min = 1.0e2
    sim_params.stress_visualization_max = 1e5

    # Set FleX-specific parameters
    sim_params.flex.solver_type = 5
    sim_params.flex.num_outer_iterations = 10
    sim_params.flex.num_inner_iterations = 200
    sim_params.flex.relaxation = 0.75
    sim_params.flex.warm_start = 0.8

    sim_params.flex.deterministic_mode = True

    # Set contact parameters
    sim_params.flex.shape_collision_distance = 5e-4
    sim_params.flex.contact_regularization = 1.0e-6
    sim_params.flex.shape_collision_margin = 1.0e-4
    sim_params.flex.dynamic_friction = 0.7

    # Create Sim object
    gpu_physics = 0
    gpu_render = 0
    if not use_viewer:
        gpu_render = -1
    return gym.create_sim(gpu_physics, gpu_render, sim_type,
                          sim_params), sim_params


def main():
    """Run grasp evaluation."""
    # Create command line flag options
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--object', required=True, help="Name of object")

    parser.add_argument('--grasp_ind', default=0, type=int, help="Index of grasp candidate to test")
    parser.add_argument(
        '--ori_start',
        default=0,
        type=int,
        help="Start index of vectors to test. [0, 15]")
    parser.add_argument(
        '--ori_end',
        default=5,
        type=int,
        help="End index of vectors to test. [0, 15]")
    parser.add_argument('--density', default='1000', type=str, help="Density of object [kg/m^3]")
    parser.add_argument('--youngs', default='5e5', type=str, help="Elastic modulus of object [Pa]")
    parser.add_argument('--poissons', default='0.3', type=str, help="Poisson's ratio of object")
    parser.add_argument(
        '--mode',
        default='pickup',
        type=str,
        help="Name of test to run, one of {pickup, reorient, shake, twist}")
    parser.add_argument(
        '--viewer',
        dest='viewer',
        action='store_true',
        help="Display graphical viewer")
    parser.add_argument(
        '--no-viewer',
        dest='viewer',
        action='store_false',
        help="Do not display graphical viewer.")

    parser.add_argument('--fill', dest='fill', action='store_true',
                        help="Run only tests without pre-existing results")
    parser.add_argument('--no-fill', dest='fill', action='store_false',
                        help="Run all tests regardless of pre-existing results")
    parser.add_argument('--write', dest='write_results', action='store_true', help="Record results")
    parser.add_argument('--no-write',
                        dest='write_results',
                        action='store_false', help="Do not record results")
    parser.add_argument(
        '--tag',
        default='',
        type=str,
        help="Additional tring to add onto name of results files.")
    parser.set_defaults(viewer=False)
    parser.set_defaults(viewer=True)
    parser.set_defaults(write_results=False)
    args = parser.parse_args()

    use_viewer = args.viewer
    write_results = args.write_results
    object_name = args.object
    object_path = os.path.join(ASSETS_DIR, object_name)

    folder_name = object_name + RESULTS_STORAGE_TAG
    object_file_name = object_name + "_" + args.density + "_" + args.youngs + "_" + \
        args.poissons + "_" + args.mode + "_tag" + args.tag + "_results.h5"
    h5_file_path = os.path.join(RESULTS_DIR, folder_name, args.youngs, object_file_name)

    # Optionally skip data collection if good data already exists (args.fill flag)
    if os.path.exists(h5_file_path) and args.fill:
        existing_h5 = h5py.File(h5_file_path, 'r')

        existing_timed_out = existing_h5['timed_out'][args.grasp_ind,
                                                      args.ori_start]
        existing_succeeded = True

        if args.mode == "pickup":
            existing_pos_under_gravity_dset = existing_h5[
                'positions_under_gravity']
            if np.all(existing_pos_under_gravity_dset[args.grasp_ind] == 0):
                existing_succeeded = False

        if args.mode == "reorient":
            reorientation_meshes_dset = existing_h5['reorientation_meshes']
            if np.all(reorientation_meshes_dset[args.grasp_ind, args.ori_start,
                                                0] == 0):
                existing_succeeded = False

        if args.mode == "shake":
            shake_fail_accs_dset = existing_h5['shake_fail_accs']
            if shake_fail_accs_dset[args.grasp_ind, args.ori_start] == 0.0:
                existing_succeeded = False

        if args.mode == "twist":
            twist_fail_accs_dset = existing_h5['twist_fail_accs']
            if twist_fail_accs_dset[args.grasp_ind, args.ori_start] == 0.0:
                existing_succeeded = False

        existing_h5.close()
        if existing_timed_out == 0.0 and existing_succeeded:
            print("Data already exists, returning")
            return
        else:
            print("Existing data is imperfect, rerunning")

    # Get the grasp candidates
    grasp_file_name = object_name + "_grasps.h5"
    f = h5py.File(os.path.realpath(os.path.join(object_path, grasp_file_name)), 'r')
    grasp_candidate_poses = f['poses'][args.grasp_ind:args.grasp_ind + 1]
    num_grasp_poses = f['poses'].shape[0]
    f.close()

    # Create Gym object
    gym = gymapi.acquire_gym()
    sim, sim_params = create_sim(gym, use_viewer, args)

    # Define scene and environments
    envs_per_row = 6
    env_dim = 0.3
    if args.mode in ["shake", "twist"]:
        env_dim = 1.0

    # Define environment as half-cube (half in vertical direction)
    env_lower = gymapi.Vec3(-env_dim, 0, -env_dim)
    env_upper = gymapi.Vec3(env_dim, env_dim, env_dim)

    # Define asset properties
    asset_root = './'
    asset_options = gymapi.AssetOptions()
    asset_options.flip_visual_attachments = False
    asset_options.armature = 0.0  # Additional moment of inertia due to motors
    # 1e-4  # Collision distance for rigid bodies. Minkowski sum of collision
    # mesh and sphere. Default value is large, so set explicitly
    asset_options.thickness = 0.0
    asset_options.linear_damping = 1.0  # Linear damping for rigid bodies
    asset_options.angular_damping = 0.0  # Angular damping for rigid bodies
    asset_options.disable_gravity = True
    # Activates PD position, velocity, or torque controller, instead of doing
    # DOF control in post-processing
    asset_options.default_dof_drive_mode = gymapi.DOF_MODE_VEL

    # Load Franka and object assets
    asset_file_franka = 'franka_description/robots/franka_panda_fem_simple_v4_with_arm.urdf'
    asset_file_platform = os.path.join(ASSETS_DIR, 'platform.urdf')
    asset_file_object = os.path.join(object_path, "soft_body.urdf")

    # Set object parameters based on command line args (TODO: Use new methods)
    set_parameter_result = False
    fail_counter = 0
    while set_parameter_result is False and fail_counter < 10:
        try:
            set_parameter_result = set_object_parameters(
                asset_file_object,
                density=args.density,
                youngs=args.youngs,
                poissons=args.poissons)
        except BaseException:
            fail_counter += 1
            pass

    # Set asset options
    asset_options.fix_base_link = True
    asset_handle_franka = gym.load_asset(sim, asset_root, asset_file_franka,
                                         asset_options)
    asset_options.fix_base_link = False
    asset_options.min_particle_mass = 1e-20  # 1e-4 by default
    asset_handle_object = gym.load_asset(sim, asset_root, asset_file_object,
                                         asset_options)
    asset_options.fix_base_link = True
    asset_handle_platform = gym.load_asset(sim, asset_root,
                                           asset_file_platform, asset_options)

    # Define camera properties and create Viewer object
    camera_props = gymapi.CameraProperties()
    camera_props.width = 1920
    camera_props.height = 1080

    # Set up camera angles if using graphical display (viewer)
    if use_viewer:
        viewer = gym.create_viewer(sim, camera_props)

        camera_target = gymapi.Vec3(0.0, 1.0, 0.0)
        camera_pos = gymapi.Vec3(-0, 1.02, 0.5)

        if args.object == "heart":
            camera_pos = gymapi.Vec3(-0, 1.02, 0.3)

        gym.viewer_camera_look_at(viewer, None, camera_pos, camera_target)
    else:
        viewer = None

    # Define transforms to convert between Trimesh and Isaac Gym conventions

    from_trimesh_transform = gymapi.Transform()
    from_trimesh_transform.r = gymapi.Quat(0, 0.7071068, 0,
                                           0.7071068)  # Rot_y(90), R_ba
    neg_rot_x_transform = gymapi.Transform()
    neg_rot_x = gymapi.Quat(0.7071068, 0, 0, -0.7071068)
    neg_rot_x_transform.r = neg_rot_x

    # Get 16 equally spaced vectors in a unit sphere
    all_directions, _, _, _ = uniform_sphere.get_uniform_directions_regular(16)
    num_directions = len(all_directions)
    all_directions = all_directions[args.ori_start:args.ori_end + 1]

    # Create environments, Franka actor, and object actor
    env_handles = []
    franka_handles = []
    object_handles = []
    platform_handles = []
    hand_origins = []

    env_spread = grasp_candidate_poses
    if args.mode.lower() in ["reorient", "shake", "twist"]:
        env_spread = all_directions

    for i, test_grasp_pose in enumerate(env_spread):
        if args.mode.lower() in ["reorient", "shake", "twist"]:
            test_grasp_pose = grasp_candidate_poses[0]

        # Create environment
        env_handle = gym.create_env(sim, env_lower, env_upper, envs_per_row)
        env_handles.append(env_handle)

        # Define shared pose/collision parameters
        pose = gymapi.Transform()
        grasp_transform = gymapi.Transform()
        grasp_transform.r = gymapi.Quat(test_grasp_pose[4], test_grasp_pose[5],
                                        test_grasp_pose[6], test_grasp_pose[3])
        identity_quat = gymapi.Quat(0., 0., 0., 1.)

        _, desired_rpy = metrics_features_utils.get_desired_rpy(
            identity_quat, grasp_transform.r)

        collision_group = i
        collision_filter = 0

        # Create Franka actors
        pose.p = gymapi.Vec3(test_grasp_pose[0], test_grasp_pose[1],
                             test_grasp_pose[2])
        pose.p = neg_rot_x_transform.transform_vector(pose.p)
        pose.p.y += PLATFORM_HEIGHT
        franka_handle = gym.create_actor(env_handle, asset_handle_franka, pose,
                                         f"franka_{i}", collision_group, 1)
        franka_handles.append(franka_handle)
        direction = np.array(
            [all_directions[i][1], all_directions[i][2],
             all_directions[i][0]])  # Read direction as y-up convention

        curr_joint_positions = gym.get_actor_dof_states(
            env_handle, franka_handle, gymapi.STATE_ALL)

        curr_joint_positions['pos'] = [
            0., 0., 0., 0., 0., 0., 0., desired_rpy[0], desired_rpy[1],
            desired_rpy[2], 0.0, 0.0, 0.0, 0, 0.04, 0.04
        ]

        curr_joint_positions['pos'] = np.zeros(16)

        twist_axis = np.array([0., 0., 1.])
        pose_transform = R.from_euler('ZYX', desired_rpy)
        twist_transform = R.align_vectors(np.expand_dims(direction, axis=0),
                                          np.expand_dims(twist_axis,
                                                         axis=0))[0]
        twist_eulers = twist_transform.as_euler('xyz')

        pose_correction = twist_transform.inv() * pose_transform
        pose_correction_euler = pose_correction.as_euler('xyz')

        # Correct for translation offset to match grasp
        q0 = np.array([0., 0., -0.112])
        q0_ = twist_transform.apply(q0)
        disp_offset = q0 - q0_

        curr_joint_positions['pos'] = [
            disp_offset[0], disp_offset[1], disp_offset[2], twist_eulers[2],
            twist_eulers[1], twist_eulers[0], 0., pose_correction_euler[2],
            pose_correction_euler[1], pose_correction_euler[0], 0.0, 0.0, 0.0,
            0, 0.04, 0.04
        ]

        hand_origin = pose
        hand_origins.append(hand_origin)
        finger_pose = gymapi.Transform()
        finger_pose.p = pose.p

        gym.set_actor_dof_states(env_handle, franka_handle,
                                 curr_joint_positions, gymapi.STATE_ALL)

        # Create soft object
        tet_file_name = os.path.join(object_path, args.object + ".tet")
        height_of_object = get_height_of_objects(tet_file_name)
        pose = gymapi.Transform()
        pose.r = neg_rot_x
        pose.p = from_trimesh_transform.transform_vector(
            gymapi.Vec3(0.0, 0.0, 0.0))

        object_height_buffer = 0.001
        pose.p.y += PLATFORM_HEIGHT + object_height_buffer

        object_handle = gym.create_actor(env_handle, asset_handle_object, pose,
                                         f"object_{i}", collision_group,
                                         collision_filter)
        object_handles.append(object_handle)

        # Create platform
        height_of_platform = 0.005
        pose.p.y -= (height_of_platform + object_height_buffer +
                     + 0.5 * height_of_object)
        platform_handle = gym.create_actor(env_handle, asset_handle_platform,
                                           pose, f"platform_{i}",
                                           collision_group, 1)
        platform_handles.append(platform_handle)

    # Run simulation and view results
    state = 'open'
    history = 10

    f_errs = np.ones(history, dtype=np.float32)
    panda_fsms = []
    directions = all_directions

    for i in range(len(env_handles)):
        if args.mode.lower() in ["reorient", "shake", "twist"]:
            test_grasp_pose = grasp_candidate_poses[0]
            directions = all_directions[i:i + 1]

        else:
            test_grasp_pose = env_spread[i]

        pure_grasp_transform = gymapi.Transform()
        pure_grasp_transform.r = gymapi.Quat(test_grasp_pose[4],
                                             test_grasp_pose[5],
                                             test_grasp_pose[6],
                                             test_grasp_pose[3])
        grasp_transform = gymapi.Transform()
        grasp_transform.r = neg_rot_x * gymapi.Quat(
            test_grasp_pose[4], test_grasp_pose[5], test_grasp_pose[6],
            test_grasp_pose[3])

        panda_fsm = pandafsm.PandaFsm(gym_handle=gym,
                                      sim_handle=sim,
                                      env_handles=env_handles,
                                      franka_handle=franka_handles[i],
                                      platform_handle=platform_handles[i],
                                      state=state,
                                      object_cof=sim_params.flex.dynamic_friction,
                                      f_errs=f_errs,
                                      grasp_transform=grasp_transform,
                                      obj_name=object_name,
                                      env_id=i,
                                      hand_origin=hand_origins[i],
                                      viewer=viewer,
                                      envs_per_row=envs_per_row,
                                      env_dim=env_dim,
                                      youngs=args.youngs,
                                      density=args.density,
                                      directions=np.asarray(directions),
                                      mode=args.mode.lower())
        panda_fsms.append(panda_fsm)

    # Make updating plot
    all_done = False
    loop_start = timeit.default_timer()

    while not all_done:

        # If it is taking too long, declare fail
        if (timeit.default_timer() - loop_start > 700
                and panda_fsms[i].state != 'reorient') or (
                    timeit.default_timer() - loop_start > 400
                    and panda_fsms[i].state == "squeeze_for_metric"):
            print("Timed out")
            for i in range(len(env_handles)):
                if panda_fsms[i].state != "done":
                    panda_fsms[i].state = "done"
                    panda_fsms[i].timed_out = True

        if use_viewer:
            pass

        for i in range(len(env_handles)):
            panda_fsms[i].update_previous_particle_state_tensor()

        all_done = all(panda_fsms[i].state == 'done'
                       for i in range(len(env_handles)))

        gym.refresh_particle_state_tensor(sim)
        for i in range(len(env_handles)):

            if panda_fsms[i].state != "done":
                panda_fsms[i].run_state_machine()

        # Run simulation
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.clear_lines(viewer)

        gym.step_graphics(sim)

        if use_viewer:
            gym.draw_viewer(viewer, sim, True)

    # Clean up
    if use_viewer:
        gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

    print("Finished the simulation")

    if write_results:
        metrics_features_utils.write_metrics_to_h5(args, num_grasp_poses,
                                                   num_directions,
                                                   h5_file_path, panda_fsms)
    return


if __name__ == "__main__":
    start_time = timeit.default_timer()
    main()
    print("Elapsed time", timeit.default_timer() - start_time)
