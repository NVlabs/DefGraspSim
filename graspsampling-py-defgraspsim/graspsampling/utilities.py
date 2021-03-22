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
"""Helper functions for grasp sampling."""

import trimesh
import trimesh.transformations as tra
import numpy as np
import os

try:
    from trimesh.collision import fcl

    fcl_import_failed = False
except Exception:
    fcl_import_failed = True


def sample_spherical_cap(cone_dirs, cone_aperture, num_samples_per_dir=1):
    """Uniformly distributed points on a spherical cap (sphere radius = 1).

    Args:
        cone_dirs (np.array): Nx3 array that represents cone directions.
        cone_aperture (float): Aperture of cones / size of spherical cap.
        num_samples_per_dir (int, optional): Number of samples to draw per direction. Defaults to 1.

    Raises:
        NotImplementedError: [description]

    Returns:
        np.array: Nx3 array of sampled points.
    """
    # sample around north pole
    if num_samples_per_dir > 1:
        raise NotImplementedError("num_samples_per_dir > 1 is not implemented")
    num_samples = len(cone_dirs) * num_samples_per_dir
    z = np.random.rand(num_samples) * (1.0 - np.cos(cone_aperture)) + np.cos(
        cone_aperture
    )
    phi = np.random.rand(num_samples) * 2.0 * np.pi
    x = np.sqrt(1.0 - np.power(z, 2)) * np.cos(phi)
    y = np.sqrt(1.0 - np.power(z, 2)) * np.sin(phi)

    points = np.vstack([x, y, z]).T
    points = points[..., np.newaxis]

    transforms = np.array(
        [
            trimesh.geometry.align_vectors([0, 0, 1], cone_dir)[:3, :3]
            for cone_dir in cone_dirs
        ]
    )

    result = np.matmul(transforms, points)
    return np.squeeze(result, axis=2)


def sample_random_orientation_z(mean_axis_z, axis_cone_aperture):
    """Sample a random orientation around a cap defined by mean_axis_z."""
    z_axis = sample_spherical_cap(
        mean_axis_z, axis_cone_aperture, num_samples_per_dir=1
    )[0]
    while True:
        r = sample_random_direction_R3(1)[0]
        # check if collinear
        if abs(z_axis.dot(r)) < (1.0 - 1e-2):
            break
    y_axis = np.cross(z_axis, r)
    x_axis = np.cross(y_axis, z_axis)
    orientation = np.eye(4)
    orientation[:3, 0] = x_axis
    orientation[:3, 1] = y_axis
    orientation[:3, 2] = z_axis

    return orientation


def random_quaternions(size):
    """Generate random quaternions, uniformly distributed on SO(3).

    See: http://planning.cs.uiuc.edu/node198.html
    Args:
        size (int): Number of quaternions.

    Returns:
        np.array: sizex4 array of quaternions in w-x-y-z format.
    """
    u = np.random.rand(size, 3)

    r1 = np.sqrt(1.0 - u[:, 0])
    r2 = np.sqrt(u[:, 0])

    t = 2.0 * np.pi * u[:, 1:]

    qw = np.cos(t[:, 1]) * r2
    qx = np.sin(t[:, 0]) * r1
    qy = np.cos(t[:, 0]) * r1
    qz = np.sin(t[:, 1]) * r2

    return np.vstack([qw, qx, qy, qz]).T


def sample_random_direction_R3(number_of_directions):
    """Uniformly distributed directions on S2.

    Sampled from a multivariate Gaussian, followed by normalization.
    Args:
        number_of_directions (int): Number of directions to sample.

    Returns:
        np.array: number_of_directionsx3 array of directions.
    """
    # sample multivariate Gaussian and normalize
    dir = np.random.normal(0, 1, (number_of_directions, 3))
    dir = dir / np.linalg.norm(dir, axis=1)[:, np.newaxis]

    return dir


def discretized_SO3(resolution):
    """Return an array of quaternions that are equidistant on SO(3).

    Args:
        resolution (int): A number in {72, 576, 4608, 36864}.

    Raises:
        ValueError: Argument represents an unknown resolution.

    Returns:
        np.array: Nx4 array of quaternions (in w-x-y-z format)
    """
    available_resolutions = [72, 576, 4608, 36864]
    if resolution not in available_resolutions:
        raise ValueError(
            f"SO3 resolution {resolution} unknown. Available resolutions: \
            {', '.join([str(x) for x in available_resolutions])}"
        )

    res_path = get_resource_path("data/discretizations")
    res_name = os.path.join(res_path, f"so3_{int(resolution)}_quaternionxyzw.npy")
    quaternions = np.load(res_name)
    return quats_xyzw_to_wxyz(quaternions)


def numpy_to_fcl_transform(arr):
    """Convert numpy matrix to fcl transform."""
    return fcl.Transform(arr[:3, :3], arr[:3, 3])


def fcl_transform_to_numpy(arr):
    """Convert fcl transform to numpy matrix."""
    ret = np.eye(4)
    ret[:3, :3] = arr.getRotation()
    ret[:3, 3] = arr.getTranslation()
    return ret


def mat_to_pose_wxyz(mat):
    """Convert matrix to pos and wxyz quaternion."""
    p = mat[:3, 3].tolist()
    p += tra.quaternion_from_matrix(mat).tolist()
    return np.array(p)


def mat_to_pose_xyzw(mat):
    """Convert matrix to pos and xyzw quaternion."""
    p = mat[:3, 3].tolist()
    p += np.roll(tra.quaternion_from_matrix(mat), -1).tolist()
    return np.array(p)


def pose_wxyz_to_mat(p):
    """Convert pos and wxyz quaternion to matrix."""
    tmp = tra.quaternion_matrix(p[3:])
    tmp[:3, 3] = p[:3]
    return tmp


def pose_xyzw_to_mat(p):
    """Convert pos and xyzw quaternion to matrix."""
    tmp = tra.quaternion_matrix(np.roll(p[3:], +1))
    tmp[:3, 3] = p[:3]
    return tmp


def poses_xyzw_to_mats(poses):
    """Convert multiple pos and xyzw quaternion to matrices."""
    mats = []
    for p in poses:
        # convert each transform to a pose
        mat = pose_xyzw_to_mat(np.asarray(p))
        mats.append(mat.tolist())
    return mats


def poses_wxyz_to_mats(poses):
    """Convert multiple pos and wxyz quaternion to matrices."""
    mats = []
    for p in poses:
        mat = pose_wxyz_to_mat(np.asarray(p))
        mats.append(mat.tolist())
    return mats


def quats_xyzw_to_wxyz(q):
    """Convert from xyzw to wxyz quaternions."""
    return np.roll(q, 1, axis=1)


def quats_wxyz_to_xyzw(q):
    """Convert from wxyz to xyzw quaternions."""
    return np.roll(q, -1, axis=1)


def pose_xyzw_to_wxyz(p):
    """Convert from pose and xyzw to wxyz quatenrions."""
    tmp = p[:3].tolist()
    tmp += np.roll(p[3:], +1).tolist()
    return np.array(tmp)


# the main convention is: w - x - y - z
def pose_wxyz_to_xyzw(p):
    """Convert from pose and wxyz to xyzw quaternions."""
    tmp = p[:3].tolist()
    tmp += np.roll(p[3:], -1).tolist()
    return np.array(tmp)


def mats_to_poses_xyzw(transforms):
    """Convert matrices to pos and xyzw quaternions."""
    poses = []
    for t in transforms:
        # convert each transform to a pose
        pose = mat_to_pose_xyzw(np.asarray(t))
        poses.append(pose.tolist())

    return poses


def mats_to_poses_wxyz(transforms):
    """Convert matrices to pos and wxyz quaternions."""
    poses = []
    for t in transforms:
        # convert each transform to a pose
        pose = mat_to_pose_wxyz(np.asarray(t))
        poses.append(pose.tolist())

    return poses


def get_gripper_object_bounds(gripper_mesh, object_mesh):
    """Get bounds of object with gripper."""
    gripper_size = np.abs(gripper_mesh.bounding_sphere.bounds).max()

    lower_bound, upper_bound = np.split(
        object_mesh.bounds + [[-gripper_size], [gripper_size]], 2, axis=0
    )
    return lower_bound, upper_bound


def get_resource_path(path=""):
    """Get path to resouce."""
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), "../", path)


def instantiate_mesh(**kwargs):
    """Instantiate scaled mesh."""
    fname = get_resource_path(kwargs["file"])
    mesh = trimesh.load(fname)
    mesh.apply_scale(kwargs["scale"])
    return mesh
