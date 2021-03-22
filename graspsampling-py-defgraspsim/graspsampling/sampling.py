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
"""Sampler classes."""

from abc import ABC, abstractmethod
import itertools
import sys

import trimesh
import trimesh.transformations as tra
import numpy as np

import logging

from . import utilities
from . import collision


class GraspSampler(ABC):
    """Super class for grasp samplers."""

    @abstractmethod
    def sample(self, **kwargs):
        """Sample a grasp."""
        pass

    def sequence_is_finished(self):
        """Return whether enough grasps are sampled."""
        return False


class UniformSampler(GraspSampler):
    """Uniform sampling."""

    def __init__(self, gripper, object_mesh):
        """Samples uniformly in SE3 space.

        Translations are bounded by the object's bounding box and the gripper's bounding sphere.
        Args:
            gripper (graspsampling.hands.Hand): Gripper.
            object_mesh (trimesh.Trimesh): Object mesh.
        """
        self.gripper = gripper
        self.object_mesh = object_mesh

    def sample(self, number_of_candidates):
        """Sample grasp poses.

        Args:
            number_of_candidates (int): Number of grasps to sample.

        Returns:
            dict: Contains grasp poses (7D: x, y, z, qx, qy, qz, qw).
        """
        lower_bound, upper_bound = utilities.get_gripper_object_bounds(
            self.gripper.mesh, self.object_mesh
        )

        # stretch to fit volume
        positions = np.random.rand(number_of_candidates, 3)
        positions = positions * (upper_bound - lower_bound) + lower_bound

        if False:
            orientations = np.array(
                [tra.random_quaternion() for i in range(number_of_candidates)]
            )
        else:
            orientations = utilities.random_quaternions(number_of_candidates)

        poses = np.hstack([positions, orientations])

        return {"poses": poses.tolist()}


class GridSampler(GraspSampler):
    """Samples lie on a grid in SE3 space.

    Translations are bounded by the object's bounding box and the gripper's bounding sphere.
    Args:
        gripper (graspsampling.hands.Hand): Gripper.
        object_mesh (trimesh.Trimesh): Object mesh.
        resolution_xyz (float): Resolution of translation (in meters).
        resolution_orientation (int): Resolution of orientation. See utilities.discretized_SO3.
    """

    def __init__(self, gripper, object_mesh, resolution_xyz, resolution_orientation):
        """Initialize attributes."""
        self.gripper = gripper
        self.object_mesh = object_mesh

        self.resolution_xyz = resolution_xyz
        self.resolution_orientation = resolution_orientation
        self.so3_discretization = utilities.discretized_SO3(resolution_orientation)

        self.sequence = 0

        gripper_size = np.abs(self.gripper.mesh.bounding_sphere.bounds).max()

        logging.debug(f"Radius of gripper's bounding sphere: {gripper_size}")
        logging.debug(f"Mesh bounds: {self.object_mesh.bounds}")
        logging.debug(f"Mesh extents: {self.object_mesh.extents}")

        x_minmax, y_minmax, z_minmax = np.split(
            self.object_mesh.bounds + [[-gripper_size], [gripper_size]], 3, axis=1
        )

        num_steps_x = int((x_minmax[1][0] - x_minmax[0][0]) // self.resolution_xyz)
        num_steps_y = int((y_minmax[1][0] - y_minmax[0][0]) // self.resolution_xyz)
        num_steps_z = int((z_minmax[1][0] - z_minmax[0][0]) // self.resolution_xyz)

        self.x = np.linspace(x_minmax[0][0], x_minmax[1][0], num_steps_x)
        self.y = np.linspace(y_minmax[0][0], y_minmax[1][0], num_steps_y)
        self.z = np.linspace(z_minmax[0][0], z_minmax[1][0], num_steps_z)

        logging.debug(
            f"Number of steps (x,y,z): {num_steps_x} {num_steps_y} {num_steps_z}"
        )
        logging.debug(f"Interval start (x,y,z): {self.x[0]} {self.y[0]} {self.z[0]}")
        logging.debug(
            f"Number of total poses: \
            {num_steps_x * num_steps_y * num_steps_z * self.resolution_orientation}"
        )

        self.sequence_length = (
            num_steps_x * num_steps_y * num_steps_z * self.resolution_orientation
        )

    def sample(self, number_of_samples):
        """Sample grasp poses.

        Args:
            number_of_grasps (int): Number of grasps to sample. Note, that this will be ignored!

        Returns:
            dict: Contains grasp poses (7D: x, y, z, qx, qy, qz, qw).
        """
        poses = []

        pose_iterator = itertools.product(
            self.x, self.y, self.z, self.so3_discretization
        )
        pose_iterator = itertools.islice(
            pose_iterator, self.sequence, self.sequence + number_of_samples
        )
        poses = np.array(list(pose_iterator))
        poses = np.hstack([poses[:, :3], np.stack(poses[:, 3])])

        logging.debug(f"Number of returned poses: {len(poses)}")

        self.sequence += number_of_samples

        logging.debug(f"Sequence number: {self.sequence}/{self.sequence_length}")

        return {"poses": poses.tolist()}

    def sequence_is_finished(self):
        """Return whether enough grasps are sampled."""
        return self.sequence >= self.sequence_length


class PolarCoordinateSampler(GraspSampler):
    """A grasp samppler used in the Hauser paper.

    A grasp sampler that aligns the gripper's approach vector with the lines of a polar c
    oordinate system originating at the object's center of mass/centroid.

    Args:
        gripper (graspsampling.hands.Hand): Gripper.
        object_mesh (trimesh.Trimesh): Object mesh.
        step_size (float, optional): [description]. Defaults to 0.05.
        orientation_towards_com (float, optional): [description]. Defaults to np.radians(30.0).
    """

    def __init__(
        self,
        gripper,
        object_mesh,
        step_size=0.05,
        orientation_towards_com=np.radians(30.0),
    ):
        """Initialize attributes."""
        self.gripper = gripper
        self.object_mesh = object_mesh

        self.step_size = step_size
        self.orientation_towards_com = orientation_towards_com

    def sample(self, number_of_grasps):
        """Sample grasp poses.

        Args:
            number_of_grasps (int): Number of grasps to sample.

        Returns:
            dict: Contains grasp poses (7D: x, y, z, qx, qy, qz, qw).
        """
        # get bounding box for the boundaries of polar coordinates
        lower_bound, upper_bound = utilities.get_gripper_object_bounds(
            self.gripper.mesh, self.object_mesh
        )
        bbox = trimesh.primitives.Box(
            extents=(upper_bound - lower_bound)[0],
            transform=self.object_mesh.bounding_box.primitive.transform,
        )

        avg_ray_length = np.mean((upper_bound - lower_bound)[0] / 2.0)
        number_of_rays = int(number_of_grasps * self.step_size / avg_ray_length)
        logging.debug("Number of rays:", number_of_rays)

        while True:
            directions = utilities.sample_random_direction_R3(number_of_rays)
            ray_hits = collision.raycasts(
                np.tile(self.object_mesh.center_mass, (number_of_rays, 1)),
                directions,
                bbox,
            )
            if len(ray_hits) == number_of_rays:
                break

        ray_lengths = np.linalg.norm(ray_hits - self.object_mesh.center_mass, axis=1)
        ray_starts = np.random.uniform(0, self.step_size, number_of_rays)

        positions = []
        orientations = []
        for ray_start, ray_end, ray in zip(ray_starts, ray_lengths, directions):
            steps = np.arange(ray_start, ray_end, self.step_size)

            pos = ray * steps[:, np.newaxis]
            pos += self.object_mesh.center_mass

            if self.orientation_towards_com >= np.pi:
                ori = [tra.random_quaternion()] * len(steps)
            else:
                ori = [
                    tra.quaternion_from_matrix(
                        utilities.sample_random_orientation_z(
                            [-ray], self.orientation_towards_com
                        )
                    )
                ] * len(steps)

        positions.extend(pos)
        orientations.extend(ori)

        # multiple magnitudes per ray
        # np.random.uniform(-step_size*0.5, step_size*0.5, number_of_candidates)
        # np.arange(0, ray_lengths)

        # positions = directions * magnitudes[:, np.newaxis]
        # orientations = rand_quats(number_of_candidates)

        poses = np.concatenate([positions, orientations], axis=-1)

        return {"poses": poses.tolist()}


class SurfaceApproachSampler(GraspSampler):
    """A grasp sampler that aligns the gripper's approach vector with the object surface.

    Args:
        gripper (graspsampling.hands.Hand): Gripper.
        object_mesh (trimesh.Trimesh): Object mesh.
        surface_normal_cone (float, optional): [description]. Defaults to 0.0.
        approach_cone (float, optional): [description]. Defaults to 0.0.
    """

    def __init__(
        self, gripper, object_mesh, surface_normal_cone=0.0, approach_cone=0.0
    ):
        """Iniitialize attributes."""
        self.surface_normal_cone = surface_normal_cone
        self.approach_cone = approach_cone

        self.gripper = gripper
        self.object_mesh = object_mesh

    def sample(self, number_of_grasps):
        """Sample grasp poses.

        Args:
            number_of_grasps (int): Number of grasps to sample.

        Raises:
            NotImplementedError: Some parameter values (approach cone > 0)
            are not yet implemented.

        Returns:
            dict: Contains grasp poses (7D: x, y, z, qx, qy, qz, qw).
        """
        mesh_points, face_indices = self.object_mesh.sample(
            number_of_grasps, return_index=True
        )
        mesh_normals = self.object_mesh.face_normals[face_indices]

        if self.surface_normal_cone > 0.0:
            # perturb normals randomly
            normals = utilities.sample_spherical_cap(
                mesh_normals, self.surface_normal_cone, num_samples_per_dir=1
            )
        else:
            normals = mesh_normals

        # sample random standoffs
        logging.debug(
            "Standoff: %f, %f"
            % (self.gripper.standoff_range[0], self.gripper.standoff_range[1])
        )
        standoffs = np.random.uniform(
            self.gripper.standoff_range[0],
            self.gripper.standoff_range[1],
            size=(number_of_grasps, 1),
        )

        positions = mesh_points + normals * standoffs

        surface_orientations = np.array(
            [trimesh.geometry.align_vectors([0, 0, -1], normal) for normal in normals]
        )

        roll_orientations = np.array(
            [
                tra.quaternion_about_axis(angle, [0, 0, 1])
                for angle in np.random.rand(number_of_grasps) * 2.0 * np.pi
            ]
        )
        roll_orientations = np.array(
            [tra.quaternion_matrix(q) for q in roll_orientations]
        )
        if self.approach_cone > 0.0:
            raise NotImplementedError(
                "Feature is not implemented! (sampling in approach_cone > 0)"
            )

        orientations = np.array(
            [np.dot(s, r) for s, r in zip(surface_orientations, roll_orientations)]
        )
        orientations_q = np.array(
            [tra.quaternion_from_matrix(m, isprecise=True) for m in orientations]
        )

        poses = np.hstack([positions, orientations_q])

        return {"poses": poses.tolist()}


class AntipodalSampler(GraspSampler):
    """A grasp sampler that aligns the gripper's approach vector with the object surface.

    Args:
        gripper (graspsampling.hands.Hand): Gripper.
        object_mesh (trimesh.Trimesh): Object mesh.
        friction_cone (float, optional): Half of the aperture of the
            friction cone (in radians). Defaults to 0.0.
        number_of_orientations (int, optional): How many orientations to evenly sample
             around the grasp axis. Defaults to 1.
        bite (float, optional): Additional distance added/subtracted from the fingertip
            standoffs of the gripper. Defaults to 0.0.
    """

    def __init__(
        self,
        gripper,
        object_mesh,
        friction_cone=0.0,
        number_of_orientations=1,
        bite=0.0,
        mirror_grasp=False,
    ):
        """Initialize attributes."""
        self.gripper = gripper
        self.object_mesh = object_mesh

        if trimesh.ray.has_embree:
            logging.debug("Using trimesh.ray.ray_pyembree.RayMeshIntersector")
            self.object_intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(
                self.object_mesh, scale_to_box=True
            )
        else:
            logging.debug(
                "Using trimesh.ray.ray_triangle.RayMeshIntersector since embree is not found."
            )
            self.object_intersector = trimesh.ray.ray_triangle.RayMeshIntersector(
                self.object_mesh
            )

        self.friction_cone = friction_cone
        self.number_of_orientations = number_of_orientations
        self.bite = bite
        self.mirror_grasp = mirror_grasp

    def sample(self, number_of_grasps):
        """TODO: CHANGE THIS.
        Samples antipodal pairs using rejection sampling.
        The proposal sampling ditribution is to choose a random point on
        the object surface, then sample random directions within the friction cone,
        then form a grasp axis along the direction,
        close the fingers, and keep the grasp if the other contact point is also in the
        friction cone.
        """

        gripper_width = self.gripper.maximum_aperture

        # get mesh points
        batch_size = max(1, int(number_of_grasps // self.number_of_orientations))
        tmp_points, face_indices = self.object_mesh.sample(
            batch_size, return_index=True
        )
        tmp_normals = self.object_mesh.face_normals[face_indices]

        # (TODO:) Check if mesh point valid (see weird ShapeNet models)

        # get mesh normals and cast ray (in opposite direction)
        if self.friction_cone > 0:
            # TODO: how is friction coefficient related?
            axis = utilities.sample_spherical_cap(
                -tmp_normals, self.friction_cone, num_samples_per_dir=1
            )
        else:
            axis = -tmp_normals
        locations, index_ray, _ = self.object_intersector.intersects_location(
            tmp_points, axis, multiple_hits=True
        )

        candidates = []
        candidates_axis = []
        # and choose most distant point
        for i in range(batch_size):
            # TODO: find out which is location that is farthest away from tmp_point
            # SOMETHING LIKE THIS
            #     # since we are not returning multiple hits, we need to
            #     # figure out which hit is first
            #     if len(index_ray) == 0:
            #         return index_tri, index_ray, location

            #     first = np.zeros(len(index_ray), dtype=np.bool)
            #     groups = grouping.group(index_ray)
            #     for group in groups:
            #         index = group[distance[group].argmin()]
            #         first[index] = True
            # return index_tri[first], index_ray[first], location[first]
            try:
                opponent = locations[np.where(index_ray == i)][-1]

                axis_vec = opponent - tmp_points[i]
                dist = np.linalg.norm(axis_vec)

                # check if below gripper_width
                if dist > trimesh.tol.zero and dist <= gripper_width:
                    candidates.append(tmp_points[i] + axis_vec * 0.5)
                    candidates_axis.append(axis_vec)
            except IndexError:
                logging.error("Error finding opposing contact.")

        # discretize around axis
        pitch_angles = np.linspace(
            -np.pi, np.pi, self.number_of_orientations, endpoint=False
        )

        # distance of contact axis to gripper origin
        # TODO: this is gripper-specific
        standoff = self.gripper.standoff_fingertips + self.bite
        pitch_tfs = [
            tra.rotation_matrix(angle=x, direction=[1, 0, 0]).dot(
                tra.translation_matrix([0, 0, -standoff])
            )
            for x in pitch_angles
        ]

        cand_tfs = [
            tra.translation_matrix(x)
            .dot(trimesh.geometry.align_vectors([1, 0, 0], np.copy(al)))
            .dot(y)
            .dot(self.gripper.offset_inv)
            for x, al in zip(candidates, candidates_axis)
            for y in pitch_tfs
        ]

        if self.mirror_grasp:
            cand_tfs += [tf @ tra.euler_matrix(0, 0, np.pi) for tf in cand_tfs]

        # convert to poses
        poses = utilities.mats_to_poses_wxyz(cand_tfs)

        return {"poses": poses}


def collision_free_grasps(
    gripper,
    object_mesh,
    sampler,
    number_of_grasps,
    env_mesh=None,
    max_attempts=sys.maxsize,
):
    """Return grasps that are not in collision with the object."""
    results = []
    num_attempts = 0
    while (
        len(results) < number_of_grasps
        and not sampler.sequence_is_finished()
        and num_attempts < max_attempts
    ):
        # sample grasps
        tmp = sampler.sample(number_of_grasps)
        tmp_len = len(tmp["poses"])
        num_attempts += number_of_grasps

        # test for collisions and content of closing region
        if env_mesh:
            in_collision = collision.in_collision_with_gripper(gripper, env_mesh, **tmp)
        else:
            in_collision = collision.in_collision_with_gripper(
                gripper, object_mesh, **tmp
            )
        closing_region_nonempty = collision.check_gripper_nonempty(
            gripper, object_mesh, **tmp
        )

        logging.info(
            f"Collision-free: {sum(~in_collision)}/{tmp_len},  \
            Non-empty closing region: {sum(closing_region_nonempty)}/{tmp_len},   \
            Both: {sum(closing_region_nonempty & ~in_collision)}"
        )

        grasps = np.array(tmp["poses"])
        results.extend(grasps[closing_region_nonempty & ~in_collision].tolist())

    return results[:number_of_grasps]
