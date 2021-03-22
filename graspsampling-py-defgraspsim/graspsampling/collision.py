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
"""Detect collisions between gripper and object."""

import numpy as np
import trimesh

from . import utilities


def in_collision_with_gripper(gripper, object_mesh, poses):
    """Check if gripper is in collision with object mesh in different poses.

    Args:
        gripper (hands.Hand): The gripper/hand.
        object_mesh (trimesh.Trimesh): The object mesh.
        poses (np.array): Nx7 array of gripper poses.

    Returns:
        np.array (dtype=np.bool): Nx1 array of collision check results.
    """
    # gripper_meshes = gripper.get_obbs()
    gripper_meshes = [gripper.mesh]

    gripper_transforms = np.array(utilities.poses_wxyz_to_mats(poses))

    return _in_collision(object_mesh, gripper_meshes, gripper_transforms)


def check_gripper_nonempty(gripper, object_mesh, poses):
    """Check if object collides with gripper's closing region."""
    gripper_transforms = np.array(utilities.poses_wxyz_to_mats(poses))

    return _in_collision(object_mesh, gripper.closing_region, gripper_transforms)


def _in_collision(mesh_a, mesh_b, transforms_b):
    """Check for collision between mesh_a and mesh_b under transforms_b."""
    manager = trimesh.collision.CollisionManager()
    manager.add_object("object", mesh_a)

    if isinstance(mesh_b, list):
        in_collision = [
            any([manager.in_collision_single(m_b, transform=t_b) for m_b in mesh_b])
            for t_b in transforms_b
        ]
    else:
        in_collision = [
            manager.in_collision_single(mesh_b, transform=t_b) for t_b in transforms_b
        ]

    return np.array(in_collision, dtype=np.bool)


def raycasts(origins, directions, mesh):
    assert len(origins) == len(directions)
    if trimesh.ray.has_embree:
        intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(
            mesh, scale_to_box=True
        )
    else:
        intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)

    locations, _, _ = intersector.intersects_location(
        origins, directions, multiple_hits=False
    )
    return locations
