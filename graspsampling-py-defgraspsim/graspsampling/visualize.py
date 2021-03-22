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
"""Create scene for visualizing grasps on an object."""

import trimesh
import numpy as np

from . import hands
from . import utilities


def create_scene(object_mesh, gripper_name, **kwargs):
    """Create a trimesh scene object populated with the object and a set of grasps.

    Args:
        object_mesh (trimesh.Trimesh): Mesh of the object to be grasped.
        gripper_name (str): Type of the gripper.
        poses (np.array): Grasp poses.
        qualities (np.array): Qualities.

    Returns:
        trimesh.scene.Scene: A scene that can be visualized via trimesh.scene.Scene.show().
    """
    scene = trimesh.Scene([object_mesh])

    gripper = hands.create_gripper(gripper_name, 0.04)

    transformation_matrices = utilities.poses_wxyz_to_mats(kwargs["poses"])

    if "qualities" in kwargs:
        qualities = kwargs["qualities"]
    else:
        qualities = len(transformation_matrices) * [1.0]

    for quality, transform in zip(qualities, transformation_matrices):
        mesh = gripper.mesh.copy()
        mesh.apply_transform(transform)
        mesh.visual.face_colors[:, :3] = quality * np.array([0, 255, 0])

        scene.add_geometry(mesh)

    return scene
