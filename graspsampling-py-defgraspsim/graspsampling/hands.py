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
"""Represent various robotic gripper hands."""

import os
import numpy as np

import trimesh
import trimesh.transformations as tra

from . import utilities

try:
    from trimesh.collision import fcl

    fcl_import_failed = False
except Exception:
    fcl_import_failed = True


# TODO: inheritance
class Hand(object):
    """Super class for hands."""
    def __init__(self):
        """Initialize offset."""
        self.offset = np.eye(4)

    def get_closing_rays(self, transform):
        """Closing rays."""
        return (
            transform[:3, :].dot(self.ray_origins.T).T,
            transform[:3, :3].dot(self.ray_directions.T).T,
        )

    def get_obbs(self):
        """Get oriented bounding boxes."""
        return [
            self.finger_l.bounding_box,
            self.finger_r.bounding_box,
            self.base.bounding_box,
        ]

    @property
    def offset(self):
        """Return offset."""
        return self._offset

    @property
    def offset_inv(self):
        """Return inverse of offset."""
        return self._offset_inv

    @offset.setter
    def offset(self, value):
        """Set offset and inverse offset."""
        self._offset = value
        self._offset_inv = tra.inverse_matrix(value)

    def show(self, show_rays=False):
        """Visualize hand."""
        if show_rays:
            a, b = self.get_closing_rays(np.eye(4))
            rays_as_points = []
            for x in np.linspace(0, 0.03, 20):
                rays_as_points.append(
                    trimesh.points.PointCloud(vertices=a + b * x, colors=[255, 0, 0])
                )
            trimesh.Scene([self.mesh] + rays_as_points).show()
        else:
            self.mesh.show()


class PandaGripper(Hand):
    """Class for the Panda gripper."""
    def __init__(
        self,
        configuration=None,
        num_contact_points_per_finger=10,
        finger_mesh_filename="data/hands/panda_gripper/finger.stl",
        palm_mesh_filename="data/hands/panda_gripper/hand.stl",
        offset=np.eye(4),
        finger_scale=[1.0, 1.0, 1.0],
    ):
        """Initialize attributes."""
        self.joint_limits = [0.0, 0.04]
        self.default_pregrasp_configuration = 0.0

        self.maximum_aperture = 0.08
        self.standoff_fingertips = 0.1
        self.offset = offset

        self.closing_region = trimesh.primitives.creation.box(
            extents=[0.08, 0.01, 0.04],
            transform=tra.translation_matrix([0.0, 0.0, 0.09]),
        )

        if configuration is None:
            configuration = self.default_pregrasp_configuration

        self.configuration = configuration
        res_path = utilities.get_resource_path()
        fn_base = os.path.join(res_path, palm_mesh_filename)
        fn_finger = os.path.join(res_path, finger_mesh_filename)
        self.base = trimesh.load(fn_base)

        # After API change:
        # https://github.com/mikedh/trimesh/issues/507
        if isinstance(self.base, trimesh.scene.Scene):
            self.base = self.base.dump().tolist()
            self.base = trimesh.util.concatenate(self.base)

        if isinstance(self.base, list) and len(self.base) == 5:
            for i in range(len(self.base)):
                self.base[i].visual = trimesh.visual.ColorVisuals()
                for facet in self.base[i].facets:
                    self.base.visual.face_colors[facet] = trimesh.visual.random_color()

            self.base = trimesh.util.concatenate(self.base)

        self.finger_l = trimesh.load(fn_finger)
        self.finger_l.apply_scale(finger_scale)

        # After API change:
        # https://github.com/mikedh/trimesh/issues/507
        if isinstance(self.finger_l, trimesh.scene.Scene):
            self.finger_l = self.finger_l.dump().tolist()

        if isinstance(self.finger_l, list) and len(self.finger_l) == 2:
            for i in range(len(self.finger_l)):
                self.finger_l[i].visual = trimesh.visual.ColorVisuals()

            # finger - silver
            self.finger_l[0].visual.face_colors[:] = np.array(
                [192, 192, 192, 255], dtype=np.uint8
            )
            # fingertip - black
            self.finger_l[1].visual.face_colors[:] = np.array(
                [9, 9, 9, 255], dtype=np.uint8
            )

            self.finger_l = trimesh.util.concatenate(self.finger_l)

        self.finger_r = self.finger_l.copy()

        # transform fingers relative to the base
        self.finger_l.apply_transform(tra.euler_matrix(0, 0, np.pi))
        self.finger_l.apply_translation([+configuration, 0, 0.0584])
        self.finger_r.apply_translation([-configuration, 0, 0.0584])

        # generate fcl collision geometry
        if not fcl_import_failed and fcl:
            self.fcl_objs = [
                fcl.CollisionObject(
                    fcl.Box(*part.bounding_box.primitive.extents),
                    utilities.numpy_to_fcl_transform(
                        offset @ part.bounding_box.primitive.transform
                    ),
                )
                for part in [self.finger_l, self.finger_r, self.base]
            ]
            self.fcl_transforms = [
                offset @ part.bounding_box.primitive.transform
                for part in [self.finger_l, self.finger_r, self.base]
            ]

        # generate rays for heuristics and contact tests
        self.ray_origins = []
        self.ray_directions = []
        for i in np.linspace(
            -0.01 * finger_scale[-1],
            0.02 * finger_scale[-1],
            num_contact_points_per_finger,
        ):
            self.ray_origins.append(
                offset @ np.r_[self.finger_l.bounding_box.centroid + [0, 0, i], 1]
            )
            self.ray_origins.append(
                offset @ np.r_[self.finger_r.bounding_box.centroid + [0, 0, i], 1]
            )
            tmp = (
                offset
                @ np.r_[-self.finger_l.bounding_box.primitive.transform[:3, 0], 1]
            )
            self.ray_directions.append(tmp[:3])
            tmp = (
                offset
                @ np.r_[+self.finger_r.bounding_box.primitive.transform[:3, 0], 1]
            )
            self.ray_directions.append(tmp[:3])

        self.ray_origins = np.array(self.ray_origins)
        self.ray_directions = np.array(self.ray_directions)

        # transform according to offset
        self.base.apply_transform(offset)
        self.closing_region.apply_transform(offset)
        self.finger_l.apply_transform(offset)
        self.finger_r.apply_transform(offset)

        self.fingers = trimesh.util.concatenate([self.finger_l, self.finger_r])
        self.mesh = trimesh.util.concatenate([self.fingers, self.base])

        self.standoff_range = np.array(
            [
                max(
                    self.finger_l.bounding_box.bounds[0, 2],
                    self.base.bounding_box.bounds[1, 2],
                ),
                self.finger_l.bounding_box.bounds[1, 2],
            ]
        )
        self.standoff_range[0] += 0.001

    def get_obbs(self):
        """Get oriented bounding boxes."""
        return [
            self.finger_l.bounding_box,
            self.finger_r.bounding_box,
            self.base.bounding_box,
        ]

    def get_fcl_collision_objects(self):
        """Get objects in collision."""
        return self.fcl_objs

    def get_fcl_transforms(self):
        """Get fcl transforms."""
        return self.fcl_transforms


available_grippers = {
    "panda": {"cls": PandaGripper, "params": {}},
    "panda_original": {
        "cls": PandaGripper,
        "params": {
            "offset": tra.rotation_matrix(-np.pi / 2.0, [0, 0, 1]),
        },
    },
    "panda_franka_link7": {
        "cls": PandaGripper,
        "params": {
            "offset": tra.compose_matrix(
                angles=[0, 0, -0.75 * np.pi], translate=[0, 0, 0.107]
            ),
        },
    },
    "panda_franka_link7_longfingers": {
        "cls": PandaGripper,
        "params": {
            "offset": tra.compose_matrix(
                angles=[0, 0, -0.75 * np.pi], translate=[0, 0, 0.107]
            ),
            "finger_scale": [1.0, 1.0, 1.75],
        },
    },
    "panda_original_longfingers": {
        "cls": PandaGripper,
        "params": {
            "offset": tra.rotation_matrix(-np.pi / 2.0, [0, 0, 1]),
            "finger_scale": [1.0, 1.0, 1.75],
        },
    },
    "panda_visual": {
        "cls": PandaGripper,
        "params": {
            "finger_mesh_filename": "data/hands/panda_gripper/visual/finger_detail.stl",
            "palm_mesh_filename": "data/hands/panda_gripper/visual/hand_detail.stl",
        },
    },
    "panda_visual_colored": {
        "cls": PandaGripper,
        "params": {
            "finger_mesh_filename": "data/hands/panda_gripper/visual/finger_detail.obj",
            "palm_mesh_filename": "data/hands/panda_gripper/visual/hand_detail.obj",
        },
    },
    "panda_tube": {
        "cls": PandaGripper,
        "params": {
            "finger_mesh_filename": "data/hands/panda_gripper/visual/finger_tube.stl",
            "palm_mesh_filename": "data/hands/panda_gripper/visual/base_tube.stl",
        },
    },
    "panda_tube_franka_link7": {
        "cls": PandaGripper,
        "params": {
            "finger_mesh_filename": "data/hands/panda_gripper/visual/finger_tube.stl",
            "palm_mesh_filename": "data/hands/panda_gripper/visual/base_tube.stl",
            "offset": tra.compose_matrix(
                angles=[0, 0, -0.75 * np.pi], translate=[0, 0, 0.107]
            ),
        },
    },
    "panda_tube_franka_link7_longfingers": {
        "cls": PandaGripper,
        "params": {
            "finger_mesh_filename": "data/hands/panda_gripper/visual/finger_tube.stl",
            "palm_mesh_filename": "data/hands/panda_gripper/visual/base_tube.stl",
            "offset": tra.compose_matrix(
                angles=[0, 0, -0.75 * np.pi], translate=[0, 0, 0.107]
            ),
            "finger_scale": [1.0, 1.0, 1.75],
        },
    },
}


def get_available_grippers():
    """Get available grippers."""
    return list(available_grippers.keys())


def get_gripper_name(gripper):
    """Get gripper name."""
    for k, v in available_grippers.items():
        if isinstance(gripper, v["cls"]):  # TODO: check meshes
            return k


def create_gripper(name, configuration=None):
    """Create a gripper."""
    cfg = available_grippers[name.lower()]
    return cfg["cls"](configuration=configuration, **cfg["params"])
