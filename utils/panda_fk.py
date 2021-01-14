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
"""Helper functions for forward kinematics of the Panda gripper."""

import numpy as np
from scipy.linalg import expm


def skew(a):
    """Return the skew-symmetric matrix of a vector."""
    return np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])


def wedge(xi):
    """Return the matrix form of a twist vector."""
    v, w = xi[:3], xi[-3:]
    a = np.zeros((4, 4))
    a[:3, :3] = skew(w)
    a[:3, 3] = v
    return a


def adjoint(g):
    """Return the adjoint of a rigid body transformation g."""
    adg = np.zeros((6, 6))
    R_part, p = g[:3, :3], g[:3, 3]
    pR = skew(p) @ R_part
    adg[:3, :3] = R_part
    adg[-3:, -3:] = R_part
    adg[:3, -3:] = pR
    return adg


def jacobian(full_joints, hand_origin):
    """Return the robot spatial Jacobian."""
    list_xis, list_transforms = get_fk(full_joints, hand_origin, mode="all")
    list_transforms.insert(0, np.eye(4))
    joints = full_joints[:6]
    J = np.zeros((6, len(joints)))
    curr_trans = np.eye(4)
    for j in range(len(joints)):
        curr_trans = curr_trans @ list_transforms[j]
        xi_prime = adjoint(curr_trans) @ list_xis[j]
        J[:, j] = xi_prime
    return J


def get_fk(joints, hand_origin, mode="left"):
    """Get the forward kinematics of the hand from joint angles."""
    hand_origin_pos = np.array(
        [hand_origin.p.x, hand_origin.p.y, hand_origin.p.z])
    finger_origin_pos = np.array(
        [hand_origin.p.x, hand_origin.p.y, hand_origin.p.z + 0.112])

    # Prismatic joints
    xi_x = np.array([1, 0, 0, 0, 0, 0])
    xi_y = np.array([0, 1, 0, 0, 0, 0])
    xi_z = np.array([0, 0, 1, 0, 0, 0])

    # Joints at the gripper base
    xi_rev_z = np.concatenate(
        [np.cross(hand_origin_pos, np.array([0, 0, 1])),
         np.array([0, 0, 1])])
    xi_rev_y = np.concatenate(
        [np.cross(hand_origin_pos, np.array([0, 1, 0])),
         np.array([0, 1, 0])])
    xi_rev_x = np.concatenate(
        [np.cross(hand_origin_pos, np.array([1, 0, 0])),
         np.array([1, 0, 0])])

    # Joints at the gripper fingers
    xi_rev_z2 = np.concatenate([
        np.cross(finger_origin_pos, np.array([0, 0, 1])),
        np.array([0, 0, 1])
    ])
    xi_rev_y2 = np.concatenate([
        np.cross(finger_origin_pos, np.array([0, 1, 0])),
        np.array([0, 1, 0])
    ])
    xi_rev_x2 = np.concatenate([
        np.cross(finger_origin_pos, np.array([1, 0, 0])),
        np.array([1, 0, 0])
    ])

    xi4 = np.array([0, 0, 1, 0, 0, 0])  # prismatic hand joint
    xi5 = np.array([0, 1, 0, 0, 0, 0])  # prismatic top finger joint
    xi6 = np.array([0, -1, 0, 0, 0, 0])  # prismatic bottom finger joint

    # Arm joint
    e0 = expm(wedge(xi_x) * joints[0])
    e1 = expm(wedge(xi_y) * joints[1])
    e2 = expm(wedge(xi_z) * joints[2])

    # Rev joints at fingertips
    e3 = expm(wedge(xi_rev_z2) * joints[3])
    e4 = expm(wedge(xi_rev_y2) * joints[4])
    e5 = expm(wedge(xi_rev_x2) * joints[5])
    e6 = expm(wedge(xi_rev_z2) * joints[6])

    # Rev joints at base
    e7 = expm(wedge(xi_rev_z) * joints[7])
    e8 = expm(wedge(xi_rev_y) * joints[8])
    e9 = expm(wedge(xi_rev_x) * joints[9])

    # Prismatic joints sliders
    e10 = expm(wedge(xi_x) * joints[10])
    e11 = expm(wedge(xi_y) * joints[11])
    e12 = expm(wedge(xi_z) * joints[12])

    # Hand and finger joints
    e13 = expm(wedge(xi4) * joints[13])
    e14 = expm(wedge(xi5) * joints[14])
    e15 = expm(wedge(xi6) * joints[15])

    list_xis = [xi_x, xi_y, xi_z, xi_rev_z2, xi_rev_y2, xi_rev_x2]
    list_transforms = [e0, e1, e2, e3, e4, e5]

    if mode == "left":
        return e0 @ e1 @ e2 @ e3 @ e4 @ e5 @ e6 @ e7 @ e8 @ e9 @ e10 @ e11 @ e12 @ e13 @ e14
    elif mode == "right":
        return e0 @ e1 @ e2 @ e3 @ e4 @ e5 @ e6 @ e7 @ e8 @ e9 @ e10 @ e11 @ e12 @ e13 @ e15
    elif mode == "mid":
        return e0 @ e1 @ e2 @ e3 @ e4 @ e5 @ e6 @ e7 @ e8 @ e9 @ e10 @ e11 @ e12 @ e13
    elif mode == "slides":
        slide_x = e0 @ e1 @ e2 @ e3 @ e4 @ e5 @ e6 @ e7 @ e8 @ e9
        slide_y = e0 @ e1 @ e2 @ e3 @ e4 @ e5 @ e6 @ e7 @ e8 @ e9 @ e10
        slide_z = e0 @ e1 @ e2 @ e3 @ e4 @ e5 @ e6 @ e7 @ e8 @ e9 @ e10 @ e11
        return slide_x, slide_y, slide_z

    else:
        return list_xis, list_transforms
