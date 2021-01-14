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
"""Unit tests for panda_fk.py."""

import numpy as np
import pytest

from isaacgym import gymapi

import utils.panda_fk


def test_skew():
    """Test `panda_fk.skew` functionality."""
    a = np.array([1., 2., 3.])
    skew_actual = utils.panda_fk.skew(a)
    skew_expected = np.array([[0., -3., 2.], [3., 0., -1.], [-2., 1., 0.]])
    assert skew_expected == pytest.approx(skew_actual)


def test_wedge():
    """Test 'panda_fk.wedge' functionality."""
    xi = np.array([1., 2., 3., 4., 5., 6.])
    wedge_actual = utils.panda_fk.wedge(xi)
    wedge_expected = np.array([[0, -6., 5., 1.], [6., 0, -4., 2.],
                               [-5., 4, 0., 3.], [0., 0., 0., 0.]])
    assert wedge_expected == pytest.approx(wedge_actual)


def test_adjoint():
    """Test 'panda_fk.adjoint' functionality."""
    g = np.array([[1., 2., 3., 4.], [5., 6., 7., 8.], [9., 10., 11., 12.], [13., 14., 15., 16.]])
    adjoint_actual = utils.panda_fk.adjoint(g)
    adjoint_expected = np.array([[1., 2., 3., 12., 8., 4.], [5., 6., 7., -24., -16., -8.],
                                 [9., 10., 11., 12., 8., 4.], [0., 0., 0., 1., 2., 3],
                                 [0., 0., 0., 5., 6., 7.], [0., 0., 0., 9., 10., 11.]])
    assert adjoint_expected == pytest.approx(adjoint_actual)


def test_get_fk():
    """Test 'panda_fk.get_fk' functionality."""
    hand_origin = gymapi.Transform()
    hand_origin.p = gymapi.Vec3(1., 2., 3.)

    # Joint angles are zero
    joints = np.zeros(16)
    fk_left_actual = utils.panda_fk.get_fk(joints, hand_origin, "left")
    fk_right_actual = utils.panda_fk.get_fk(joints, hand_origin, "right")
    fk_mid_actual = utils.panda_fk.get_fk(joints, hand_origin, "mid")
    fk_expected = np.eye(4)
    assert fk_expected == pytest.approx(fk_left_actual)
    assert fk_expected == pytest.approx(fk_right_actual)
    assert fk_expected == pytest.approx(fk_mid_actual)

    # Joint angles are non-zero
    joints = np.array([0.1, -0.2, 1.4, -0.4, 0.4, -0.6, 1.2, -1.1, 0.35,
                       0.76, 0.22, -0.33, 0.72, 0.78, 0.03, 0.02])
    fk_actual = utils.panda_fk.get_fk(joints, hand_origin, "right")
    fk_expected = np.array([[0.62718193, 0.56042884, 0.54089032, -1.5753273],
                            [-0.39134237, 0.82717439, -0.40327866, 0.72207522],
                            [-0.67341961, 0.04125579, 0.73810838, 3.74801295], [0., 0., 0., 1.]])
    assert fk_expected == pytest.approx(fk_actual)


def test_jacobian():
    """Test 'panda_fk.jacobian' functionality."""
    hand_origin = gymapi.Transform()
    hand_origin.p = gymapi.Vec3(1., 2., 3.)

    # Joint angles are zero
    joints = np.zeros(16)
    jacobian_actual = utils.panda_fk.jacobian(joints, hand_origin)
    jacobian_expected = np.array([[1., 0., 0., 2., -3.112, 0.], [0., 1., 0., -1., 0., 3.112],
                                  [0., 0., 1., 0., 1., -2.], [0., 0., 0.,
                                                              0., 0., 1.], [0., 0., 0., 0., 1., 0.],
                                  [0., 0., 0., 1., 0., 0.]])
    assert jacobian_expected == pytest.approx(jacobian_actual)
