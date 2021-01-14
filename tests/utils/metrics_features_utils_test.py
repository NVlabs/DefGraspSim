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
"""Unit tests for metrics_features_utils.py."""

import numpy as np
import pytest

from isaacgym import gymapi

import utils.metrics_features_utils


def test_get_desired_rpy():
    """Test `metrics_features_utils.get_desired_rpy` functionality."""
    reorient_quat = gymapi.Quat(0., 0., 0., 1.)
    grasp_quat = gymapi.Quat(0.3096829, 0.4129106, 0.8258211, -0.2272021)
    _, rpy_actual = utils.metrics_features_utils.get_desired_rpy(reorient_quat, grasp_quat)
    rpy_expected = np.array([0.70899426, -0.58928654, -2.99733187])
    assert rpy_expected == pytest.approx(rpy_actual)
