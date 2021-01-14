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
"""Return regularly spaced vectors in a unit sphere.

Implementation follows https://www.cmu.edu/biolphys/deserno/pdf/sphere_equi.pdf
"""

import numpy as np


def quat_between_vectors(u, v):
    """Return the quaternion transformation between two vectors."""
    xyz = np.cross(u, v)
    w = np.sqrt(np.linalg.norm(u)**2 * np.linalg.norm(v)**2) + np.dot(u, v)
    q = [xyz[0], xyz[1], xyz[2], w]
    return q / np.linalg.norm(q)


def point_from_spherical_coords(r, theta, phi):
    """Convert spherical coordinates to 3D position."""
    return [r * np.sin(theta) * np.cos(phi), r * np.sin(theta) * np.sin(phi), r * np.cos(theta)]


def get_uniform_directions_regular(N):
    """Return N regularly spaced vectors in a unit sphere."""
    xs, ys, zs = [], [], []
    N_count = 0
    r = 1.0
    a = 4 * np.pi * r**2 / N
    d = np.sqrt(a)
    M_theta = int(np.pi / d)
    d_theta = np.pi / M_theta
    d_phi = a / d_theta

    directions = []
    for m in range(M_theta):
        theta = np.pi * (m + 0.5) / M_theta
        M_phi = int(2 * np.pi * np.sin(theta) / d_phi)
        for n in range(M_phi):
            phi = 2 * np.pi * n / M_phi
            [x, y, z] = point_from_spherical_coords(r, theta, phi)
            directions.append([x, y, z])
            xs.append(x)
            ys.append(y)
            zs.append(z)
            N_count += 1
    return directions, xs, ys, zs
