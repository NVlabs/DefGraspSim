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
"""Helper functions to calculate mesh-based metrics."""

import numpy as np
from scipy.signal import butter, filtfilt


def get_strain_energy_of_element(ts, ti, particle_state_tensor, youngs):
    """Return the strain energy, volume, and center of a tetahedral element."""
    voigt_stress = np.array([ts.x.x, ts.y.y, ts.z.z, ts.z.y, ts.z.x, ts.y.z])
    invE = 1 / youngs
    nu = 0.3  # poissons #TODO, CHNAGE THIS TO REFLECT POISSNS
    D_inv = np.zeros((6, 6))
    D_inv[0][0] = invE
    D_inv[0][1] = -invE * nu
    D_inv[0][2] = -invE * nu
    D_inv[1][0] = -invE * nu
    D_inv[1][1] = invE
    D_inv[1][2] = -invE * nu
    D_inv[2][0] = -invE * nu
    D_inv[2][1] = -invE * nu
    D_inv[2][2] = invE
    D_inv[3][3] = invE * (1.0 + nu)
    D_inv[4][4] = invE * (1.0 + nu)
    D_inv[5][5] = invE * (1.0 + nu)

    voigt_strain = D_inv @ voigt_stress

    # Get area of the tetrahedra based on its four particle locations
    p0 = particle_state_tensor[ti[0], :3]
    p1 = particle_state_tensor[ti[1], :3]
    p2 = particle_state_tensor[ti[2], :3]
    p3 = particle_state_tensor[ti[3], :3]
    all_particles = np.vstack((p0, p1, p2, p3))
    V = (1 / 6) * np.abs(np.dot((p0 - p3), np.cross((p1 - p3), (p2 - p3))))

    return 0.5 * V * np.dot(voigt_strain, voigt_stress), V, np.mean(
        all_particles, axis=0)


def get_stresses_only(gym, sim, envs, env_index, particle_state_tensor):
    """Return the von Mises stresses for all tetrahedra."""
    (tet_particles, tet_stresses) = gym.get_sim_tetrahedra(sim)
    num_envs = gym.get_env_count(sim)
    num_tets_per_env = int(len(tet_stresses) / num_envs)

    vm_stresses = np.zeros((num_envs, num_tets_per_env))
    env = envs[env_index]
    tet_range = gym.get_actor_tetrahedra_range(env, 1, 0)

    for global_tet_index in range(tet_range.start,
                                  tet_range.start + tet_range.count):
        ts = tet_stresses[global_tet_index]

        vm_stress = np.sqrt(0.5
                            * ((ts.x.x - ts.y.y)**2 + (ts.y.y - ts.z.z)**2
                               + (ts.z.z - ts.x.x)**2 + 6
                               * (ts.y.x**2 + ts.z.y**2 + ts.z.x**2)))
        local_tet_index = global_tet_index % num_tets_per_env

        vm_stresses[env_index][local_tet_index] = vm_stress

    return vm_stresses


def get_tet_based_metrics(gym, sim, envs, env_index, particle_state_tensor,
                          youngs):
    """Return the stresses, strain energy, volume, and centroid of a mesh."""
    (tet_particles, tet_stresses) = gym.get_sim_tetrahedra(sim)

    total_strain_energy = 0.0
    total_volume = 0.0
    env = envs[env_index]
    tet_range = gym.get_actor_tetrahedra_range(env, 1, 0)

    weighted_location = 0.0

    for global_tet_index in range(tet_range.start,
                                  tet_range.start + tet_range.count):
        ts = tet_stresses[global_tet_index]

        ti = tet_particles[4 * global_tet_index:4 * global_tet_index
                           + 4]  # The indices of the four tet particles

        strain_energy, volume, tet_center = get_strain_energy_of_element(
            ts, ti, particle_state_tensor, youngs)
        weighted_location += tet_center * volume
        total_strain_energy += strain_energy
        total_volume += volume

    weighted_centroid = weighted_location / total_volume

    vm_stresses = get_stresses_only(gym, sim, envs, env_index, particle_state_tensor)
    return vm_stresses, total_strain_energy, total_volume, weighted_centroid


def butter_lowpass_filter(data):
    """Low-pass filter the dynamics data."""
    fs = 20
    cutoff = 0.05
    nyq = 0.5 * fs
    order = 1
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data, axis=0)
    return y
