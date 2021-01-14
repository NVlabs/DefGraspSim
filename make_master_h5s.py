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
"""Create master files for each of the 4 tests per elastic modulus."""

import os
import shutil

import h5py
import numpy as np


# Set this to the results directory for the object of interest.
OBJ_RESULTS_DIR = "results/rectangle_v2"

modes = ["reorient", "pickup", "twist", "shake"]
all_moduli = os.listdir(OBJ_RESULTS_DIR)
for mod in all_moduli:
    results_dir = os.path.join(OBJ_RESULTS_DIR, mod)
    print(results_dir)

    all_files = os.listdir(results_dir)
    for mode in modes:
        print("Current mode", mode)
        mode_files = [os.path.join(results_dir, o) for o in all_files if o.endswith(
            ".h5") and "master" not in o and mode in o]
        if len(mode_files) == 0:
            continue

        mode_master_file = mode + "_master.h5"
        shutil.copy(mode_files[0], os.path.join(results_dir, mode_master_file))

        master_f = h5py.File(os.path.join(results_dir, mode_master_file), 'a')
        dataset_names = master_f.keys()

        num_grasps = master_f['directions'].shape[0]

        for file in mode_files:
            f = h5py.File(file, 'r')
            print("======", file)
            for i in range(num_grasps):
                if not np.all(f['directions'][i] == 0.0):
                    # Populate master CVS row with contents here
                    for dataset_name in dataset_names:
                        dataset = f[dataset_name]
                        master_dataset = master_f[dataset_name]
                        try:
                            master_dataset[i] = dataset[i]
                        except BaseException:
                            pass
            f.close()

        # Patch in the holes
        for file in mode_files:
            print("~~~~~~", file)
            f = h5py.File(file, 'r')
            for i in range(num_grasps):
                if not np.all(f['directions'][i] == 0.0):
                    # Populate master CVS row with contents here
                    master_timedout = master_f['timed_out']
                    f_timedout = f['timed_out']

                    num_dirs = master_timedout[i].shape[0]

                    for d in range(num_dirs):
                        if master_timedout[i][d] == 0:  # No timeout, so no need to replace
                            continue
                        master_timedout[i, d] = f_timedout[i, d]
                        master_f['reorientation_meshes'][i, d, :, :,
                                                         :] = f['reorientation_meshes'][i, d, :,
                                                                                        :, :]
                        master_f['shake_fail_accs'][i, d] = f['shake_fail_accs'][i, d]
                        master_f['twist_fail_accs'][i, d] = f['twist_fail_accs'][i, d]

            f.close()

        master_f.close()
