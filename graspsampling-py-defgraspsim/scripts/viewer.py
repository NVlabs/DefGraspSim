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
"""Create visual scene for grasps."""

from graspsampling import visualize, io, utilities

import os
import argparse


def make_parser():
    """Create parser for grasp file and mesh file."""
    parser = argparse.ArgumentParser(description='Visualize grasp file.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input', type=str,
                        help='Input file name.')
    parser.add_argument('--mesh_root_dir', type=str, default='.',
                        help='Root folder for mesh file names.')
    return parser


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    # Load file
    grasp_dict = io.load(args.input)

    # Load object mesh
    object_mesh = utilities.instantiate_mesh(
        file=os.path.join(
            args.mesh_root_dir,
            grasp_dict['object']),
        scale=grasp_dict['object_scale'])

    visualize.create_scene(object_mesh, gripper_name=grasp_dict['gripper'], **grasp_dict).show()
