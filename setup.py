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
"""Deformable Object Grasping package setuptools."""
import setuptools


with open('README.md', 'r') as file_handle:
    long_description = file_handle.read()


setuptools.setup(
    name='deformable_object_grasping',
    version='0.1.0',
    author='Isabella Huang',
    author_email='isabellah@nvidia.com',
    description=('A framework to automatically perform grasp tests '
                 'on an arbitrary object model of choice.'),
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/NVlabs/deformable_object_grasping/',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'h5py',
        'matplotlib',
        'plotly',
        'numpy',
    ],
    entry_points={
        "console_scripts": [
            "run_grasp_evaluation = run_grasp_evaluation:main",
        ],
    }
)
