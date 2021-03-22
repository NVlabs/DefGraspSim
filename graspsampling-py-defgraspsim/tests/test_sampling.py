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
"""Samples grasps on a test object."""

from context import graspsampling

from graspsampling import sampling, utilities, hands

import logging


def test_sampling(cls_sampler=graspsampling.sampling.UniformSampler, number_of_grasps=100):
    """Sample grasps on a test object."""
    gripper = hands.create_gripper('panda')

    # Load object
    fname_object = 'data/objects/banana.obj'
    logging.info("Loading", fname_object)
    test_object = utilities.instantiate_mesh(file=fname_object, scale=0.01)
    logging.info("Extents of loaded mesh:", test_object.extents)

    # Instantiate and run sampler
    sampler = cls_sampler(gripper, test_object)
    results = sampler.sample(number_of_grasps)

    assert('poses' in results)
    assert(len(results['poses']) == number_of_grasps)

    return gripper, test_object, results


if __name__ == "__main__":
    test_sampling()
