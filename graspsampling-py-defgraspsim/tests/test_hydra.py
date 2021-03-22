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
"""Test if config files are imported correctly."""

import hydra
from omegaconf import DictConfig

from context import graspsampling
from graspsampling import utilities, sampling


@hydra.main(config_path='../conf', config_name="config")
def my_app(cfg: DictConfig) -> None:
    """Test if config files are imported correctly."""
    print(cfg.pretty())

    object_mesh = utilities.instantiate_mesh(**cfg.object)
    gripper = hydra.utils.instantiate(cfg.gripper)

    sampler = hydra.utils.instantiate(cfg.sampler, gripper=gripper, object_mesh=object_mesh)
    results = sampling.collision_free_grasps(gripper, object_mesh, sampler, cfg.number_of_grasps)

    output = hydra.utils.instantiate(cfg.output)
    output.write(results)


if __name__ == "__main__":
    # load configuration via hydra
    my_app()
