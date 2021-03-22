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
"""Helper functions for intermediate file formets."""

from abc import ABC, abstractmethod

import json

import logging


def load(filename):
    """Load a json file."""
    if filename.endswith('.json'):
        return json.load(open(filename, 'r'))
    else:
        raise NotImplementedError("Cannot load file type:", filename)


class GraspWriter(ABC):
    """Class to write grasps to a file."""
    def __init__(self, file):
        """Writes grasps to a file.

        Args:
            file (str): Output file name.
        """
        self.file = file

    @abstractmethod
    def write(self, **kwargs):
        """Write grasps."""
        pass


class JsonWriter(GraspWriter):
    """Class to write grasps to a json file."""
    def write(self, **kwargs):
        """Write grasps to a json file.

        Args:
            grasps ([type]): [description]

        Raises:
            TypeError: [description]
        """
        with open(self.file, 'w') as f:
            logging.info(f"Writing results to: {self.file}")
            json.dump(kwargs, f)


class H5Writer(GraspWriter):
    """Class to write grasps to an h5 file."""
    def write(self, **kwargs):
        """Write grasps to a json file.

        Args:
            grasps ([type]): [description]

        Raises:
            TypeError: [description]
        """
        import h5py
        hf = h5py.File(self.file, 'w')

        for k, v in kwargs.items():
            if isinstance(v, list):
                hf.create_dataset(k, data=v, compression="gzip")
            else:
                hf.create_dataset(k, data=v)

        # create dataset
        logging.info("Writing results to: %s" % (self.file))
        hf.close()
