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
"""Visualize a TET mesh in Plotly. Requires download of Plotly library."""

import csv
import os

import numpy as np
import plotly.graph_objects as go


def main():
    """Visualize tet in mesh_path."""
    mesh_path = os.path.join('assets', 'small_mat_taller', 'small_mat_taller.stl.tet')

    verts, tets = import_tet(path=mesh_path)
    tris = tets_to_tris(tets=tets)
    plot_mesh(verts=verts, tris=tris)


def import_tet(path):
    """Import nodes and elements from a TET file."""
    # Get verts and tets
    verts = []
    tets = []
    with open(path, 'r') as file_obj:
        csv_reader = csv.reader(file_obj, delimiter=' ')
        for row in csv_reader:
            if row:  # If row is not blank
                if row[0] == 'v':  # If row contains verts
                    verts.append(row[1:4])
                if row[0] == 't':  # If row contains tet
                    tets.append(row[1:])
        verts = np.asarray(verts).astype('float')
        tets = np.asarray(tets).astype('int')

    return verts, tets


def plot_mesh(verts, tris):
    """Plot a triangular mesh using Plotly."""
    # Define trace
    trace = go.Mesh3d(
        x=verts[:, 0],
        y=verts[:, 1],
        z=verts[:, 2],
        i=tris[:, 0],
        j=tris[:, 1],
        k=tris[:, 2],
        opacity=0.5,
        color='cyan')

    fig = go.Figure(data=[trace])
    fig.update_layout(scene_aspectmode='data')

    fig.show()


def tets_to_tris(tets):
    """Convert an array of tets into an array of corresponding tris."""
    tris = []
    for tet in tets:
        # Tri definitions based on deformable.h
        tris.append([tet[0], tet[2], tet[1]])
        tris.append([tet[1], tet[2], tet[3]])
        tris.append([tet[0], tet[1], tet[3]])
        tris.append([tet[0], tet[3], tet[2]])
    tris = np.asarray(tris)

    return tris


if __name__ == "__main__":
    main()
