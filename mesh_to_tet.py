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

"""Convert a .mesh file (fTetWild format) to .tet (IsaacGym format)."""

# Define input and output file names
mesh_file = open("/home/isabella/Downloads/hollow_flask.mesh", "r")
tet_output = open(
    "/home/isabella/carbgym/DeformableGrasping/output_mesh.tet", "w")

# Parse .mesh file
mesh_lines = list(mesh_file)
mesh_lines = [line.strip('\n') for line in mesh_lines]
vertices_start = mesh_lines.index('Vertices')
num_vertices = mesh_lines[vertices_start + 1]

vertices = mesh_lines[vertices_start + 2:vertices_start + 2
                      + int(num_vertices)]

tetrahedra_start = mesh_lines.index('Tetrahedra')
num_tetrahedra = mesh_lines[tetrahedra_start + 1]
tetrahedra = mesh_lines[tetrahedra_start + 2:tetrahedra_start + 2
                        + int(num_tetrahedra)]

print("# Vertices, # Tetrahedra:", num_vertices, num_tetrahedra)

# Write to tet output
tet_output.write("# Tetrahedral mesh generated using\n\n")
tet_output.write("# " + num_vertices + " vertices\n")
for v in vertices:
    tet_output.write("v " + v + "\n")
tet_output.write("\n")
tet_output.write("# " + num_tetrahedra + " tetrahedra\n")
for t in tetrahedra:
    line = t.split(' 0')[0]
    line = line.split(" ")
    line = [str(int(k) - 1) for k in line]
    l_text = ' '.join(line)
    tet_output.write("t " + l_text + "\n")
