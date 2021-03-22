# graspsampling-py

Python library for different grasp sampling schemes.

## Requirements
* Python3
* trimesh
* urdfpy
* scipy
* cvxopt
* Optional:
  * [hydra](hydra.cc) (for running large-scale experiments)
  * h5py (for writing h5 files)

## Sampling Schemes
For more information see: ["A Billion Ways to Grasp: An Evaluation of Grasp Sampling Schemeson a Dense, Physics-based Grasp Data Set"](https://arxiv.org/pdf/1912.05604.pdf).
The following grasp sampling schemes are available:

### Grid Sampler
Simply places equidistant poses in SE3. The resolution is defined separately for
translation (`resolution_xyz` in [m]) and rotation (`resolution_orientation`
from `{72, 576, 4608, 36864}` which corresponds to `{60, 30, 15, 7.5deg}`).
The orientations are generated using Yershova et al. (2008): "Generating Uniform
Incremental Grids on SO(3) Using the Hopf Fibration".
```
sampler:
  cls: graspsampling.sampling.GridSampler
  params:
    resolution_xyz: 0.02
    resolution_orientation: 72
```

[//]: # (TODO: multilevel sequence)

### Uniform Sampler
Uniform sampling in SE(3). Position samples are limited to the volume defined
by the non-empty intersection between the object's bounding box and the gripper's
bounding sphere.
```
sampler:
  cls: graspsampling.sampling.UniformSampler
  params:
    {}
```

[//]: # (TODO: add halton, soble sequence.)

### Polar Coordinate Sampler
Sample a random ray through the object's center of mass, and positions along this
ray `step_size` [m] apart. The grasp's orientation is aligned with the ray and
 sampled within a cone of aperture `orientation_towards_com` [rad].
This scheme is used in:
> Yilun Zhou and Kris Hauser (2017): [6-DOF Grasp Planning by Optimizing a Deep Learning Scoring Function"](http://motion.pratt.duke.edu/papers/RSS2017Workshop-Zhou-6DOFGraspPlanning.pdf)

```
sampler:
  cls: graspsampling.sampling.PolarCoordinateSampler
  params:
    step_size: 0.01
    orientation_towards_com: 0.0
```

### Approach-based Surface Sampler

<img src="docs/images/approach_based_sampling.png" width="300"/>

This sampler aligns the gripper's approach vector with the surface normals of the target object.
Points on the object surface are sampled randomly. The parameterization includes
the aperture of the cone aligned with the surface normal in which samples are drawn
(`surface_normal_cone` [rad]; $`\alpha`$ in the figure above) and similarly the `approach_cone` [rad]
($`\beta`$ in the figure above).

This scheme (or slight variations of it) is used in:
> Kappler, Bohg, Schaal (2015): [Leveraging big data for grasp planning](https://ieeexplore.ieee.org/iel7/7128761/7138973/07139793.pdf)<br />
> Kleinhans, Rosman, Michalik, Tripp, Detry (2015): [G3DB: A database of successful and failed grasps with rgb-d images, point clouds, meshmodels and gripper parameters](https://researchspace.csir.co.za/dspace/bitstream/handle/10204/8613/Kleinhans2_2015.pdf?sequence=1&isAllowed=y)<br />
> Veres , Moussa, Taylor (2017): [An integrated simulator and dataset that combines grasping and vision for deep learning](https://arxiv.org/pdf/1702.02103.pdf)

```
sampler:
  cls: graspsampling.sampling.SurfaceApproachSampler
  params:
    surface_normal_cone: 0.0
    approach_cone: 0.0
```

### Antipodal-based Sampler
<img src="docs/images/antipodal_based_sampling.png" width="300"/>

A point on the object surface is sampled randomly and defines the contact point for the first finger.
Then ray is shot in the opposite direction of the surface normal. It intersection
with the object surface defines the second contact point. The ray does not need
to be aligned with the surface normal but can be sampled from a cone around it,
with aperture `friction_cone` [rad] ($`\alpha`$ in the figure above). The two contact
points (sometimes referred to as _grasp axis_) do not constrain the rotation of 
the gripper around the grasp axis. The `number_of_orientations` defines how many
random rotations around the grasp axis are drawn.

This scheme is used in:
> Mahler, Liang, Niyaz, Laskey, Doan, Liu, Ojea, Goldberg (2017): [Dex-net 2.0: Deep learning to plan robust grasps with synthetic pointclouds and analytic grasp metrics](https://arxiv.org/pdf/1703.09312.pdf)

```
sampler:
  cls: graspsampling.sampling.AntipodalSampler
  params:
    friction_cone: 0.0
    number_of_orientations: 6
```

### Bounding Box/Sphere/Cylinder Sampler
Doesn't exist yet. Should be added.

[//]: # (### Sampler used in OpenRAVE)

### How to add a new sampling scheme?
Derive a new class from `GraspSampler` and overwrite all abstract methods (so far only `sample`):
```
class MySampler(GraspSampler):

    def __init__(self, gripper, object_mesh):
        self.gripper = gripper
        self.mesh = mesh
    
    def sample(self, number_of_grasps):
        # Do your thing
        return {'poses': [[]]}
```

## General Sampling Options

### Reject collisions
To remove grasps that collide with the object use [`collision.in_collision_with_gripper`](graspsampling/collision.py).

### Non-empty closing region heuristic
A common heuristic is to ignore grasps for which the gripper's closing region (the space
between the fingers) is empty. This is described in:
> Andreas ten Pas, Marcus Gualtieri, Kate Saenko, Robert Platt (2017): [Grasp Pose Detection in Point Clouds](https://arxiv.org/pdf/1706.09911)

This can be achieved by using the function [`collision.check_gripper_nonempty`](graspsampling/collision.py).

[//]: # (* Calculate grasp metrics)

## Grippers

<table>
  <tr>
    <td><img src="docs/images/hands_panda.png" height="200"/></td>
  </tr>
  <tr>
    <td align="center"><em>[hands.PandaGripper](/graspsampling/hands.py)</em></td>
  </tr>
</table>

## Visualizing Grasps

This can be done using trimesh's viewer via [`visualize.create_scene`](/graspsampling/visualize.py).
Or see [tests/test_visualizer.py](/tests/test_visualizer.py) for an example.
In addition, there is a viewer script [/scripts/viewer.py](/scripts/viewer.py) that takes a grasp file as a argument and visualizes it.

Here are some examples:

<table>
  <tr>
    <td><img src="docs/images/sampling_uniform.png" height="200"/></td>
    <td><img src="docs/images/sampling_grid.png" height="200"/></td>
    <td><img src="docs/images/sampling_polar.png" height="200"/></td>
    <td><img src="docs/images/sampling_approach.png" height="200"/></td>
    <td><img src="docs/images/sampling_antipodal.png" height="200"/></td>
  </tr>
  <tr>
    <td align="center"><em>[sampling.UniformSampler](/graspsampling/sampling.py)</em></td>
    <td align="center"><em>[sampling.GridSampler](/graspsampling/sampling.py)</em></td>
    <td align="center"><em>[sampling.PolarCoordinateSampler](/graspsampling/sampling.py)</em></td>
    <td align="center"><em>[sampling.ApproachSurfaceSampler](/graspsampling/sampling.py)</em></td>
    <td align="center"><em>[sampling.AntipodalSampler](/graspsampling/sampling.py)</em></td>
  </tr>
</table>

## Storing Results

Results can be stored as `json` or `h5`. See the [/graspsampling/io.py](/graspsampling/io.py) module for implementations, and [/tests/test_io.py](/tests/test_io.py) for examples.

## Using Hydra

[Hydra](https://www.hydra.cc) enables clean configuration management. This enables running parameter sweeps for samplers and more.
See [/tests/test_hydra.py](/tests/test_hydra.py) for an example.