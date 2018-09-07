# tensorflow_mesh_renderer
Rastering algorithm to approximate the rendering of a 3D model silhouette in a fully differentiable way.

---

This code accompanies the paper:
> "End-to-end 6-DoF Object Pose Estimation through Differentiable Rasterization"

> Andrea Palazzi, Luca Bergamini, Simone Calderara, Rita Cucchiara

to appear in "*Second Workshop on 3D Reconstruction Meets Semantics (3DRMS)*" at ECCVW 2018.


<table style="width:100%">
    <tr>
        <th>
            <p align="center">
            <img src="./img/rot_y.gif" alt="Rotation on Y axis" width="50%" height="50%">
            <br>Rendering while rotating the camera around Y axis.
            </p>
        </th>
        <th>
            <p align="center">
            <img src="./img/rot_z.gif" alt="Rotation on Z axis" width="50%" height="50%">
            <br>Rendering while rotating the camera around Z axis.
            </p>
        </th>
     </tr>
 </table>

---

## Docs

#### Input meshes
The input meshes are expected to be `np.ndarray` of shape (n_triangles, 3, 3). Each mesh can be composed of a variable number of triangles. Five meshes of car 3D models are already in [data/](./data/) directory to test the Rasterer.

#### Hello World
A short snippet to test the Rasterer is available in [`main.py`](./main.py). Just run it.

Three 3D models are randomly sampled from the dummy dataset and rendered in a batch. In this snippet the position of the camera is kept fixed for all three (but it may be changed).

If everything went fine, you should see the rendering output for the 3D models sampled. Something like this:

<p align="center">
 <img src="./img/output.png" alt="render_output" width="33%" height="33%">
 <br>Output with resolution=(128, 128)
</p>


