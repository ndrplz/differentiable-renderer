"""
Rasterer class implement a differentiable rasterer.
"""
import numpy as np
import tensorflow as tf

from rastering.utils import calibration_matrix
from rastering.utils import project_in_2D


def keep_top_n(meshes, top_n):
    """
    Filter a list of meshes keeping only the top `n` wider triangles
    """
    def triangle_area(edge_1, edge_2):
        return np.linalg.norm(np.cross(edge_1, edge_2)) / 2.0

    meshes_filtered = []
    for mesh in meshes:
        areas = [triangle_area(edge_1=t[1] - t[0], edge_2=t[1] - t[2]) for t in mesh]
        widest_triangles = mesh[np.argsort(areas)[-top_n:]]
        meshes_filtered.append(widest_triangles)
    return np.asarray(meshes_filtered)


class Rasterer:
    def __init__(self, meshes: list, resolution_px: tuple, resolution_mm: tuple, focal_len_mm: int,
                 pl_camera_pose: tf.Tensor, pl_model_idx: tf.Tensor, max_triangles: int=1000):
        """
        Create a Rasterer object. The rendering output can be accessed through `image` attribute.
         
        :param meshes: List of meshes. Each mesh is an array of shape (n_triangles, 3, 3).
        :param resolution_px: Camera resolution in pixel.
        :param resolution_mm: Camera resolution in millimeters.
        :param focal_len_mm: Camera focal length in millimeters.
        :param pl_camera_pose: Placeholder to control the camera pose.
        :param pl_model_idx: Placeholder to choose which model in `meshes` is to be rendered.
        :param max_triangles: For computational reasons, all meshes are pre-processed to have
                              a maximum number of triangles `max_triangles`.
        """
        meshes = keep_top_n(meshes, top_n=max_triangles)
        self.meshes = tf.constant(value=meshes, dtype=tf.float32)

        # Placeholders
        self.pl_model_idx   = pl_model_idx     # (None,)
        self.pl_camera_pose = pl_camera_pose   # (None, 4, 4)

        self.res_x_px, self.res_y_px = resolution_px       # image resolution in pixels
        self.res_x_mm, self.res_y_mm = resolution_mm       # size of camera sensor in mm
        self.aspect_ratio = self.res_y_px / self.res_x_px  # vertical

        # Prepare the meshgrid once
        self.meshgrid = tf.meshgrid(tf.range(0, self.res_x_px), tf.range(0, self.res_y_px))
        self.meshgrid = tf.stack([self.meshgrid[0], self.meshgrid[1]], axis=-1)
        self.meshgrid_flat = tf.reshape(self.meshgrid, shape=(-1, 2))

        # Store the calibration matrix
        K = calibration_matrix(resolution_px=(self.res_x_px, self.res_y_px),
                               resolution_mm=(self.res_x_mm, self.res_y_mm),
                               focal_len_mm=focal_len_mm, skew=0)
        self.K = tf.constant(value=K, dtype=tf.float32)

        # Iterate over all examples in the batch
        i_init = tf.constant(value=0, dtype=tf.int32)
        i_max  = tf.shape(self.pl_model_idx)[0]
        outputs = tf.TensorArray(dtype=tf.float32, size=i_max)

        def should_continue(i, *args):
            return i < i_max

        def iteration(i, outputs, meshes, pl_model_idx, pl_camera_pose):
            # Select camera pose and 3D mesh for current example in the batch
            cur_camera_pose = pl_camera_pose[i]
            cur_model_idx   = pl_model_idx[i]
            cur_mesh = meshes[cur_model_idx]

            # Project current mesh in 2D
            cur_triangles_2d_flat = project_in_2D(self.K, cur_camera_pose, cur_mesh,
                                                  resolution_px=(self.res_x_px, self.res_y_px))
            cur_triangles_2d = tf.reshape(cur_triangles_2d_flat, shape=(-1, 3, 2))

            inside_scalar_b = self.inside_outside(self.meshgrid_flat, cur_triangles_2d)

            _min = tf.reduce_min(inside_scalar_b, axis=1)
            _max = tf.reduce_max(inside_scalar_b, axis=1)
            norm_update = ((inside_scalar_b - tf.expand_dims(_min, axis=-1)) /
                           tf.expand_dims(_max - _min + 1e-08, axis=-1))
            norm_update = tf.nn.tanh(norm_update)

            norm_update_sum = tf.reduce_sum(norm_update, axis=0)
            cur_image = tf.reshape(norm_update_sum, shape=(self.res_y_px, self.res_x_px))

            outputs_ = outputs.write(i, cur_image)

            return i + 1, outputs_, meshes, pl_model_idx, pl_camera_pose

        i, outputs, a, b, c = tf.while_loop(should_continue, iteration,
                                            loop_vars=[i_init, outputs, self.meshes,
                                                       self.pl_model_idx, self.pl_camera_pose])
        images = outputs.stack()

        images.set_shape((None, self.res_y_px, self.res_x_px))

        self.image = images

    def get_coords_to_test(self, triangle: tf.Tensor):
        """
        Get from the meshgrid only the coordinates
        which lie inside the triangle bounding box
        """
        minima = tf.maximum(tf.reduce_min(triangle - 1., axis=0), 0.)
        maxima = tf.minimum(tf.reduce_max(triangle + 1., axis=0), (self.res_x_px, self.res_y_px))

        min_x = tf.to_int32(minima[0])
        min_y = tf.to_int32(minima[1])
        max_x = tf.to_int32(maxima[0])
        max_y = tf.to_int32(maxima[1])

        coord_to_test = self.meshgrid[min_x: max_x, min_y: max_y]
        return tf.reshape(coord_to_test, shape=(-1, 2))

    @staticmethod
    def inside_outside(points_2d: tf.Tensor, triangle_2d: tf.Tensor):
        """
        Test for each pixel whether is lies in one or more 2D triangles.
        Each pixel has a non-zero value in the output iif it lies in at least one triangle.
        Please see the paper for details.
        """
        # Re-cast points to float for picky TF
        points_2d_float = tf.to_float(points_2d)

        edge_0 = triangle_2d[:, 1] - triangle_2d[:, 0]
        edge_1 = triangle_2d[:, 2] - triangle_2d[:, 1]
        edge_2 = triangle_2d[:, 0] - triangle_2d[:, 2]

        # Notice: 2D edges can be collinear, leading to a 0 normal
        N = edge_0[:, 0]*edge_2[:, 1] - edge_0[:, 1]*edge_2[:, 0] + 1e-8

        # vectors from vertices to the point
        C0 = triangle_2d[:, None, 0] - tf.to_float(points_2d_float)
        C1 = triangle_2d[:, None, 1] - tf.to_float(points_2d_float)
        C2 = triangle_2d[:, None, 2] - tf.to_float(points_2d_float)

        t_1 = (edge_0[:, 0, None] * C0[:, :, 1] - edge_0[:, 1, None] * C0[:, :, 0]) * tf.expand_dims(N, -1)
        t_2 = (edge_1[:, 0, None] * C1[:, :, 1] - edge_1[:, 1, None] * C1[:, :, 0]) * tf.expand_dims(N, -1)
        t_3 = (edge_2[:, 0, None] * C2[:, :, 1] - edge_2[:, 1, None] * C2[:, :, 0]) * tf.expand_dims(N, -1)

        # Approximate check
        inside_scalar = tf.maximum(t_1, 0.) * tf.maximum(t_2, 0.) * tf.maximum(t_3, 0.)

        return inside_scalar
