"""
Rasterer class implement a differentiable rasterer.
"""
import numpy as np
import tensorflow as tf
import torch
from torch.autograd import Variable
import torch.nn.functional as F
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


class Rasterer(torch.nn.Module):
    def __init__(self, meshes: list, resolution_px: tuple, resolution_mm: tuple, focal_len_mm: int,
                 max_triangles: int=1000):
        """
        Create a Rasterer object. The rendering output can be accessed through `image` attribute.
         
        :param meshes: List of meshes. Each mesh is an array of shape (n_triangles, 3, 3).
        :param resolution_px: Camera resolution in pixel.
        :param resolution_mm: Camera resolution in millimeters.
        :param focal_len_mm: Camera focal length in millimeters.
        :param max_triangles: For computational reasons, all meshes are pre-processed to have
                              a maximum number of triangles `max_triangles`.
        """
        super(Rasterer, self).__init__()

        meshes = keep_top_n(meshes, top_n=max_triangles)
        self.meshes = torch.from_numpy(meshes).float()

        self.res_x_px, self.res_y_px = resolution_px       # image resolution in pixels
        self.res_x_mm, self.res_y_mm = resolution_mm       # size of camera sensor in mm
        self.aspect_ratio = self.res_y_px / self.res_x_px  # vertical

        # Prepare the meshgrid once
        xx, yy = np.mgrid[0: self.res_x_px, 0: self.res_y_px]
        meshgrid = np.concatenate((xx[..., None], yy[..., None]), axis=-1)
        self.meshgrid_flat = torch.from_numpy(meshgrid.reshape((1, -1, 2)))

        # Store the calibration matrix
        K = calibration_matrix(resolution_px=(self.res_x_px, self.res_y_px),
                               resolution_mm=(self.res_x_mm, self.res_y_mm),
                               focal_len_mm=focal_len_mm, skew=0)
        self.K = torch.from_numpy(K).float()

    def __call__(self, *args, **kwargs):
        return super(Rasterer, self).__call__(*args, **kwargs)

    def forward(self, model_idxs, camera_poses):
        outputs = []
        for model_idx, camera_matrix in zip(model_idxs, camera_poses):
            cur_mesh = self.meshes[model_idx]

            # Project current mesh in 2D
            triangles_2d_flat = project_in_2D(self.K, camera_matrix, cur_mesh,
                                              resolution_px=(self.res_x_px, self.res_y_px))
            triangles_2d = triangles_2d_flat.view(-1, 3, 2)

            inside_scalar_b = self.inside_outside(self.meshgrid_flat, triangles_2d)

            # Normalization
            _min, _ = torch.min(inside_scalar_b, dim=1)
            _max, _ = torch.max(inside_scalar_b, dim=1)
            norm_update = (inside_scalar_b - _min.unsqueeze(-1)) / (_max - _min + 1e-08).unsqueeze(-1)
            norm_update = F.tanh(norm_update)

            image = torch.sum(norm_update, dim=0).view(self.res_y_px, self.res_x_px)
            outputs.append(image)
        return torch.stack(outputs)

    @staticmethod
    def inside_outside(points_2d: torch.Tensor, triangle_2d: torch.Tensor):
        """
        Test for each pixel whether is lies in one or more 2D triangles.
        Each pixel has a non-zero value in the output iif it lies in at least one triangle.
        Please see the paper for details.
        """
        # Re-cast points to float
        points_2d_float = points_2d.float()

        edge_0 = triangle_2d[:, 1] - triangle_2d[:, 0]
        edge_1 = triangle_2d[:, 2] - triangle_2d[:, 1]
        edge_2 = triangle_2d[:, 0] - triangle_2d[:, 2]

        # Notice: 2D edges can be collinear, leading to a 0 normal
        N = edge_0[:, 0]*edge_2[:, 1] - edge_0[:, 1]*edge_2[:, 0] + 1e-8
        N = torch.unsqueeze(N, dim=-1)
        # vectors from vertices to the point
        C0 = triangle_2d[:, None, 0] - points_2d_float
        C1 = triangle_2d[:, None, 1] - points_2d_float
        C2 = triangle_2d[:, None, 2] - points_2d_float

        t_1 = (edge_0[:, 0, None] * C0[:, :, 1] - edge_0[:, 1, None] * C0[:, :, 0]) * N
        t_2 = (edge_1[:, 0, None] * C1[:, :, 1] - edge_1[:, 1, None] * C1[:, :, 0]) * N
        t_3 = (edge_2[:, 0, None] * C2[:, :, 1] - edge_2[:, 1, None] * C2[:, :, 0]) * N

        # Approximate check
        lower_bound = torch.Tensor([0])
        t_1_clip = torch.max(t_1, lower_bound)
        t_2_clip = torch.max(t_2, lower_bound)
        t_3_clip = torch.max(t_3, lower_bound)
        return t_1_clip * t_2_clip * t_3_clip
