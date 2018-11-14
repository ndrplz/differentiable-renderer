import glob

import matplotlib.pyplot as plt
import numpy as np
import torch
from rastering import Rasterer
from rastering import RotoTranslation
from rastering import Vector


if __name__ == '__main__':

    camera_pose = RotoTranslation(rotation=Vector(x=90., y=0., z=0.),
                                  translation=Vector(x=0., y=-8., z=0.),
                                  angle_unit='degrees')
    print(camera_pose, '\n')

    # Toy dataset containing 5 random car meshes from Shapenetcore.
    mesh_dataset = [np.load(mesh_path) for mesh_path in glob.glob('data/*/*.npy')]

    camera_intrinsics = {'resolution_px': (128, 128), 'resolution_mm': (32, 32), 'focal_len_mm': 35}
    renderer = Rasterer(meshes=mesh_dataset, max_triangles=3000, **camera_intrinsics)

    # Sample a bunch of models from the mesh dataset. There will be rendered in the same batch.
    n_mesh_to_render = 3
    model_idxs = np.random.choice(np.arange(len(mesh_dataset)), size=n_mesh_to_render)

    # Load camera pose. In this case it is the same for all renderings.
    camera_matrix = camera_pose.matrix
    camera_matrices = np.tile(camera_matrix[None, ...], [n_mesh_to_render, 1, 1])

    render_output = renderer(torch.from_numpy(model_idxs),
                             torch.from_numpy(camera_matrices))

    for b in range(n_mesh_to_render):
        plt.imshow(render_output[b].to('cpu').numpy())
        plt.waitforbuttonpress()
