import glob
import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

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
    renderer = Rasterer(meshes=mesh_dataset,
                        pl_camera_pose=tf.placeholder(dtype=tf.float32, shape=(None, 4, 4)),
                        pl_model_idx=tf.placeholder(dtype=tf.int32, shape=(None,)),
                        max_triangles=3000, **camera_intrinsics)

    # Sample a bunch of models from the mesh dataset. There will be rendered in the same batch.
    n_meshes = 3
    model_idxs = np.asarray(random.sample(range(len(mesh_dataset)), k=n_meshes))

    # Load camera pose. In this case it is the same for all renderings.
    camera_matrix = camera_pose.matrix
    camera_matrices = np.tile(camera_matrix[None, ...], [n_meshes, 1, 1])

    with tf.Session() as tf_session:
        tf_session.run(tf.variables_initializer(tf.global_variables()))

        render_output = tf_session.run(renderer.image, feed_dict={renderer.pl_model_idx: model_idxs,
                                                                  renderer.pl_camera_pose: camera_matrices})

        for b in range(n_meshes):
            plt.imshow(render_output[b])
            plt.waitforbuttonpress()
