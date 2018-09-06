import glob
import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from rasterer import Rasterer


if __name__ == '__main__':

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
    camera_pose = np.load('data/RT.npy')
    camera_poses = np.tile(camera_pose[None, ...], [n_meshes, 1, 1])

    with tf.Session() as tf_session:
        tf_session.run(tf.variables_initializer(tf.global_variables()))

        render_output = tf_session.run(renderer.image, feed_dict={renderer.pl_model_idx: model_idxs,
                                                                  renderer.pl_camera_pose: camera_poses})

        for b in range(n_meshes):
            plt.imshow(render_output[b])
            plt.waitforbuttonpress()
