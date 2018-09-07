import numpy as np
import tensorflow as tf


def project_in_2D(K, camera_pose, mesh, resolution_px):
    """
    Project all 3D triangle vertices in the mesh into
    the 2D image of given resolution

    Parameters
    ----------
    K: ndarray
        Camera intrinsics matrix, 3x3
    camera_pose: ndarray
        Camera pose (inverse of extrinsics), 4x4
    mesh: ndarray
        Triangles to be projected in 2d, (Nx3x3) 
    resolution_px: tuple
        Resolution of image in pixel

    Returns
    -------
    coords_projected_2D: ndarray
        Triangle vertices projected in 2D and clipped to
        image resolution
    """
    resolution_x_px, resolution_y_px = resolution_px  # image resolution in pixels

    # Decompose camera pose into rotation and translation
    RT = camera_pose[:-1, :]  # remove homogeneous row
    R = RT[:, :-1]  # rotation matrix 3x3
    T = RT[:, -1:]  # translation vector 3x1

    # Invert the camera pose matrix to get the camera extrinsics
    # Due to the particular matrix geometry we can avoid raw inversion
    Rc = tf.matrix_transpose(R)
    Tc = -Rc @ T
    RT = tf.concat([Rc, Tc], axis=-1)  # camera extrinsics

    # Correct reference system of extrinsics matrix
    #   y is down: (to align to the actual pixel coordinates used in digital images)
    #   right-handed: positive z look-at direction
    correction_factor = tf.constant(value=np.array([[1., 0., 0.],
                                                    [0., -1., 0.],
                                                    [0., 0., -1.]]), dtype=tf.float32)
    RT = correction_factor @ RT

    # Compose whole camera projection matrix (3x4)
    P = K @ RT

    mesh_flat = tf.reshape(mesh, shape=(-1, 3))
    len_mesh_flat = tf.shape(mesh_flat)[0]
    # Create constant tensor to store 3D model coordinates
    coords_3d_h = tf.concat([mesh_flat, tf.ones(shape=(len_mesh_flat, 1), dtype=tf.float32)], axis=-1)  # n_triangles, 4
    coords_3d_h = tf.transpose(coords_3d_h, perm=[1, 0])  # 4, n_triangles

    # Project 3D vertices into 2D
    coords_projected_2D_h = tf.transpose(P @ coords_3d_h, perm=[1, 0])  # n_triangles, 3
    coords_projected_2D = coords_projected_2D_h[:, :2] / (coords_projected_2D_h[:, 2:3] + 1e-8)

    # Clip indexes in image range
    coords_projected_2D_x_clip = tf.clip_by_value(coords_projected_2D[:, 0:0 + 1],
                                                  clip_value_min=-1, clip_value_max=resolution_x_px)
    coords_projected_2D_y_clip = tf.clip_by_value(coords_projected_2D[:, 1:1 + 1],
                                                  clip_value_min=-1, clip_value_max=resolution_y_px)

    return tf.concat([coords_projected_2D_x_clip, coords_projected_2D_y_clip], axis=-1)


def calibration_matrix(resolution_px, resolution_mm, focal_len_mm, skew=0.):
    """
    Return calibration matrix K given camera information
    """
    # Camera intrinsics parameters
    resolution_x_px, resolution_y_px = resolution_px  # image resolution in pixels
    resolution_x_mm, resolution_y_mm = resolution_mm  # size of camera sensor in mm

    skew = skew  # "skew param will be zero for most normal cameras" Hartley, Zisserman
    focal_len_mm = focal_len_mm  # camera focal len in mm

    assert (resolution_x_px / resolution_y_px == resolution_x_mm / resolution_y_mm)

    m_x = resolution_x_px / resolution_x_mm
    m_y = resolution_y_px / resolution_y_mm

    alpha_x = focal_len_mm * m_x  # focal length of the camera in pixels
    alpha_y = focal_len_mm * m_y  # focal length of the camera in pixels

    x_0 = resolution_x_px / 2
    y_0 = resolution_y_px / 2

    return np.array([[alpha_x, skew, x_0],
                     [0, alpha_y, y_0],
                     [0, 0, 1]])
