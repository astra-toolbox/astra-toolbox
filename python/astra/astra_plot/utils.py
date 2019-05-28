# -----------------------------------------------------------------------
#   brief             utility functions for the drawing of geometries
#   - last update     09.05.2019
# -----------------------------------------------------------------------
# Copyright: 2010-2019, imec Vision Lab, University of Antwerp
#            2013-2019, CWI, Amsterdam
#
# Contact: astra@astra-toolbox.com
# Website: http://www.astra-toolbox.com/
#
# This file is part of the ASTRA Toolbox.
#
# The ASTRA Toolbox is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# The ASTRA Toolbox is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with the ASTRA Toolbox. If not, see <http://www.gnu.org/licenses/>.
#
# -----------------------------------------------------------------------
from __future__ import print_function, absolute_import
import numpy as np
import scipy


__all__ = ["set_axes_equal",
           "rotate_around3d",
           "translate_3d",
           "magnify_proj",
           "rotate_detector",
           "eucl_dist3d",
           "null"]


def set_axes_radius(h_ax, origin, radius):
    """
    :brief:         set the axis limits to a radius around a specific point
    :param h_ax:    the axis to modify
    :param origin:  the point the radius is supposed to be centered around
    :param radius:  the distance from the origin in each direction
    """
    h_ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    h_ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    h_ax.set_zlim3d([origin[2] - radius, origin[2] + radius])


def set_axes_equal(h_ax):
    """
    :brief:         make axes of 3D plot have equal scale so that spheres appear as spheres,
                    cubes as cubes, etc..  This is one possible solution to Matplotlib's
                    h_ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    :param h_ax:    matplotlib axis, e.g., as output from plt.gca().
    """
    limits = np.array([h_ax.get_xlim3d(), h_ax.get_ylim3d(), h_ax.get_zlim3d()])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    set_axes_radius(h_ax, origin, radius)


def rotate_around3d(vec, axis, angle):
    """
    :brief:         rotate a vector or matrix of vectors around an axis by an angle
    :param vec:     the vector(s) to rotate. must be n x 3
    :param axis:    the axis to rotate round, 1 x 3  vector
    :param angle:   the angle to rotate by (scalar or n x 1 vector)
    :return:        rotated vector(s)
    """
    rot_vec = np.zeros_like(vec)
    if len(rot_vec.shape) == 1:
        rot_vec = rot_vec[:, None]
    if len(vec.shape) == 1:
        vec = vec[:, None]

    rot_vec[:, 0] = (axis[0] * axis[0] * (1-np.cos(angle)) + np.cos(angle)) * vec[:, 0] +\
                    (axis[0] * axis[1] * (1-np.cos(angle)) - axis[2] * np.sin(angle)) * vec[:, 1] +\
                    (axis[0] * axis[2] * (1-np.cos(angle)) + axis[1] * np.sin(angle)) * vec[:, 2]

    rot_vec[:, 1] = (axis[0] * axis[1] * (1-np.cos(angle)) + axis[2] * np.sin(angle)) * vec[:, 0] +\
                    (axis[1] * axis[1] * (1-np.cos(angle)) + np.cos(angle)) * vec[:, 1] +\
                    (axis[1] * axis[2] * (1-np.cos(angle)) - axis[0] * np.sin(angle)) * vec[:, 2]

    rot_vec[:, 2] = (axis[0] * axis[2] * (1-np.cos(angle)) - axis[1] * np.sin(angle)) * vec[:, 0] +\
                    (axis[1] * axis[2] * (1-np.cos(angle)) + axis[0] * np.sin(angle)) * vec[:, 1] +\
                    (axis[2] * axis[2] * (1-np.cos(angle)) + np.cos(angle)) * vec[:, 2]
    return rot_vec


def translate_3d(vec, translation):
    """
    :brief:                 translate vectors by another vector
    :param vec:             the vectors to translate
    :param translation:     translation vector to be added to @p vec
    """
    vec_t = vec.copy() + translation
    return vec_t


def magnify_proj(vec, dsdd):
    """
    :brief:         magnify the projection
    :param vec:     the vectors to magnify. must be in the format of a vector geometry!
    :param dsdd:    deviation of the source detector distance
    :return:        magnified @p vec_geom
    """
    magnified_vec_geom = vec
    vec_sd_direction = vec[:,0:3] - vec[:,3:6]
    norm_sd = np.tile(np.sqrt(np.sum(vec_sd_direction**2, 1)), [3, 1]).T
    vec_sd_direction /= norm_sd
    magnified_vec_geom[:, 3:6] = vec[:,3:6] - dsdd * vec_sd_direction

    return magnified_vec_geom


def rotate_euler3d(vec, angles):
    """
    :brief:         rotate vectors according to euler angles
    :param vec:     the vectors to rotate
    :param angles:  the euler angles
    :return:        rotated vectors
    """
    roll_mat = np.array([[1, 0, 0],
                         [0, np.cos(angles[0]), -np.sin(angles[0])],
                         [0, np.sin(angles[0]), np.cos(angles[0])]])
    yaw_mat = np.array([[np.cos(angles[1]), 0, -np.sin(angles[1])],
                        [0, 1, 0],
                        [np.sin(angles[1]), 0, np.cos(angles[1])]])

    pitch_mat = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                         [np.sin(angles[2]), np.cos(angles[2]), 0],
                         [0, 0, 1]])
    # simulate rotation and translation of the DETECTOR
    rot_mat = roll_mat @ yaw_mat @ pitch_mat
    vectors_rot = vec
    num_vectors = vec.shape[0]

    for ii in range(num_vectors):
        vectors_rot[ii, :] = (rot_mat @ vec[ii, :].T).T

    return vectors_rot


def rotate_detector(vec, angles):
    """
    :brief:         given the vectors of a vector geometry, rotate the detector according to the given euler angles
    :param vec:     the vectors of the vector geometry
    :param angles:  the euler angles for the detector rotation
    :return:        the rotated detectors
    """
    vectors_rot = vec
    vectors_rot[:, 0: 3] = rotate_euler3d(vec[:, 0: 3], angles)
    vectors_rot[:, 3: 6] = rotate_euler3d(vec[:, 3: 6], angles)
    vectors_rot[:, 6: 9] = rotate_euler3d(vec[:, 6: 9], angles)
    vectors_rot[:, 9:12] = rotate_euler3d(vec[:, 9:12], angles)

    return vectors_rot


def eucl_dist3d(a, b):
    """
    :brief:     given two vectors or matrices of vectors, compute their euclidean distances
    :param a:   first vector(s)
    :param b:   second vector(s)
    :return:    the distance(s)
    """
    dist = np.sqrt((a[:, 0] - b[:, 0])**2 + (a[:, 1] - b[:, 1])**2 + (a[:, 2] - b[:, 2])**2)
    return dist


def null(matA, eps=1e-12):
    """
    :brief:         implementation of the function null() as present in MATLAB.
                    This function finds the solution to Ax = 0.
                    where @p A is a matrix representing a set of equations,
                    which is done by computing the kernel (or null space) of @p A
    :param matA:    matrix to find null space of
    :param eps:     epsilon value that decides when a numerical value is considered
                    equal to zero
    :return:        the solution x of system Ax = 0
    """
    # test if A is matrix and turn it into one if not
    if len(matA.shape) < 2:
        matA = matA.reshape(matA.shape + (1,)).T

    _, ss, vh = scipy.linalg.svd(matA)
    padding = max(0, np.shape(matA)[1] - np.shape(ss)[0])
    null_mask = np.concatenate(((ss <= eps), np.ones((padding,), dtype=bool)), axis=0)
    null_space = scipy.compress(null_mask, vh, axis=0)
    return scipy.transpose(null_space)