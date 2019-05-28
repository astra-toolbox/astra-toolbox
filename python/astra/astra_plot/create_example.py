# -----------------------------------------------------------------------
#   brief             creation of example geometries
#   - last update     08.05.2019
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
import astra
import numpy as np
from .utils import rotate_around3d, translate_3d, magnify_proj, rotate_detector


__all__ = ['create_example']


def create_example(proj_type, example_type=''):
    """
    :param proj_type:       which kind of projection? Available options are
                            'cone'
                            'parallel3d'
                            'fanflat'
    :param example_type:     type of geometry to create. provided as one of the
                             following strings (not case sensitive):
                             ''             -  default geometry for a given type
                             'normal'       -   standard (non vector) geometry
                             'vec'          -   example vector geometry
                             'helix'        -   helical trajectory vector geometry
                             'deform_vec'   -   deformed vector geometry example,
                                                 courtesy of Van Nguyen
    :return:                proj_geom       -   the geometry that was created
    :date:                  08.05.2019
    :author:                Tim Elberfeld
                            imec VisionLab
                            University of Antwerp
    :last mod:              08.05.2019
    """
    def raise_exeption(*args):
        raise ValueError('ASTRA: invalid projection type {0}'.format(proj_type))
    create = {'cone': create_example_cone,
              'parallel3d' : create_example_parallel3d,
              'fanflat' : create_example_fanflat}.get(proj_type, raise_exeption)
    return create(example_type)


def create_example_cone(example_type):
    """
    :brief:                 create and example of a parallel3d geometry
    :param example_type:    for the examples, there are different options
                            '' or 'normal':
                            -   example of a standard cone beam geometry
                            'vec':
                            -   a 'realistic' vector geometry
                            'helix':
                            -   helical trajectory vector geometry
                            'deform_vec':
                            -   same as the 'vec' geometry, but the coordinates are deformed so that the
                                rotation axis 'wobbles'
    """
    t_low = example_type.lower()
    proj_geom = None

    if t_low == '' or t_low == 'normal':
        det_spacing = np.array([0.035, 0.035])
        detector_px = np.array([1200.0, 1200.0])
        angles = np.linspace(0.0, 2*np.pi, 100)
        source_origin = 30.0
        origin_det = 200.0
        phantom_size = 5.0
        phantom_px = 150.0 # voxels for the phantom
        vx_size = phantom_size / phantom_px # voxel size

        # now express all measurements in terms of the voxel size
        det_spacing = det_spacing / vx_size
        origin_det = origin_det / vx_size
        source_origin = source_origin / vx_size

        proj_geom = astra.create_proj_geom('cone',  det_spacing[0], det_spacing[1], detector_px[0],
                                           detector_px[1], angles, source_origin, origin_det)
    elif t_low == 'vec':
        # geometry settings taken from code from Alice Presenti
        detector_pixel_size = 0.0748
        projection_size = [1536, 1944, 21] # detector size and number of projections
        SOD = 679.238020    # [mm]
        SDD = 791.365618    # [mm]
        gamma = np.linspace(0, 300, 21) * np.pi/180.0
        Sorig = np.array([-SOD, 0, 0,                   # the ray origin vector (source)
                          SDD - SOD, 0, 0,              # detector center
                          0, -detector_pixel_size, 0,   # detector u axis
                          0, 0, detector_pixel_size])   # detector v axis
        z_axis = [0, 0, 1]
        S0 = np.tile(np.reshape(Sorig, [12, 1]), [1, projection_size[2]]).T
        S0[:, 0: 3] = rotate_around3d(S0[:, 0: 3], z_axis, gamma)
        S0[:, 3: 6] = rotate_around3d(S0[:, 3: 6], z_axis, gamma)
        S0[:, 6: 9] = rotate_around3d(S0[:, 6: 9], z_axis, gamma)
        S0[:, 9:12] = rotate_around3d(S0[:, 9:12], z_axis, gamma)

        proj_geom = astra.create_proj_geom('cone_vec', projection_size[1], projection_size[0], S0)
    elif t_low == 'helix':
        det_shape = [220, 220]
        obj_src_dist = 500      # mm
        rot_step = 3.6          # deg
        num_angles = 200
        z_translation = 0.5     # mm
        z_dist = z_translation * num_angles;
        vectors = np.zeros([num_angles, 12])
        translation = -z_dist + (z_translation * np.array(range(num_angles)))

        min_angle = 0 # just assume we start at 0
        maxAngle = (num_angles * rot_step)/180 * np.pi  # convert deg to rad
        angles = np.linspace(min_angle, maxAngle, 200)

        # source position per angle
        vectors[:,  0] = np.sin(angles) * obj_src_dist
        vectors[:,  1] = -np.cos(angles) * obj_src_dist
        vectors[:,  2] = translation

        # detector position per angle
        vectors[:,  3] = 0
        vectors[:,  4] = 0
        vectors[:,  5] = translation

        # vector from detector pixel (0,0) to (0,1)
        vectors[:,  6] = np.cos(angles)
        vectors[:,  7] = np.sin(angles)
        vectors[:,  8] = np.zeros([num_angles,])

        # vector from detector pixel (0,0) to (1, 0)
        vectors[:,  9] = np.zeros([num_angles,])
        vectors[:, 10] = np.zeros([num_angles,])
        vectors[:, 11] = np.ones([num_angles, ])

        proj_geom = astra.create_proj_geom('cone_vec', det_shape[0], det_shape[1], vectors)
    elif t_low == 'deform_vec':
        detector_pixel_size = 0.0748
        projection_size = [1536, 1944, 21] # detector size and number of projections
        SOD = 679.238020    # [mm]
        SDD = 791.365618    # [mm]
        gamma = np.linspace(0, 300, 21) * np.pi/180.0
        Sorig = np.array([-SOD, 0, 0,                   # the ray origin vector (source)
                          SDD - SOD, 0, 0,              # detector center
                          0, -detector_pixel_size, 0,   # detector u axis
                          0, 0, detector_pixel_size])   # detector v axis
        z_axis = [0, 0, 1]
        S0 = np.tile(np.reshape(Sorig, [12, 1]), [1, projection_size[2]]).T
        S0[:, 0:3] = translate_3d(S0[:, 0:3], [100, 150, 0])
        S0[:, 3:6] = translate_3d(S0[:, 3:6], [100, 150, 0])

        S0 = rotate_detector(S0, [0.48,0.32,0])
        S0 = magnify_proj(S0, 100)
        S0[:, 0: 3] = rotate_around3d(S0[:, 0: 3], z_axis, gamma)
        S0[:, 3: 6] = rotate_around3d(S0[:, 3: 6], z_axis, gamma)
        S0[:, 6: 9] = rotate_around3d(S0[:, 6: 9], z_axis, gamma)
        S0[:, 9:12] = rotate_around3d(S0[:, 9:12], z_axis, gamma)

        proj_geom = astra.create_proj_geom('cone_vec', projection_size[1], projection_size[0], S0)

    return proj_geom

def create_example_parallel3d(example_type=''):
    """
    :brief:                 create an example of a parallel3d geometry
    :param example_type:    for the examples, there are different options
                            -   '' or 'normal':
                                example of a standard parallel geometry with 100 angles
                            -   'vec':
                                same geometry as for normal, but converted to a vector geometry
                                this gives a different plot visualizing the individual angles on
                                top of the geometry
    :return:                the example geometry
    """
    det_spacing = [0.035, 0.035]
    detector_px = [1000, 1000]
    angles = np.linspace(0, 2*np.pi, 100)

    if example_type == '' or example_type == 'normal':
        proj_geom = astra.create_proj_geom('parallel3d', det_spacing[0], det_spacing[1], detector_px[0],
                                           detector_px[1], angles)
    elif example_type == 'vec':
        proj_geom = astra.geom_2vec(create_example_parallel3d())
    else:
        raise ValueError(
            "ASTRA: example type {0} not recognized for parallel3d geometry".format(example_type))

    return proj_geom

def create_example_fanflat(example_type=''):
    """
    :brief:                 create an example geometry of type fanflat
    :param example_type:    -   '' or 'normal':
                                example of a standard fanflat geometry with 100 angles
                            -   'vec':
                                same geometry as for normal, but converted to a vector geometry
                                this gives a different plot visualizing the individual angles on
                                top of the geometry
    :return:                the example geometry
    """
    def make_normal_geometry():
        det_spacing = 0.035
        detector_px = 1200
        angles = np.linspace(0, 2*np.pi, 100)
        source_origin = 30
        origin_det = 200
        phantom_size = 5
        phantom_px = 150                        # voxels for the phantom
        vx_size = phantom_size / phantom_px     # voxel size

        # now express all measurements in terms of the voxel size
        det_spacing = det_spacing / vx_size
        origin_det = origin_det / vx_size
        source_origin = source_origin / vx_size

        geom = astra.create_proj_geom('fanflat',  det_spacing, detector_px, angles, source_origin, origin_det)
        return geom

    if example_type == '' or example_type == 'normal':
        proj_geom = make_normal_geometry()
    elif example_type == 'vec':
        proj_geom = astra.geom_2vec(make_normal_geometry())
    else:
        raise ValueError("ASTRA: example type {0} not recognized for parallel3d geometry".format(example_type))

    return proj_geom