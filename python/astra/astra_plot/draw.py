# -----------------------------------------------------------------------
#   brief             functions that draw the geometries
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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from .utils import set_axes_equal, eucl_dist3d, null


__all__ = ['proj_geom',
           'vol_geom']


def proj_geom(geometry, h_ax=None, *args, **kwargs):
    """
    :brief:             main plotting function for projection geometries
                        tests for the different types and calls the specific
                        sub-functions to draw that type.
    :param geometry:    the geometry that is supposed to be displayed
    :param h_ax:        handle to axis to plot into. default is None, meaning that plt.gca()
                        will be used for getting an axis
    :param args:        arguments for the plot in order
    :param kwargs:      keyword arguments for the plot (same as args, but can be unordered)

                        possible options (in order) are:
                        'RotationAxis'          if specified, will change the drawn
                                                rotation axis to provided axis.
                                                Must be 3-vector. Default value is
                                                [NaN, NaN, NaN], (meaning do not draw).
                        'RotationAxisOffset'    if specified, will translate the drawn
                                                rotation axis by the provided vector.
                                                Default = [0, 0, 0]
                        'VectorIdx'             index of the vector to visualize if @p geometry
                                                is a vector geometry type. Default = 1
                        'Color'                 Color for all markers and lines if not
                                                otherwise specified
                        'DetectorMarker'        marker for the detector locations.
                                                Default = '.'
                        'DetectorMarkerColor'   color specifier for the detector marker.
                                                Default = 'k'
                        'DetectorLineColor'     color for the lines drawing the detector
                                                outline
                        'DetectorLineWidth'     line width of detector rectangle
                        'SourceMarker'          marker for the source locations
                        'SourceMarkerColor'     color specifier for the source marker
                        'SourceDistance'        (only for parallel3d and parallel3d_vec)
                                                distance of source to origin
                        'OpticalAxisColor'      Color for drawing the optical axis
    :return:            ax  -   handle to the axis that the geometry is now in
    """
    if h_ax is None:
        fig = plt.figure()
        h_ax = fig.add_subplot(111, projection='3d')
    options = parse_options_proj_geom(*args, **kwargs)

    if geometry['type'] == 'parallel3d':
        print('type parallel3d')
        print('detector spacing [{0}, {1}]'.format(geometry['DetectorSpacingX'],
            geometry['DetectorSpacingY']))
        print('detector px: [{0}, {1}]'.format(geometry['DetectorRowCount'],
            geometry['DetectorColCount']))
        print('angle lo: {0}'.format(geometry['ProjectionAngles'][0]))
        print('angle hi: {0}'.format(geometry['ProjectionAngles'][-1]))
        print('# angles: {0}'.format(len(geometry['ProjectionAngles'])))
        print('DistanceOriginDetector inf')
        print('DistanceOriginSource inf')
        draw_parallel3d(h_ax, geometry, options)

    elif geometry['type'] == 'parallel3d_vec':
        print('type: parallel3d_vec')
        print('detector px: [{0}, {1}]'.format(geometry['DetectorRowCount'],
            geometry['DetectorColCount']))
        print('# angles: {0}'.format(geometry['Vectors'].shape[0]))
        draw_parallel3d_vec(h_ax, geometry, options)

    elif geometry['type'] == 'cone':
        print('type: cone')
        print('detector spacing: [{0}, {1}]'.format(geometry['DetectorSpacingX'],
            geometry['DetectorSpacingY']))
        print('detector px: [{0}, {1}]'.format(geometry['DetectorRowCount'],
            geometry['DetectorColCount']))
        print('angle lo: {0}'.format(geometry['ProjectionAngles'][0]))
        print('angle hi: {0}'.format(geometry['ProjectionAngles'][-1]))
        print('# angles: {0}'.format(len(geometry['ProjectionAngles'])))
        print('DistanceOriginDetector {0}'.format(geometry['DistanceOriginDetector']))
        print('DistanceOriginSource {0}'.format(geometry['DistanceOriginSource']))
        draw_cone_geom(h_ax, geometry, options)

    elif geometry['type'] == 'cone_vec':
        print('type: cone_vec')
        print('detector px: [{0}, {1}]'.format(geometry['DetectorRowCount'],
            geometry['DetectorColCount']))
        print('# angles: {0}'.format(geometry['Vectors'].shape[0]))
        draw_cone_geom_vec(h_ax, geometry, options)

    elif geometry['type'] == 'fanflat':
        print('type: fanflat')
        print('detector px: {0}'.format(geometry['DetectorCount']))
        print('angle lo: {0}'.format(geometry['ProjectionAngles'][0]))
        print('angle hi: {0}'.format(geometry['ProjectionAngles'][-1]))
        print('# angles: {0}'.format(len(geometry['ProjectionAngles'])))
        print('DistanceOriginDetector {0}'.format(geometry['DistanceOriginDetector']))
        print('DistanceOriginSource {0}'.format(geometry['DistanceOriginSource']))

        draw_fanflat_geom(h_ax, geometry, options)

    elif geometry['type'] == 'fanflat_vec':
        print('type: fanflat_vec')
        print('detector px: {0}'.format(geometry['DetectorCount']))
        print('# angles: {0}'.format(geometry['Vectors'].shape[0]))

        draw_fanflat_geom_vec(h_ax, geometry, options)

    else:
        raise ValueError("Geometry type {0} not recognized".format(geometry['type']))

    set_axes_equal(h_ax)

    return h_ax


def vol_geom(geometry, h_ax=None, vx_size=1, *args, **kwargs):
    """
    :brief:             plot a volume geometry
    :param geometry:    the geometry to draw
    :param h_ax:        matplotlib axis handle to plot into. needs to support 3d plotting
                        default is plt.gca()
    :param vx_size:     size of the voxels. default 1
    :param args:        arguments for the plot in order
    :param kwargs:      keyword arguments for the plot (same as args, but can be unordered)

                        possible options (in order) are:

                        'Color'                 Color for all markers and lines if not
                                                otherwise specified
                        'LineWidth'             line width of detector rectangle
                        'Magnification'         magnification factor of the volume geometry
    :return:            ax  -   handle to the axis that the geometry is now in
    """
    def draw_face(h_ax, face_coords, options):
        h_ax.plot(face_coords[0, 0:2],
                  face_coords[1, 0:2],
                  face_coords[2, 0:2], linewidth=options['LineWidth'], c=options['Color'])
        h_ax.plot(face_coords[0, 2:4],
                  face_coords[1, 2:4],
                  face_coords[2, 2:4], linewidth=options['LineWidth'], c=options['Color'])
        h_ax.plot(np.array([face_coords[0, 3], face_coords[0, 1]]),
                  np.array([face_coords[1, 3], face_coords[1, 1]]),
                  np.array([face_coords[2, 3], face_coords[2, 1]]), linewidth=options['LineWidth'], c=options['Color'])
        h_ax.plot(np.array([face_coords[0, 0], face_coords[0, 2]]),
                  np.array([face_coords[1, 0], face_coords[1, 2]]),
                  np.array([face_coords[2, 0], face_coords[2, 2]]), linewidth=options['LineWidth'], c=options['Color'])

    nargin = len(args)
    if h_ax is None:
        fig = plt.figure()
        h_ax = fig.add_subplot(111, projection='3d')

    options = parse_options_vol_geom(*args, **kwargs)

    phantom_height = geometry['GridRowCount'] * vx_size
    phantom_width = geometry['GridColCount'] * vx_size
    phantom_depth = geometry['GridSliceCount'] * vx_size

    if 'option' in geometry:
        g_options = geometry['option']
        minx = g_options['WindowMinX'] * vx_size
        maxx = g_options['WindowMaxX'] * vx_size
        miny = g_options['WindowMinY'] * vx_size
        maxy = g_options['WindowMaxY'] * vx_size
        minz = g_options['WindowMinZ'] * vx_size
        maxz = g_options['WindowMaxZ'] * vx_size
    else:
        minx = phantom_width / 2 * vx_size
        maxx = phantom_width / 2 * vx_size
        miny = phantom_height / 2 * vx_size
        maxy = phantom_height / 2 * vx_size
        minz = phantom_depth / 2 * vx_size
        maxz = phantom_depth / 2 * vx_size

    xx_phantom = options['Magnification'] * np.array([minx, minx, minx, minx, maxx, maxx, maxx, maxx])
    yy_phantom = options['Magnification'] * np.array([miny, miny, maxy, maxy, miny, miny, maxy, maxy])
    zz_phantom = options['Magnification'] * np.array([minz, maxz, minz, maxz, minz, maxz, minz, maxz])

    # as we draw only a wire frame, we need only to draw 4 of the faces
    face1 = np.array([xx_phantom[0:4], yy_phantom[0:4], zz_phantom[0:4]])
    draw_face(h_ax, face1, options)
    face2 = face1.copy()
    face2[0, 0:2] = xx_phantom[0:2]
    face2[0, 2:4] = xx_phantom[4:6]
    face2[1, 0:2] = yy_phantom[0:2]
    face2[1, 2:4] = yy_phantom[4:6]
    face2[2, 0:2] = zz_phantom[0:2]
    face2[2, 2:4] = zz_phantom[4:6]
    draw_face(h_ax, face2, options)
    face3 = face1.copy()
    face3[0, 0:2] = xx_phantom[2:4]
    face3[0, 2:4] = xx_phantom[6:8]
    face3[1, 0:2] = yy_phantom[2:4]
    face3[1, 2:4] = yy_phantom[6:8]
    face3[2, 0:2] = zz_phantom[2:4]
    face3[2, 2:4] = zz_phantom[6:8]
    draw_face(h_ax, face3, options)
    face4 = face1.copy()
    face4[0, :] = xx_phantom[4:8]
    face4[1, :] = yy_phantom[4:8]
    face4[2, :] = zz_phantom[4:8]
    draw_face(h_ax, face4, options)

    return h_ax


def cad_phantom(geometry, *args, **kwargs):
    raise NotImplementedError('drawing CAD phantoms is not yet implemented')


def draw_parallel3d(h_ax, geometry, options):
    """
    :brief:             draw an astra parallel3d projection geometry
    :param h_ax:        handle to axis to draw into
    :param geom:        the geometry to draw
    :param options:     options dictionary with additional settings
                        see docstring in proj_geom() function for further information
    """
    # draw source
    dist_origin_detector = options['SourceDistance']
    dist_origin_source = options['SourceDistance']
    h_ax.scatter([0], [-dist_origin_source], [0], c=options['SourceMarkerColor'],
        marker=options['SourceMarker'])

    vertices, width, height = draw_detector(h_ax, geometry, dist_origin_detector, options)

    h_ax.scatter(0,0,0, s=120, c='k', marker='+') # draw origin

    draw_optical_axis(h_ax, dist_origin_source, dist_origin_detector, options)

    # connect source to detector edges
    for idx in range(4):
        h_ax.plot([vertices[0, idx], vertices[0, idx]], [-dist_origin_source, dist_origin_detector],
            [vertices[2, idx], vertices[2, idx]], c='k', linestyle=':')

    # draw rotation axis
    h_ax.plot([0,0],[0,0], 0.6*np.array([-height, height]), linewidth=2, c='k', linestyle='--')

    draw_labels(h_ax, width, height, dist_origin_detector, dist_origin_source)


def draw_parallel3d_vec(h_ax, geometry, options):
    """
    :brief:             plot a parallel3d vector geometry
    :param h_ax:        axis into which to plot the geometry
    :param geometry:    the parallel3d vector geometry
    :param options:     options dictionary with additional settings
                        see docstring in proj_geom() function for further information
    """
    vectors = geometry['Vectors']
    xray_source = vectors[:, 0:3] * options['SourceDistance'] # source
    detector_center = vectors[:, 3:6]   # center of detector

    # draw the points and connect with lines
    h_ax.scatter(xray_source[:, 0], xray_source[:, 1], xray_source[:, 2],
                 c=options['SourceMarkerColor'], marker=options['SourceMarker'])
    h_ax.scatter(detector_center[:, 0], detector_center[:, 1], detector_center[:, 2],
                 c='k', marker='.')

    detector_base = np.array([vectors[options['VectorIdx'], 6:9], vectors[options['VectorIdx'], 9:12]])
    detector_height = geometry['DetectorColCount']
    detector_width = geometry['DetectorRowCount']
    detector_vertices = draw_detector_vec(h_ax, detector_center, detector_base, detector_width, detector_height,
        options)

    connect_source_detector_parallel(h_ax, detector_vertices, detector_center, detector_base, xray_source,
        options)


def draw_cone_geom(h_ax, geometry, options):
    """
    :brief:             draw a cone beam geometry
    :param h_ax:        axis to draw into
    :param geometry:    the cone beam geometry
    :param options:     options dictionary with additional settings
                        see docstring in proj_geom() function for further information
    """
    h_ax.scatter(0,0,0, s=120, c='k', marker='+') # draw origin

    draw_optical_axis(h_ax, geometry['DistanceOriginDetector'], geometry['DistanceOriginSource'], options)

    # draw source
    h_ax.scatter(0, -geometry['DistanceOriginSource'], 0, c=options['SourceMarkerColor'],
        marker=options['SourceMarker'])

    vertices, width, height = draw_detector(h_ax, geometry, geometry['DistanceOriginDetector'], options)

    # connect source to detector edges
    for idx in range(4):
        h_ax.plot([0, vertices[0, idx]], [-geometry['DistanceOriginSource'],
            geometry['DistanceOriginDetector']],  [0, vertices[2, idx]], c='k', linestyle=':')

    # draw rotation axis
    h_ax.plot([0, 0], [0, 0], 0.6 * np.array([-height, height]), linewidth=2, c='k', linestyle='--')

    draw_labels(h_ax, width, height, geometry['DistanceOriginDetector'], geometry['DistanceOriginSource'])


def draw_cone_geom_vec(h_ax, geometry, options):
    """
    :brief:             draw an astra cone beam vectorized projection geometry
    :param h_ax:        handle to axis to draw into
    :param geometry:    the geometry to draw
    :param options:     dictionary holding options for drawing
    """
    vectors = geometry['Vectors']
    xray_source = vectors[:, 0:3]       # source
    detector_center = vectors[:, 3:6]   # center of detector

    # draw the points and connect with lines
    h_ax.scatter(xray_source[:, 0], xray_source[:, 1], xray_source[:, 2],
        c=options['SourceMarkerColor'], marker=options['SourceMarker'])
    h_ax.scatter(detector_center[:, 0], detector_center[:, 1], detector_center[:, 2],
        c=options['DetectorMarkerColor'], marker=options['DetectorMarker'])

    idx = options['VectorIdx']
    det_base = np.array([vectors[idx, 6: 9], vectors[idx, 9:12]])
    det_height = geometry['DetectorColCount']
    det_width = geometry['DetectorRowCount']

    vertices = draw_detector_vec(h_ax, detector_center, det_base, det_width, det_height, options)

    connect_source_detector(h_ax, vertices, detector_center, xray_source, options)

    # rotation axis will be roughly as long as the source detector distance
    distances = eucl_dist3d(detector_center, xray_source)
    mean_sdd = np.mean(distances)  # mean source detector distance
    draw_rotation_axis(h_ax, mean_sdd, options)

    idx = options['VectorIdx']
    h_ax.text(xray_source[idx, 0], xray_source[idx, 1], xray_source[idx, 2], 'x-ray source')
    h_ax.text(detector_center[idx, 0],  detector_center[idx, 1], detector_center[idx, 2], 'detector')


def draw_fanflat_geom(h_ax, geometry, options):
    """
    :brief:             draw an astra fanflat projection geometry
    :param h_ax:        handle to axis to draw into
    :param geometry:    the geometry to draw
    :param options:     dictionary holding options for drawing
    """
    # convert to faux cone geometry so we don't have to write more code :)!
    cone_geom = astra.create_proj_geom('cone', geometry['DetectorWidth'],
        geometry['DetectorWidth'], 1, geometry['DetectorCount'], geometry['ProjectionAngles'],
        geometry['DistanceOriginSource'], geometry['DistanceOriginDetector'])

    draw_cone_geom(h_ax, cone_geom, options)


def draw_fanflat_geom_vec(h_ax, geometry, options):
    """
    :brief:             draw an astra fanflat vector projection geometry
    :param h_ax:        handle to axis to draw into
    :param geometry:    the geometry to draw
    :param options:     dictionary holding options for drawing
    """
    vectors = geometry['Vectors']
    idx = options['VectorIdx']
    num_angles = vectors.shape[0]

    xray_source = np.zeros([num_angles, 3])
    xray_source[:, 0:2] = vectors[:, 0:2]
    detector_center = np.zeros([num_angles, 3])
    detector_center[:, 0:2] = vectors[:, 2:4]

    h_ax.scatter(xray_source[:, 0], xray_source [:, 1], xray_source [:, 2], c=options['SourceMarkerColor'],
        marker=options['SourceMarker'])
    h_ax.scatter(detector_center[:, 0], detector_center[:, 1], detector_center[:, 2], c=options['DetectorMarkerColor'],
        marker=options['DetectorMarker'])

    detector_u = np.zeros([1, 3])
    detector_u[:, 0:2] = np.array(vectors[idx, 4:6]).reshape([1, 2])
    detector_v = np.fliplr(detector_u)
    detector_base = np.squeeze(np.array([detector_u, detector_v]))
    detector_height = 1
    detector_width = geometry['DetectorCount']

    vertices = draw_detector_vec(h_ax, detector_center, detector_base, detector_width, detector_height, options)
    connect_source_detector(h_ax, vertices, detector_center, xray_source, options)

    # rotation axis will be roughly as long as the source detector distance
    distances = eucl_dist3d(detector_center, xray_source)
    mean_sdd = np.mean(distances) # mean source detector distance
    draw_rotation_axis(h_ax, mean_sdd, options)

    h_ax.text(xray_source[idx, 0], xray_source[idx, 1], xray_source[idx, 2], 'x-ray source')
    h_ax.text(detector_center[idx, 0], detector_center[idx, 1], detector_center[idx, 2], 'detector')


def draw_labels(h_ax, width, height, dist_origin_detector, dist_origin_source):
    """
    :brief:                         draw the labels of the parts of the geometry
    :param h_ax:                    handle axis in which to draw the labels
    :param width:                   width of the detector
    :param height:                  height of the detector
    :param dist_origin_detector:    distance of origin to the detector
    :param dist_origin_source:      distance of origin to the x-ray source
    """
    perc = 0.05
    h_ax.text(perc*width, perc*width, 0.8*height, 'rotation axis')
    h_ax.text(width*perc, 0, perc*height, 'origin')
    h_ax.text(width*perc, -dist_origin_source, perc*height, 'x-ray source')
    h_ax.text(0, dist_origin_detector, 0, 'detector')


def draw_optical_axis(h_ax, distance_origin_detector, distance_origin_source, options):
    # draw lines between source, origin and detector
    h_ax.plot([0, 0], [0, distance_origin_detector], [0, 0], c=options['OpticalAxisColor'])
    h_ax.plot([0, 0], [0, -distance_origin_source], [0, 0], c=options['OpticalAxisColor'])


def draw_detector(h_ax, geometry, dist_origin_detector, options):
    """
    :brief:                         draw the detector
    :param h_ax:                    handle axis in which to draw the detector
    :param geometry:                the geometry to which the detector belongs
    :param dist_origin_detector:    distance of the origin to the detector
    :param options:                 dictionary with additional plotting information
    :return:                        vertices    -   the coordiantes of the corners of the detector
                                    width       -   width of the detector
                                    height      -   height of the detector
    """
    height = geometry['DetectorRowCount'] * geometry['DetectorSpacingY']
    width = geometry['DetectorColCount'] * geometry['DetectorSpacingX']

    vertices = np.zeros([3, 5])
    vertices[0, :] = 0.5 * np.array([-width, -width, width, width, -width])
    vertices[1, :] = np.array([dist_origin_detector, dist_origin_detector, dist_origin_detector,
        dist_origin_detector, dist_origin_detector])
    vertices[2, :] = 0.5 * np.array([height, -height, -height, height, height])
    h_ax.plot(vertices[0, :], vertices[1, :], vertices[2, :], options['DetectorLineColor'])

    return vertices, width, height


def draw_detector_vec(h_ax, det_origin, det_base, det_width, det_height, options):
    """
    :brief:             draw a detector for a vector geometry
    :param h_ax:        handle to axis to draw into
    :param det_origin:  origin/center of the detector
    :param det_base:    2x3 vector specifying the basis vectors of the detector coordinate system
    :param det_shape:   width and height of the detector (number of pixels in both coordinate
                        directions)
    :param options:     dictionary with options
                        'DetectorLineColor'                Color for the line work
    :return:            vertices - the vertices of the detector rectangle
    """
    vertices = np.zeros((3, 5))     # draw the detector rectangle

    vertices[:, 0] = det_origin[options['VectorIdx'], :] - det_base[0, :] * det_width/2.0\
                   + det_base[1, :] * det_height/2.0
    vertices[:, 1] = det_origin[options['VectorIdx'], :] + det_base[0, :] * det_width/2.0\
                   + det_base[1, :] * det_height/2.0
    vertices[:, 2] = det_origin[options['VectorIdx'], :] + det_base[0, :] * det_width/2.0\
                   - det_base[1, :] * det_height/2.0
    vertices[:, 3] = det_origin[options['VectorIdx'], :] - det_base[0, :] * det_width/2.0\
                   - det_base[1, :] * det_height/2.0
    vertices[:, 4] = vertices[:, 0]   # to close the rectangle
    h_ax.plot(vertices[0, :], vertices[1, :], vertices[2, :], c=options['DetectorLineColor'])

    return vertices


def connect_source_detector_parallel(h_ax, detector_vertices, detector_center, detector_base, xray_source,
    options):
    """
    :brief:                     plot lines from source to detector corners (in this case it is straight lines,
                                because of the parallel beam)
    :param h_ax:                handle to the axis to plot the lines
    :param detector_vertices:   vertices of the detector corners
    :param detector_center:     detector center coordinate
    :param detector_base:       basis vectors of the detector coordinate system
    :param x_ray_source:        coordinate of the x-ray source
    :param options:             dictionary containing additional plot options
    """
    # connect source to detector origin
    idx = options['VectorIdx']
    h_ax.plot([detector_center[idx, 0], xray_source[idx, 0]],
              [detector_center[idx, 1], xray_source[idx, 1]],
              [detector_center[idx, 2], xray_source[idx, 2]],
              c=options['OpticalAxisColor'], linestyle='--')

    # compute normal of detector plane
    n = null(detector_base)

    # connect source to detector edges
    for kk in range(4):
        a = detector_vertices[0, kk] - n[0] * xray_source[idx, 0]
        b = detector_vertices[1, kk] - n[1] * xray_source[idx, 1]
        c = detector_vertices[2, kk] - n[2] * xray_source[idx, 2]
        h_ax.plot([a, detector_vertices[0, kk]], [b, detector_vertices[1, kk]],
                  [c, detector_vertices[2, kk]], c='k', linestyle=':')


def connect_source_detector(h_ax, detector_vertices, detector_center,  xray_source, options):
    """
    :brief:                     plot lines from source to detector corners
    :param h_ax:                handle to the axis to plot the lines
    :param detector_vertices:   vertices of the detector corners
    :param detector_center:     detector center coordinate
    :param detector_base:       basis vectors of the detector coordinate system
    :param x_ray_source:        coordinate of the x-ray source
    :param options:             dictionary containing additional plot options
    """
    idx = options['VectorIdx']
    h_ax.plot([detector_center[idx, 0], xray_source[idx, 0]],
              [detector_center[idx, 1], xray_source[idx, 1]],
              [detector_center[idx, 2], xray_source[idx, 2]],
              c=options['OpticalAxisColor'], linestyle='--')


    # connect source to detector edges
    for kk in range(4):
        h_ax.plot([xray_source[idx, 0], detector_vertices[0, kk]],
                  [xray_source[idx, 1], detector_vertices[1, kk]],
                  [xray_source[idx, 2], detector_vertices[2, kk]],
                  c='k', linestyle=':')

def draw_rotation_axis(h_ax, scaling, options):
    rot_axis = options['RotationAxis']
    if not np.isnan(rot_axis[0]):
        rot_axis = options['RotationAxis'] + options['RotationAxisOffset']
        origin = options['RotationAxisOffset']

        # origin of the geometry is assumed to be [0, 0, 0] always!
        h_ax.plot([origin[0], (scaling/2)*rot_axis[0]],
                  [origin[1], (scaling/2)*rot_axis[1]],
                  [origin[2], (scaling/2)*rot_axis[2]],
                  c=options['OpticalAxisColor'], linestyle='-.')
        h_ax.plot([origin[0], -(scaling/2)*rot_axis[0]],
                  [origin[1], -(scaling/2)*rot_axis[1]],
                  [origin[2], -(scaling/2)*rot_axis[2]],
                  c=options['OpticalAxisColor'], linestyle='-.')


def parse_options_proj_geom(*args, **kwargs):
    """options need to be either in the order as given below in the dictionary, or given as
        keyword arguments"""
    options = {}

    nargin = len(args)
    options['RotationAxis'] = args[0] if nargin > 0 else kwargs.get('RotationAxis', [np.nan, np.nan, np.nan])
    options['RotationAxisOffset'] = args[1] if nargin > 1 else kwargs.get('RotationAxisOffset', [0, 0, 0])
    options['VectorIdx'] = args[2] if nargin > 2 else kwargs.get('VectorIdx', 0)
    options['Color'] = args[3] if nargin > 3 else kwargs.get('Color', 'k')
    options['DetectorMarker'] = args[4] if nargin > 4 else kwargs.get('DetectorMarker', '.')
    options['DetectorMarkerColor'] = args[5] if nargin > 5 else kwargs.get('DetectorMarkerColor', 'k')
    options['DetectorLineColor'] = args[6] if nargin > 6 else kwargs.get('DetectorLineColor', options['Color'])
    options['DetectorLineWidth'] = args[7] if nargin > 7 else kwargs.get('DetectorLineWidth', 1)
    options['SourceMarker'] = args[8] if nargin > 8 else kwargs.get('SourceMarker', '*')
    options['SourceMarkerColor'] = args[9] if nargin > 9 else kwargs.get('SourceMarkerColor', options['Color'])
    options['SourceDistance'] = args[10] if nargin > 10 else kwargs.get('SourceDistance', 100)
    options['OpticalAxisColor'] = args[11] if nargin > 11 else kwargs.get('OpticalAxisColor', options['Color'])

    return options

def parse_options_vol_geom(*args, **kwargs):
    """
    :brief:         parse plotting options for plotting a volume geometry
    :param args:    ordered arguments
    :param kwargs:  keyword arguments, same as args, but can be unordered
                    'Color'                 Color for all markers and lines if not
                                            otherwise specified
                    'LineWidth'             line width of detector rectangle
                    'Magnification'         magnification factor of the volume geometry
    :return:        options dictionary
    """
    options = {}
    nargin = len(args)

    options['Color'] = args[0] if nargin > 0 else kwargs.get('Color', 'r')
    options['LineWidth'] = args[1] if nargin > 1 else kwargs.get('LineWidth', 2)
    options['Magnification'] = args[2] if nargin > 2 else kwargs.get('Magnification', 1)

    return options
