# -----------------------------------------------------------------------
# Copyright: 2010-2018, imec Vision Lab, University of Antwerp
#            2013-2018, CWI, Amsterdam
#
# Contact: astra@astra-toolbox.com
# Website: http://www.astra-toolbox.com/
#
# This file is part of the ASTRA Toolbox.
#
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

from . import astra_c as a
from . import astra_plot

def credits():
    """Print credits of the ASTRA Toolbox."""
    return a.credits()


def use_cuda():
    """Test if CUDA is enabled.

    :returns: :class:`bool` -- ``True`` if CUDA is enabled.
    """
    return a.use_cuda()

def set_gpu_index(idx, memory=0):
    """Set default GPU index to use.

    :param idx: GPU index
    :type idx: :class:`int`
    """
    a.set_gpu_index(idx, memory)

def get_gpu_info(idx=-1):
    """Get GPU info.

    :param idx: GPU index, or -1 for current device
    :type idx: :class:`int`
    :returns: :class:`str` -- GPU info
    """
    return a.get_gpu_info(idx)

def has_feature(feature):
    """Check a feature flag.

    These are used to check if certain functionality has been
    enabled at compile time, if new functionality is present, or if
    a backward-incompatible change is present.

    See include/astra/Features.h for a list.

    :param feature: The name of the feature
    :type feature: :class:`str`
    :returns: :class:`bool` -- The presence of the feature
    """
    return a.has_feature(feature)

def delete(ids):
    """Delete an astra object.

    :param ids: ID or list of ID's to delete.
    :type ids: :class:`int` or :class:`list`

    """
    return a.delete(ids)

def info(ids):
    """Print info about an astra object.

    :param ids: ID or list of ID's to show.
    :type ids: :class:`int` or :class:`list`

    """
    return a.info(ids)


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
    """
    return astra_plot.create_example(proj_type, example_type)


def plot_geom(geometry, h_ax=None, *args, **kwargs):
    """
    :brief:             plot an astra geometry
    :param geometry:    any astra geometry, either volume geometry, projection
                        geometry or an *.stl file (powered by stlRead).
    :param h_ax:        the axis to plot into
    :param args:
    :param kwargs:      args and kwargs depend on the input:
                        the parameters are either ordered arguments or keyword arguments

                        if 'geometry' is
                        - a volume geometry
                        :param VxSize:                  voxel size in unit of preference. Must
                                                        be same unit that was used to scale the
                                                        projection geometry.

                        - a volume geometry
                        :param VxSize:                  voxel size in unit of preference. Must
                                                        be same unit that was used to scale the
                                                        projection geometry.
                        :param Magnification:           magnification factor for the phantom. For
                                                        small phantoms it might be necessary to
                                                        scale the render up as otherwise it won't
                                                        show up in the plot. default = 1
                        :param LineWidth:               line width for the box wire frame.
                                                        default = 2
                        :param Color:                   color of the wire frame. default = 'r'

                        - a projection geometry
                        :param RotationAxis:            if specified, will change the drawn
                                                        rotation axis to provided axis.
                                                        Must be 3-vector. Default value is
                                                        [NaN, NaN, NaN], (meaning do not draw).
                        :param RotationAxisOffset       if specified, will translate the drawn
                                                        rotation axis by the provided vector.
                                                        default = [0, 0, 0]
                        :param VectorIdx:               index of the vector to visualize if
                                                        vector geometry type. Default = 1
                        :param Color:                   color for all markers and lines if not
                                                        otherwise specified
                        :param DetectorMarker:          marker for the detector locations.
                                                        default = '.'
                        :param DetectorMarkerColor:     color specifier for the detector marker.
                                                        default = 'k'
                        :param DetectorLineColor:       color for the lines drawing the
                                                        detector outline
                        :param DetectorLineWidth:       line width of detector rectangle
                        :param SourceMarker:            marker for the source locations
                        :param SourceMarkerColor:       color specifier for the source marker
                        :param SourceDistance:          (only for parallel3d and parallel3d_vec)
                                                        distance of source to origin
                        :param OpticalAxisColor:        color for drawing the optical axis
    :return:            h_ax    -   handle to the axis into which the geometry was drawn.
                                    If h_ax was passed as a parameter that was != None, the same
                                    axis is returned
    """
    return astra_plot.plot_geom(geometry, h_ax=None, *args, **kwargs)
