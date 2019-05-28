# -----------------------------------------------------------------------
#   brief             the main plotting function
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
import os
from . import draw


__all__ = ["plot_geom"]


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
    """
    if not isinstance(geometry, (dict, str, bytes)) :
        raise ValueError(
            "ASTRA: geometry can't be of type {0}. Please specify "
            "a valid ASTRA geometry or a path to an *.stl file".format(type(geometry)))
    if is_vol_geom(geometry):
        h_ax = draw.vol_geom(geometry, h_ax, *args, **kwargs)
    elif is_proj_geom(geometry):
        h_ax = draw.proj_geom(geometry, h_ax, *args, **kwargs)
    elif os.path.exists(geometry):  # assume 'geometry' is a path to a CAD file
        throw(ValueError("ASTRA: drawing CAD files is currently only supported in the MATLAB API"))

    return h_ax


def is_vol_geom(geom):
    return all([k in geom for k in ['GridRowCount', 'GridColCount']])


def is_proj_geom(geom):
    return 'type' in geom
