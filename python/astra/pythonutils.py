#-----------------------------------------------------------------------
# Copyright 2013 Centrum Wiskunde & Informatica, Amsterdam
#
# Author: Daniel M. Pelt
# Contact: D.M.Pelt@cwi.nl
# Website: http://dmpelt.github.io/pyastratoolbox/
#
#
# This file is part of the Python interface to the
# All Scale Tomographic Reconstruction Antwerp Toolbox ("ASTRA Toolbox").
#
# The Python interface to the ASTRA Toolbox is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#
# The Python interface to the ASTRA Toolbox is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with the Python interface to the ASTRA Toolbox. If not, see <http://www.gnu.org/licenses/>.
#
#-----------------------------------------------------------------------
"""Additional purely Python functions for PyAstraToolbox.

.. moduleauthor:: Daniel M. Pelt <D.M.Pelt@cwi.nl>


"""

def geom_size(geom, dim=None):
    """Returns the size of a volume or sinogram, based on the projection or volume geometry.

    :param geom: Geometry to calculate size from
    :type geometry: :class:`dict`
    :param dim: Optional axis index to return
    :type dim: :class:`int`
    """

    if 'GridSliceCount' in geom:
        # 3D Volume geometry?
        s = (geom['GridSliceCount'], geom[
             'GridRowCount'], geom['GridColCount'])
    elif 'GridColCount' in geom:
        # 2D Volume geometry?
        s = (geom['GridRowCount'], geom['GridColCount'])
    elif geom['type'] == 'parallel' or geom['type'] == 'fanflat':
        s = (len(geom['ProjectionAngles']), geom['DetectorCount'])
    elif geom['type'] == 'parallel3d' or geom['type'] == 'cone':
        s = (geom['DetectorRowCount'], len(
            geom['ProjectionAngles']), geom['DetectorColCount'])
    elif geom['type'] == 'fanflat_vec':
        s = (geom['Vectors'].shape[0], geom['DetectorCount'])
    elif geom['type'] == 'parallel3d_vec' or geom['type'] == 'cone_vec':
        s = (geom['DetectorRowCount'], geom[
             'Vectors'].shape[0], geom['DetectorColCount'])

    if dim != None:
        s = s[dim]

    return s
