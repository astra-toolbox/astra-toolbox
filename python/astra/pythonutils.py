# -----------------------------------------------------------------------
# Copyright: 2010-2022, imec Vision Lab, University of Antwerp
#            2013-2022, CWI, Amsterdam
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

"""Additional purely Python functions for PyAstraToolbox.

.. moduleauthor:: Daniel M. Pelt <D.M.Pelt@cwi.nl>


"""
import numpy as np


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
    elif geom['type'] == 'fanflat_vec' or geom['type'] == 'parallel_vec':
        s = (geom['Vectors'].shape[0], geom['DetectorCount'])
    elif geom['type'] == 'parallel3d_vec' or geom['type'] == 'cone_vec':
        s = (geom['DetectorRowCount'], geom[
             'Vectors'].shape[0], geom['DetectorColCount'])

    if dim != None:
        s = s[dim]

    return s

def checkArrayForLink(data):
    """Check if a numpy array is suitable for direct usage (contiguous, etc.)

    This function raises an exception if not.
    """

    if not isinstance(data, np.ndarray):
        raise ValueError("Data should be numpy.ndarray")
    if data.dtype != np.float32:
        raise ValueError("Numpy array should be float32")
    if not (data.flags['C_CONTIGUOUS'] and data.flags['ALIGNED']):
        raise ValueError("Numpy array should be C_CONTIGUOUS and ALIGNED")

class GPULink(object):
    """Utility class for astra.data3d.link with a CUDA pointer

    The CUDA pointer ptr must point to an array of floats.

    x is the fastest-changing coordinate, z the slowest-changing.

    pitch is the width in bytes of the memory block. For a contiguous
    memory block, pitch is equal to sizeof(float) * x. For a memory block
    allocated by cudaMalloc3D, pitch is the pitch as returned by cudaMalloc3D.
    """
    def __init__(self, ptr, x, y, z, pitch):
        self.ptr = ptr
        self.x = x
        self.y = y
        self.z = z
        self.pitch = pitch
