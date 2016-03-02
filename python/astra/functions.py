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
"""Additional functions for PyAstraToolbox.

.. moduleauthor:: Daniel M. Pelt <D.M.Pelt@cwi.nl>


"""

from . import creators as ac
import numpy as np
try:
    from six.moves import range
except ImportError:
    # six 1.3.0
    from six.moves import xrange as range

from . import data2d
from . import data3d
from . import projector
from . import algorithm
from . import pythonutils



def clear():
    """Clears all used memory of the ASTRA Toolbox.

    .. note::
        This is irreversible.

    """
    data2d.clear()
    data3d.clear()
    projector.clear()
    algorithm.clear()


def data_op(op, data, scalar, gpu_core, mask=None):
    """Perform data operation on data.

    :param op: Operation to perform.
    :param data: Data to perform operation on.
    :param scalar: Scalar argument to data operation.
    :param gpu_core: GPU core to perform operation on.
    :param mask: Optional mask.

    """

    cfg = ac.astra_dict('DataOperation_CUDA')
    cfg['Operation'] = op
    cfg['Scalar'] = scalar
    cfg['DataId'] = data
    if not mask == None:
        cfg['MaskId'] = mask
    cfg['option']['GPUindex'] = gpu_core
    alg_id = algorithm.create(cfg)
    algorithm.run(alg_id)
    algorithm.delete(alg_id)


def add_noise_to_sino(sinogram_in, I0, seed=None):
    """Adds Poisson noise to a sinogram.

    :param sinogram_in: Sinogram to add noise to.
    :type sinogram_in: :class:`numpy.ndarray`
    :param I0: Background intensity. Lower values lead to higher noise.
    :type I0: :class:`float`
    :returns:  :class:`numpy.ndarray` -- the sinogram with added noise.

    """

    if not seed==None:
        curstate = np.random.get_state()
        np.random.seed(seed)

    if isinstance(sinogram_in, np.ndarray):
        sinogramRaw = sinogram_in
    else:
        sinogramRaw = data2d.get(sinogram_in)
    max_sinogramRaw = sinogramRaw.max()
    sinogramRawScaled = sinogramRaw / max_sinogramRaw
    # to detector count
    sinogramCT = I0 * np.exp(-sinogramRawScaled)
    # add poison noise
    sinogramCT_C = np.zeros_like(sinogramCT)
    for i in range(sinogramCT_C.shape[0]):
        for j in range(sinogramCT_C.shape[1]):
            sinogramCT_C[i, j] = np.random.poisson(sinogramCT[i, j])
    # to density
    sinogramCT_D = sinogramCT_C / I0
    sinogram_out = -max_sinogramRaw * np.log(sinogramCT_D)

    if not isinstance(sinogram_in, np.ndarray):
        data2d.store(sinogram_in, sinogram_out)

    if not seed==None:
        np.random.set_state(curstate)

    return sinogram_out

def move_vol_geom(geom, pos, is_relative=False):
    """Moves center of volume geometry to new position.

    :param geom: Input volume geometry
    :type geom: :class:`dict`
    :param pos: Tuple (x,y[,z]) for new position, with the center of the image at (0,0[,0])
    :type pos: :class:`tuple`
    :param is_relative: Whether new position is relative to the old position
    :type is_relative: :class:`bool`
    :returns: :class:`dict` -- Volume geometry with the new center
    """

    ret_geom = geom.copy()
    ret_geom['option'] = geom['option'].copy()

    if not is_relative:
        ret_geom['option']['WindowMinX'] = -geom['GridColCount']/2.
        ret_geom['option']['WindowMaxX'] = geom['GridColCount']/2.
        ret_geom['option']['WindowMinY'] = -geom['GridRowCount']/2.
        ret_geom['option']['WindowMaxY'] = geom['GridRowCount']/2.
        if len(pos)>2:
            ret_geom['option']['WindowMinZ'] = -geom['GridSliceCount']/2.
            ret_geom['option']['WindowMaxZ'] = geom['GridSliceCount']/2.
    ret_geom['option']['WindowMinX'] += pos[0]
    ret_geom['option']['WindowMaxX'] += pos[0]
    ret_geom['option']['WindowMinY'] += pos[1]
    ret_geom['option']['WindowMaxY'] += pos[1]
    if len(pos)>2:
        ret_geom['option']['WindowMinZ'] += pos[2]
        ret_geom['option']['WindowMaxZ'] += pos[2]
    return ret_geom


def geom_size(geom, dim=None):
    """Returns the size of a volume or sinogram, based on the projection or volume geometry.

    :param geom: Geometry to calculate size from
    :type geometry: :class:`dict`
    :param dim: Optional axis index to return
    :type dim: :class:`int`
    """
    return pythonutils.geom_size(geom,dim)


def geom_2vec(proj_geom):
    """Returns a vector-based projection geometry from a basic projection geometry.

    :param proj_geom: Projection geometry to convert
    :type proj_geom: :class:`dict`
    """
    if proj_geom['type'] == 'fanflat':
        angles = proj_geom['ProjectionAngles']
        vectors = np.zeros((len(angles), 6))
        for i in range(len(angles)):

            # source
            vectors[i, 0] = np.sin(angles[i]) * proj_geom['DistanceOriginSource']
            vectors[i, 1] = -np.cos(angles[i]) * proj_geom['DistanceOriginSource']

            # center of detector
            vectors[i, 2] = -np.sin(angles[i]) * proj_geom['DistanceOriginDetector']
            vectors[i, 3] = np.cos(angles[i]) * proj_geom['DistanceOriginDetector']

            # vector from detector pixel 0 to 1
            vectors[i, 4] = np.cos(angles[i]) * proj_geom['DetectorWidth']
            vectors[i, 5] = np.sin(angles[i]) * proj_geom['DetectorWidth']
        proj_geom_out = ac.create_proj_geom(
        'fanflat_vec', proj_geom['DetectorCount'], vectors)

    elif proj_geom['type'] == 'cone':
        angles = proj_geom['ProjectionAngles']
        vectors = np.zeros((len(angles), 12))
        for i in range(len(angles)):
            # source
            vectors[i, 0] = np.sin(angles[i]) * proj_geom['DistanceOriginSource']
            vectors[i, 1] = -np.cos(angles[i]) * proj_geom['DistanceOriginSource']
            vectors[i, 2] = 0

            # center of detector
            vectors[i, 3] = -np.sin(angles[i]) * proj_geom['DistanceOriginDetector']
            vectors[i, 4] = np.cos(angles[i]) * proj_geom['DistanceOriginDetector']
            vectors[i, 5] = 0

            # vector from detector pixel (0,0) to (0,1)
            vectors[i, 6] = np.cos(angles[i]) * proj_geom['DetectorSpacingX']
            vectors[i, 7] = np.sin(angles[i]) * proj_geom['DetectorSpacingX']
            vectors[i, 8] = 0

            # vector from detector pixel (0,0) to (1,0)
            vectors[i, 9] = 0
            vectors[i, 10] = 0
            vectors[i, 11] = proj_geom['DetectorSpacingY']

        proj_geom_out = ac.create_proj_geom(
        'cone_vec', proj_geom['DetectorRowCount'], proj_geom['DetectorColCount'], vectors)

    # PARALLEL
    elif proj_geom['type'] == 'parallel3d':
        angles = proj_geom['ProjectionAngles']
        vectors = np.zeros((len(angles), 12))
        for i in range(len(angles)):

            # ray direction
            vectors[i, 0] = np.sin(angles[i])
            vectors[i, 1] = -np.cos(angles[i])
            vectors[i, 2] = 0

            # center of detector
            vectors[i, 3] = 0
            vectors[i, 4] = 0
            vectors[i, 5] = 0

            # vector from detector pixel (0,0) to (0,1)
            vectors[i, 6] = np.cos(angles[i]) * proj_geom['DetectorSpacingX']
            vectors[i, 7] = np.sin(angles[i]) * proj_geom['DetectorSpacingX']
            vectors[i, 8] = 0

            # vector from detector pixel (0,0) to (1,0)
            vectors[i, 9] = 0
            vectors[i, 10] = 0
            vectors[i, 11] = proj_geom['DetectorSpacingY']

        proj_geom_out = ac.create_proj_geom(
        'parallel3d_vec', proj_geom['DetectorRowCount'], proj_geom['DetectorColCount'], vectors)

    else:
        raise ValueError(
        'No suitable vector geometry found for type: ' + proj_geom['type'])
    return proj_geom_out
