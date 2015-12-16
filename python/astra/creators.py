#-----------------------------------------------------------------------
#Copyright 2013 Centrum Wiskunde & Informatica, Amsterdam
#
#Author: Daniel M. Pelt
#Contact: D.M.Pelt@cwi.nl
#Website: http://dmpelt.github.io/pyastratoolbox/
#
#
#This file is part of the Python interface to the
#All Scale Tomographic Reconstruction Antwerp Toolbox ("ASTRA Toolbox").
#
#The Python interface to the ASTRA Toolbox is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#
#The Python interface to the ASTRA Toolbox is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#GNU General Public License for more details.
#
#You should have received a copy of the GNU General Public License
#along with the Python interface to the ASTRA Toolbox. If not, see <http://www.gnu.org/licenses/>.
#
#-----------------------------------------------------------------------

import six
import numpy as np
import math
from . import data2d
from . import data3d
from . import projector
from . import projector3d
from . import algorithm

def astra_dict(intype):
    """Creates a dict to use with the ASTRA Toolbox.

    :param intype: Type of the ASTRA object.
    :type intype: :class:`string`
    :returns: :class:`dict` -- An ASTRA dict of type ``intype``.

    """
    if intype == 'SIRT_CUDA2':
        intype = 'SIRT_CUDA'
        six.print_('SIRT_CUDA2 has been deprecated. Use SIRT_CUDA instead.')
    elif intype == 'FP_CUDA2':
        intype = 'FP_CUDA'
        six.print_('FP_CUDA2 has been deprecated. Use FP_CUDA instead.')
    return {'type': intype}

def create_vol_geom(*varargin):
    """Create a volume geometry structure.

This method can be called in a number of ways:

``create_vol_geom(N)``:
    :returns: A 2D volume geometry of size :math:`N \\times N`.

``create_vol_geom((Y, X))``:
    :returns: A 2D volume geometry of size :math:`Y \\times X`.

``create_vol_geom(Y, X)``:
    :returns: A 2D volume geometry of size :math:`Y \\times X`.

``create_vol_geom(Y, X, minx, maxx, miny, maxy)``:
    :returns: A 2D volume geometry of size :math:`Y \\times X`, windowed as :math:`minx \\leq x \\leq maxx` and :math:`miny \\leq y \\leq maxy`.

``create_vol_geom((Y, X, Z))``:
    :returns: A 3D volume geometry of size :math:`Y \\times X \\times Z`.

``create_vol_geom(Y, X, Z)``:
    :returns: A 3D volume geometry of size :math:`Y \\times X \\times Z`.

``create_vol_geom(Y, X, Z, minx, maxx, miny, maxy, minz, maxz)``:
    :returns: A 3D volume geometry of size :math:`Y \\times X \\times Z`, windowed as :math:`minx \\leq x \\leq maxx` and :math:`miny \\leq y \\leq maxy` and :math:`minz \\leq z \\leq maxz` .


"""
    vol_geom = {'option': {}}
    # astra_create_vol_geom(row_count)
    if len(varargin) == 1 and isinstance(varargin[0], int) == 1:
        vol_geom['GridRowCount'] = varargin[0]
        vol_geom['GridColCount'] = varargin[0]
        vol_geom['option']['WindowMinX'] = -varargin[0] / 2.
        vol_geom['option']['WindowMaxX'] = varargin[0] / 2.
        vol_geom['option']['WindowMinY'] = -varargin[0] / 2.
        vol_geom['option']['WindowMaxY'] = varargin[0] / 2.
    # astra_create_vol_geom([row_count col_count])
    elif len(varargin) == 1 and len(varargin[0]) == 2:
        vol_geom['GridRowCount'] = varargin[0][0]
        vol_geom['GridColCount'] = varargin[0][1]
        vol_geom['option']['WindowMinX'] = -varargin[0][1] / 2.
        vol_geom['option']['WindowMaxX'] = varargin[0][1] / 2.
        vol_geom['option']['WindowMinY'] = -varargin[0][0] / 2.
        vol_geom['option']['WindowMaxY'] = varargin[0][0] / 2.
    # astra_create_vol_geom([row_count col_count slice_count])
    elif len(varargin) == 1 and len(varargin[0]) == 3:
        vol_geom['GridRowCount'] = varargin[0][0]
        vol_geom['GridColCount'] = varargin[0][1]
        vol_geom['GridSliceCount'] = varargin[0][2]
        vol_geom['option']['WindowMinX'] = -varargin[0][1] / 2.
        vol_geom['option']['WindowMaxX'] = varargin[0][1] / 2.
        vol_geom['option']['WindowMinY'] = -varargin[0][0] / 2.
        vol_geom['option']['WindowMaxY'] = varargin[0][0] / 2.
        vol_geom['option']['WindowMinZ'] = -varargin[0][2] / 2.
        vol_geom['option']['WindowMaxZ'] = varargin[0][2] / 2.
    # astra_create_vol_geom(row_count, col_count)
    elif len(varargin) == 2:
        vol_geom['GridRowCount'] = varargin[0]
        vol_geom['GridColCount'] = varargin[1]
        vol_geom['option']['WindowMinX'] = -varargin[1] / 2.
        vol_geom['option']['WindowMaxX'] = varargin[1] / 2.
        vol_geom['option']['WindowMinY'] = -varargin[0] / 2.
        vol_geom['option']['WindowMaxY'] = varargin[0] / 2.
    # astra_create_vol_geom(row_count, col_count, min_x, max_x, min_y, max_y)
    elif len(varargin) == 6:
        vol_geom['GridRowCount'] = varargin[0]
        vol_geom['GridColCount'] = varargin[1]
        vol_geom['option']['WindowMinX'] = varargin[2]
        vol_geom['option']['WindowMaxX'] = varargin[3]
        vol_geom['option']['WindowMinY'] = varargin[4]
        vol_geom['option']['WindowMaxY'] = varargin[5]
    # astra_create_vol_geom(row_count, col_count, slice_count)
    elif len(varargin) == 3:
        vol_geom['GridRowCount'] = varargin[0]
        vol_geom['GridColCount'] = varargin[1]
        vol_geom['GridSliceCount'] = varargin[2]
    # astra_create_vol_geom(row_count, col_count, slice_count, min_x, max_x, min_y, max_y, min_z, max_z)
    elif len(varargin) == 9:
        vol_geom['GridRowCount'] = varargin[0]
        vol_geom['GridColCount'] = varargin[1]
        vol_geom['GridSliceCount'] = varargin[2]
        vol_geom['option']['WindowMinX'] = varargin[3]
        vol_geom['option']['WindowMaxX'] = varargin[4]
        vol_geom['option']['WindowMinY'] = varargin[5]
        vol_geom['option']['WindowMaxY'] = varargin[6]
        vol_geom['option']['WindowMinZ'] = varargin[7]
        vol_geom['option']['WindowMaxZ'] = varargin[8]
    return vol_geom


def create_proj_geom(intype, *args):
    """Create a projection geometry.

This method can be called in a number of ways:

``create_proj_geom('parallel', detector_spacing, det_count, angles)``:

:param detector_spacing: Distance between two adjacent detector pixels.
:type detector_spacing: :class:`float`
:param det_count: Number of detector pixels.
:type det_count: :class:`int`
:param angles: Array of angles in radians.
:type angles: :class:`numpy.ndarray`
:returns: A parallel projection geometry.


``create_proj_geom('fanflat', det_width, det_count, angles, source_origin, origin_det)``:

:param det_width: Size of a detector pixel.
:type det_width: :class:`float`
:param det_count: Number of detector pixels.
:type det_count: :class:`int`
:param angles: Array of angles in radians.
:type angles: :class:`numpy.ndarray`
:param source_origin: Position of the source.
:param origin_det: Position of the detector
:returns: A fan-beam projection geometry.

``create_proj_geom('fanflat_vec', det_count, V)``:

:param det_count: Number of detector pixels.
:type det_count: :class:`int`
:param V: Vector array.
:type V: :class:`numpy.ndarray`
:returns: A fan-beam projection geometry.

``create_proj_geom('parallel3d', detector_spacing_x, detector_spacing_y, det_row_count, det_col_count, angles)``:

:param detector_spacing_*: Distance between two adjacent detector pixels.
:type detector_spacing_*: :class:`float`
:param det_row_count: Number of detector pixel rows.
:type det_row_count: :class:`int`
:param det_col_count: Number of detector pixel columns.
:type det_col_count: :class:`int`
:param angles: Array of angles in radians.
:type angles: :class:`numpy.ndarray`
:returns: A parallel projection geometry.

``create_proj_geom('cone', detector_spacing_x, detector_spacing_y, det_row_count, det_col_count, angles, source_origin, origin_det)``:

:param detector_spacing_*: Distance between two adjacent detector pixels.
:type detector_spacing_*: :class:`float`
:param det_row_count: Number of detector pixel rows.
:type det_row_count: :class:`int`
:param det_col_count: Number of detector pixel columns.
:type det_col_count: :class:`int`
:param angles: Array of angles in radians.
:type angles: :class:`numpy.ndarray`
:param source_origin: Distance between point source and origin.
:type source_origin: :class:`float`
:param origin_det: Distance between the detector and origin.
:type origin_det: :class:`float`
:returns: A cone-beam projection geometry.

``create_proj_geom('cone_vec', det_row_count, det_col_count, V)``:

:param det_row_count: Number of detector pixel rows.
:type det_row_count: :class:`int`
:param det_col_count: Number of detector pixel columns.
:type det_col_count: :class:`int`
:param V: Vector array.
:type V: :class:`numpy.ndarray`
:returns: A cone-beam projection geometry.

``create_proj_geom('parallel3d_vec', det_row_count, det_col_count, V)``:

:param det_row_count: Number of detector pixel rows.
:type det_row_count: :class:`int`
:param det_col_count: Number of detector pixel columns.
:type det_col_count: :class:`int`
:param V: Vector array.
:type V: :class:`numpy.ndarray`
:returns: A parallel projection geometry.

``create_proj_geom('sparse_matrix', det_width, det_count, angles, matrix_id)``:

:param det_width: Size of a detector pixel.
:type det_width: :class:`float`
:param det_count: Number of detector pixels.
:type det_count: :class:`int`
:param angles: Array of angles in radians.
:type angles: :class:`numpy.ndarray`
:param matrix_id: ID of the sparse matrix.
:type matrix_id: :class:`int`
:returns: A projection geometry based on a sparse matrix.

"""
    if intype == 'parallel':
        if len(args) < 3:
            raise Exception(
                'not enough variables: astra_create_proj_geom(parallel, detector_spacing, det_count, angles)')
        return {'type': 'parallel', 'DetectorWidth': args[0], 'DetectorCount': args[1], 'ProjectionAngles': args[2]}
    elif intype == 'fanflat':
        if len(args) < 5:
            raise Exception('not enough variables: astra_create_proj_geom(fanflat, det_width, det_count, angles, source_origin, origin_det)')
        return {'type': 'fanflat', 'DetectorWidth': args[0], 'DetectorCount': args[1], 'ProjectionAngles': args[2], 'DistanceOriginSource': args[3], 'DistanceOriginDetector': args[4]}
    elif intype == 'fanflat_vec':
        if len(args) < 2:
            raise Exception('not enough variables: astra_create_proj_geom(fanflat_vec, det_count, V)')
        if not args[1].shape[1] == 6:
            raise Exception('V should be a Nx6 matrix, with N the number of projections')
        return {'type':'fanflat_vec', 'DetectorCount':args[0], 'Vectors':args[1]}
    elif intype == 'parallel3d':
        if len(args) < 5:
            raise Exception('not enough variables: astra_create_proj_geom(parallel3d, detector_spacing_x, detector_spacing_y, det_row_count, det_col_count, angles)')
        return {'type':'parallel3d', 'DetectorSpacingX':args[0], 'DetectorSpacingY':args[1], 'DetectorRowCount':args[2], 'DetectorColCount':args[3],'ProjectionAngles':args[4]}
    elif intype == 'cone':
        if len(args) < 7:
            raise Exception('not enough variables: astra_create_proj_geom(cone, detector_spacing_x, detector_spacing_y, det_row_count, det_col_count, angles, source_origin, origin_det)')
        return {'type':	'cone','DetectorSpacingX':args[0], 'DetectorSpacingY':args[1], 'DetectorRowCount':args[2],'DetectorColCount':args[3],'ProjectionAngles':args[4],'DistanceOriginSource':	args[5],'DistanceOriginDetector':args[6]}
    elif intype == 'cone_vec':
        if len(args) < 3:
            raise Exception('not enough variables: astra_create_proj_geom(cone_vec, det_row_count, det_col_count, V)')
        if not args[2].shape[1] == 12:
            raise Exception('V should be a Nx12 matrix, with N the number of projections')
        return {'type': 'cone_vec','DetectorRowCount':args[0],'DetectorColCount':args[1],'Vectors':args[2]}
    elif intype == 'parallel3d_vec':
        if len(args) < 3:
            raise Exception('not enough variables: astra_create_proj_geom(parallel3d_vec, det_row_count, det_col_count, V)')
        if not args[2].shape[1] == 12:
            raise Exception('V should be a Nx12 matrix, with N the number of projections')
        return {'type': 'parallel3d_vec','DetectorRowCount':args[0],'DetectorColCount':args[1],'Vectors':args[2]}
    elif intype == 'sparse_matrix':
        if len(args) < 4:
            raise Exception(
                'not enough variables: astra_create_proj_geom(sparse_matrix, det_width, det_count, angles, matrix_id)')
        return {'type': 'sparse_matrix', 'DetectorWidth': args[0], 'DetectorCount': args[1], 'ProjectionAngles': args[2], 'MatrixID': args[3]}
    else:
        raise Exception('Error: unknown type ' + intype)


def create_backprojection(data, proj_id, returnData=True):
    """Create a backprojection of a sinogram (2D).

:param data: Sinogram data or ID.
:type data: :class:`numpy.ndarray` or :class:`int`
:param proj_id: ID of the projector to use.
:type proj_id: :class:`int`
:param returnData: If False, only return the ID of the backprojection.
:type returnData: :class:`bool`
:returns: :class:`int` or (:class:`int`, :class:`numpy.ndarray`) -- If ``returnData=False``, returns the ID of the backprojection. Otherwise, returns a tuple containing the ID of the backprojection and the backprojection itself, in that order.

"""
    proj_geom = projector.projection_geometry(proj_id)
    vol_geom = projector.volume_geometry(proj_id)
    if isinstance(data, np.ndarray):
        sino_id = data2d.create('-sino', proj_geom, data)
    else:
        sino_id = data
    vol_id = data2d.create('-vol', vol_geom, 0)

    if projector.is_cuda(proj_id):
        algString = 'BP_CUDA'
    else:
        algString = 'BP'

    cfg = astra_dict(algString)
    cfg['ProjectorId'] = proj_id
    cfg['ProjectionDataId'] = sino_id
    cfg['ReconstructionDataId'] = vol_id
    alg_id = algorithm.create(cfg)
    algorithm.run(alg_id)
    algorithm.delete(alg_id)

    if isinstance(data, np.ndarray):
        data2d.delete(sino_id)

    if returnData:
        return vol_id, data2d.get(vol_id)
    else:
        return vol_id

def create_backprojection3d_gpu(data, proj_geom, vol_geom, returnData=True):
    """Create a backprojection of a sinogram (3D) using CUDA.

:param data: Sinogram data or ID.
:type data: :class:`numpy.ndarray` or :class:`int`
:param proj_geom: Projection geometry.
:type proj_geom: :class:`dict`
:param vol_geom: Volume geometry.
:type vol_geom: :class:`dict`
:param returnData: If False, only return the ID of the backprojection.
:type returnData: :class:`bool`
:returns: :class:`int` or (:class:`int`, :class:`numpy.ndarray`) -- If ``returnData=False``, returns the ID of the backprojection. Otherwise, returns a tuple containing the ID of the backprojection and the backprojection itself, in that order.

"""
    if isinstance(data, np.ndarray):
        sino_id = data3d.create('-sino', proj_geom, data)
    else:
        sino_id = data

    vol_id = data3d.create('-vol', vol_geom, 0)

    cfg = astra_dict('BP3D_CUDA')
    cfg['ProjectionDataId'] = sino_id
    cfg['ReconstructionDataId'] = vol_id
    alg_id = algorithm.create(cfg)
    algorithm.run(alg_id)
    algorithm.delete(alg_id)

    if isinstance(data, np.ndarray):
        data3d.delete(sino_id)

    if returnData:
        return vol_id, data3d.get(vol_id)
    else:
        return vol_id


def create_sino(data, proj_id, returnData=True, gpuIndex=None):
    """Create a forward projection of an image (2D).

    :param data: Image data or ID.
    :type data: :class:`numpy.ndarray` or :class:`int`
    :param proj_id: ID of the projector to use.
    :type proj_id: :class:`int`
    :param returnData: If False, only return the ID of the forward projection.
    :type returnData: :class:`bool`
    :param gpuIndex: Optional GPU index.
    :type gpuIndex: :class:`int`
    :returns: :class:`int` or (:class:`int`, :class:`numpy.ndarray`)

    If ``returnData=False``, returns the ID of the forward
    projection. Otherwise, returns a tuple containing the ID of the
    forward projection and the forward projection itself, in that
    order.
"""
    proj_geom = projector.projection_geometry(proj_id)
    vol_geom = projector.volume_geometry(proj_id)

    if isinstance(data, np.ndarray):
        volume_id = data2d.create('-vol', vol_geom, data)
    else:
        volume_id = data
    sino_id = data2d.create('-sino', proj_geom, 0)
    if projector.is_cuda(proj_id):
        algString = 'FP_CUDA'
    else:
        algString = 'FP'
    cfg = astra_dict(algString)
    cfg['ProjectorId'] = proj_id
    if gpuIndex is not None:
        cfg['option'] = {'GPUindex': gpuIndex}
    cfg['ProjectionDataId'] = sino_id
    cfg['VolumeDataId'] = volume_id
    alg_id = algorithm.create(cfg)
    algorithm.run(alg_id)
    algorithm.delete(alg_id)

    if isinstance(data, np.ndarray):
        data2d.delete(volume_id)
    if returnData:
        return sino_id, data2d.get(sino_id)
    else:
        return sino_id



def create_sino3d_gpu(data, proj_geom, vol_geom, returnData=True, gpuIndex=None):
    """Create a forward projection of an image (3D).

:param data: Image data or ID.
:type data: :class:`numpy.ndarray` or :class:`int`
:param proj_geom: Projection geometry.
:type proj_geom: :class:`dict`
:param vol_geom: Volume geometry.
:type vol_geom: :class:`dict`
:param returnData: If False, only return the ID of the forward projection.
:type returnData: :class:`bool`
:param gpuIndex: Optional GPU index.
:type gpuIndex: :class:`int`
:returns: :class:`int` or (:class:`int`, :class:`numpy.ndarray`) -- If ``returnData=False``, returns the ID of the forward projection. Otherwise, returns a tuple containing the ID of the forward projection and the forward projection itself, in that order.

"""

    if isinstance(data, np.ndarray):
        volume_id = data3d.create('-vol', vol_geom, data)
    else:
        volume_id = data
    sino_id = data3d.create('-sino', proj_geom, 0)
    algString = 'FP3D_CUDA'
    cfg = astra_dict(algString)
    if not gpuIndex==None:
        cfg['option']={'GPUindex':gpuIndex}
    cfg['ProjectionDataId'] = sino_id
    cfg['VolumeDataId'] = volume_id
    alg_id = algorithm.create(cfg)
    algorithm.run(alg_id)
    algorithm.delete(alg_id)

    if isinstance(data, np.ndarray):
        data3d.delete(volume_id)
    if returnData:
        return sino_id, data3d.get(sino_id)
    else:
        return sino_id


def create_reconstruction(rec_type, proj_id, sinogram, iterations=1, use_mask='no', mask=np.array([]), use_minc='no', minc=0, use_maxc='no', maxc=255, returnData=True, filterType=None, filterData=None):
    """Create a reconstruction of a sinogram (2D).

:param rec_type: Name of the reconstruction algorithm.
:type rec_type: :class:`string`
:param proj_id: ID of the projector to use.
:type proj_id: :class:`int`
:param sinogram: Sinogram data or ID.
:type sinogram: :class:`numpy.ndarray` or :class:`int`
:param iterations: Number of iterations to run.
:type iterations: :class:`int`
:param use_mask: Whether to use a mask.
:type use_mask: ``'yes'`` or ``'no'``
:param mask: Mask data or ID
:type mask: :class:`numpy.ndarray` or :class:`int`
:param use_minc: Whether to force a minimum value on the reconstruction pixels.
:type use_minc: ``'yes'`` or ``'no'``
:param minc: Minimum value to use.
:type minc: :class:`float`
:param use_maxc: Whether to force a maximum value on the reconstruction pixels.
:type use_maxc: ``'yes'`` or ``'no'``
:param maxc: Maximum value to use.
:type maxc: :class:`float`
:param returnData: If False, only return the ID of the reconstruction.
:type returnData: :class:`bool`
:param filterType: Which type of filter to use for filter-based methods.
:type filterType: :class:`string`
:param filterData: Optional filter data for filter-based methods.
:type filterData: :class:`numpy.ndarray`
:returns: :class:`int` or (:class:`int`, :class:`numpy.ndarray`) -- If ``returnData=False``, returns the ID of the reconstruction. Otherwise, returns a tuple containing the ID of the reconstruction and reconstruction itself, in that order.

"""
    proj_geom = projector.projection_geometry(proj_id)
    if isinstance(sinogram, np.ndarray):
        sino_id = data2d.create('-sino', proj_geom, sinogram)
    else:
        sino_id = sinogram
    vol_geom = projector.volume_geometry(proj_id)
    recon_id = data2d.create('-vol', vol_geom, 0)
    cfg = astra_dict(rec_type)
    cfg['ProjectorId'] = proj_id
    cfg['ProjectionDataId'] = sino_id
    cfg['ReconstructionDataId'] = recon_id
    cfg['options'] = {}
    if use_mask == 'yes':
        if isinstance(mask, np.ndarray):
            mask_id = data2d.create('-vol', vol_geom, mask)
        else:
            mask_id = mask
        cfg['options']['ReconstructionMaskId'] = mask_id
    if not filterType == None:
        cfg['FilterType'] = filterType
    if not filterData == None:
        if isinstance(filterData, np.ndarray):
            nexpow = int(
                pow(2, math.ceil(math.log(2 * proj_geom['DetectorCount'], 2))))
            filtSize = nexpow / 2 + 1
            filt_proj_geom = create_proj_geom(
                'parallel', 1.0, filtSize, proj_geom['ProjectionAngles'])
            filt_id = data2d.create('-sino', filt_proj_geom, filterData)
        else:
            filt_id = filterData
        cfg['FilterSinogramId'] = filt_id
    cfg['options']['UseMinConstraint'] = use_minc
    cfg['options']['MinConstraintValue'] = minc
    cfg['options']['UseMaxConstraint'] = use_maxc
    cfg['options']['MaxConstraintValue'] = maxc
    cfg['options']['ProjectionOrder'] = 'random'
    alg_id = algorithm.create(cfg)
    algorithm.run(alg_id, iterations)

    algorithm.delete(alg_id)

    if isinstance(sinogram, np.ndarray):
        data2d.delete(sino_id)
    if use_mask == 'yes' and isinstance(mask, np.ndarray):
        data2d.delete(mask_id)
    if not filterData == None:
        if isinstance(filterData, np.ndarray):
            data2d.delete(filt_id)
    if returnData:
        return recon_id, data2d.get(recon_id)
    else:
        return recon_id


def create_projector(proj_type, proj_geom, vol_geom):
    """Create a 2D projector.

:param proj_type: Projector type, such as ``'line'``, ``'linear'``, ...
:type proj_type: :class:`string`
:param proj_geom: Projection geometry.
:type proj_geom: :class:`dict`
:param vol_geom: Volume geometry.
:type vol_geom: :class:`dict`
:returns: :class:`int` -- The ID of the projector.

"""
    if proj_type == 'blob':
        raise Exception('Blob type not yet implemented')
    cfg = astra_dict(proj_type)
    cfg['ProjectionGeometry'] = proj_geom
    cfg['VolumeGeometry'] = vol_geom
    types3d = ['linear3d', 'linearcone', 'cuda3d']
    if proj_type in types3d:
        return projector3d.create(cfg)
    else:
        return projector.create(cfg)
