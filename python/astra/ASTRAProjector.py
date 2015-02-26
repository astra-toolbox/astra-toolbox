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

import math
from . import creators as ac
from . import data2d


class ASTRAProjector2DTranspose():
    """Implements the ``proj.T`` functionality.

    Do not use directly, since it can be accessed as member ``.T`` of
    an :class:`ASTRAProjector2D` object.

    """
    def __init__(self, parentProj):
        self.parentProj = parentProj

    def __mul__(self, data):
        return self.parentProj.backProject(data)


class ASTRAProjector2D(object):
    """Helps with various common ASTRA Toolbox 2D operations.

    This class can perform several often used toolbox operations, such as:

    * Forward projecting
    * Back projecting
    * Reconstructing

    Note that this class has a some computational overhead, because it
    copies a lot of data. If you use many repeated operations, directly
    using the PyAstraToolbox methods directly is faster.

    You can use this class as an abstracted weight matrix :math:`W`: multiplying an instance
    ``proj`` of this class by an image results in a forward projection of the image, and multiplying
    ``proj.T`` by a sinogram results in a backprojection of the sinogram::

        proj = ASTRAProjector2D(...)
        fp = proj*image
        bp = proj.T*sinogram

    :param proj_geom: The projection geometry.
    :type proj_geom: :class:`dict`
    :param vol_geom: The volume geometry.
    :type vol_geom: :class:`dict`
    :param proj_type: Projector type, such as ``'line'``, ``'linear'``, ...
    :type proj_type: :class:`string`
    :param useCUDA: If ``True``, use CUDA for calculations, when possible.
    :type useCUDA: :class:`bool`
    """

    def __init__(self, proj_geom, vol_geom, proj_type, useCUDA=False):
        self.vol_geom = vol_geom
        self.recSize = vol_geom['GridColCount']
        self.angles = proj_geom['ProjectionAngles']
        self.nDet = proj_geom['DetectorCount']
        nexpow = int(pow(2, math.ceil(math.log(2 * self.nDet, 2))))
        self.filterSize = nexpow / 2 + 1
        self.nProj = self.angles.shape[0]
        self.proj_geom = proj_geom
        self.proj_id = ac.create_projector(proj_type, proj_geom, vol_geom)
        self.useCUDA = useCUDA
        self.T = ASTRAProjector2DTranspose(self)

    def backProject(self, data):
        """Backproject a sinogram.

        :param data: The sinogram data or ID.
        :type data: :class:`numpy.ndarray` or :class:`int`
        :returns: :class:`numpy.ndarray` -- The backprojection.

        """
        vol_id, vol = ac.create_backprojection(
            data, self.proj_id, useCUDA=self.useCUDA, returnData=True)
        data2d.delete(vol_id)
        return vol

    def forwardProject(self, data):
        """Forward project an image.

        :param data: The image data or ID.
        :type data: :class:`numpy.ndarray` or :class:`int`
        :returns: :class:`numpy.ndarray` -- The forward projection.

        """
        sin_id, sino = ac.create_sino(data, self.proj_id, useCUDA=self.useCUDA, returnData=True)
        data2d.delete(sin_id)
        return sino

    def reconstruct(self, data, method, **kwargs):
        """Reconstruct an image from a sinogram.

        :param data: The sinogram data or ID.
        :type data: :class:`numpy.ndarray` or :class:`int`
        :param method: Name of the reconstruction algorithm.
        :type method: :class:`string`
        :param kwargs: Additional named parameters to pass to :func:`astra.creators.create_reconstruction`.
        :returns: :class:`numpy.ndarray` -- The reconstruction.

        Example of a SIRT reconstruction using CUDA::

            proj = ASTRAProjector2D(...)
            rec = proj.reconstruct(sinogram,'SIRT_CUDA',iterations=1000)

        """
        kwargs['returnData'] = True
        rec_id, rec = ac.create_reconstruction(
            method, self.proj_id, data, **kwargs)
        data2d.delete(rec_id)
        return rec

    def __mul__(self, data):
        return self.forwardProject(data)
