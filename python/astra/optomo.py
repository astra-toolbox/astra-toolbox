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

from . import data2d
from . import data3d
from . import projector
from . import projector3d
from . import creators
from . import algorithm
from . import functions
import numpy as np
from six.moves import reduce
try:
    from six.moves import range
except ImportError:
    # six 1.3.0
    from six.moves import xrange as range

import operator
import scipy.sparse.linalg

class OpTomo(scipy.sparse.linalg.LinearOperator):
    """Object that imitates a projection matrix with a given projector.

    This object can do forward projection by using the ``*`` operator::

        W = astra.OpTomo(proj_id)
        fp = W*image
        bp = W.T*sinogram

    It can also be used in minimization methods of the :mod:`scipy.sparse.linalg` module::

        W = astra.OpTomo(proj_id)
        output = scipy.sparse.linalg.lsqr(W,sinogram)

    :param proj_id: ID to a projector.
    :type proj_id: :class:`int`
    """

    def __init__(self,proj_id):
        self.dtype = np.float32
        try:
            self.vg = projector.volume_geometry(proj_id)
            self.pg = projector.projection_geometry(proj_id)
            self.data_mod = data2d
            self.appendString = ""
            if projector.is_cuda(proj_id):
                self.appendString += "_CUDA"
        except Exception:
            self.vg = projector3d.volume_geometry(proj_id)
            self.pg = projector3d.projection_geometry(proj_id)
            self.data_mod = data3d
            self.appendString = "3D"
            if projector3d.is_cuda(proj_id):
                self.appendString += "_CUDA"

        self.vshape = functions.geom_size(self.vg)
        self.vsize = reduce(operator.mul,self.vshape)
        self.sshape = functions.geom_size(self.pg)
        self.ssize = reduce(operator.mul,self.sshape)

        self.shape = (self.ssize, self.vsize)

        self.proj_id = proj_id

        self.transposeOpTomo = OpTomoTranspose(self)
        try:
            self.T = self.transposeOpTomo
        except AttributeError:
            # Scipy >= 0.16 defines self.T using self._transpose()
            pass

    def _transpose(self):
        return self.transposeOpTomo

    def __checkArray(self, arr, shp):
        if len(arr.shape)==1:
            arr = arr.reshape(shp)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        if arr.flags['C_CONTIGUOUS']==False:
            arr = np.ascontiguousarray(arr)
        return arr

    def _matvec(self,v):
        """Implements the forward operator.

        :param v: Volume to forward project.
        :type v: :class:`numpy.ndarray`
        """
        v = self.__checkArray(v, self.vshape)
        vid = self.data_mod.link('-vol',self.vg,v)
        s = np.zeros(self.sshape,dtype=np.float32)
        sid = self.data_mod.link('-sino',self.pg,s)

        cfg = creators.astra_dict('FP'+self.appendString)
        cfg['ProjectionDataId'] = sid
        cfg['VolumeDataId'] = vid
        cfg['ProjectorId'] = self.proj_id
        fp_id = algorithm.create(cfg)
        algorithm.run(fp_id)

        algorithm.delete(fp_id)
        self.data_mod.delete([vid,sid])
        return s.flatten()

    def rmatvec(self,s):
        """Implements the transpose operator.

        :param s: The projection data.
        :type s: :class:`numpy.ndarray`
        """
        s = self.__checkArray(s, self.sshape)
        sid = self.data_mod.link('-sino',self.pg,s)
        v = np.zeros(self.vshape,dtype=np.float32)
        vid = self.data_mod.link('-vol',self.vg,v)

        cfg = creators.astra_dict('BP'+self.appendString)
        cfg['ProjectionDataId'] = sid
        cfg['ReconstructionDataId'] = vid
        cfg['ProjectorId'] = self.proj_id
        bp_id = algorithm.create(cfg)
        algorithm.run(bp_id)

        algorithm.delete(bp_id)
        self.data_mod.delete([vid,sid])
        return v.flatten()

    def __mul__(self,v):
        """Provides easy forward operator by *.

        :param v: Volume to forward project.
        :type v: :class:`numpy.ndarray`
        """
        # Catch the case of a forward projection of a 2D/3D image
        if isinstance(v, np.ndarray) and v.shape==self.vshape:
            return self._matvec(v)
        return scipy.sparse.linalg.LinearOperator.__mul__(self, v)

    def reconstruct(self, method, s, iterations=1, extraOptions = {}):
        """Reconstruct an object.

        :param method: Method to use for reconstruction.
        :type method: :class:`string`
        :param s: The projection data.
        :type s: :class:`numpy.ndarray`
        :param iterations: Number of iterations to use.
        :type iterations: :class:`int`
        :param extraOptions: Extra options to use during reconstruction (i.e. for cfg['option']).
        :type extraOptions: :class:`dict`
        """
        s = self.__checkArray(s, self.sshape)
        sid = self.data_mod.link('-sino',self.pg,s)
        v = np.zeros(self.vshape,dtype=np.float32)
        vid = self.data_mod.link('-vol',self.vg,v)
        cfg = creators.astra_dict(method)
        cfg['ProjectionDataId'] = sid
        cfg['ReconstructionDataId'] = vid
        cfg['ProjectorId'] = self.proj_id
        cfg['option'] = extraOptions
        alg_id = algorithm.create(cfg)
        algorithm.run(alg_id,iterations)
        algorithm.delete(alg_id)
        self.data_mod.delete([vid,sid])
        return v

class OpTomoTranspose(scipy.sparse.linalg.LinearOperator):
    """This object provides the transpose operation (``.T``) of the OpTomo object.

    Do not use directly, since it can be accessed as member ``.T`` of
    an :class:`OpTomo` object.
    """
    def __init__(self,parent):
        self.parent = parent
        self.dtype = np.float32
        self.shape = (parent.shape[1], parent.shape[0])
        try:
            self.T = self.parent
        except AttributeError:
            # Scipy >= 0.16 defines self.T using self._transpose()
            pass

    def _matvec(self, s):
        return self.parent.rmatvec(s)

    def rmatvec(self, v):
        return self.parent.matvec(v)

    def _transpose(self):
        return self.parent

    def __mul__(self,s):
        # Catch the case of a backprojection of 2D/3D data
        if isinstance(s, np.ndarray) and s.shape==self.parent.sshape:
            return self._matvec(s)
        return scipy.sparse.linalg.LinearOperator.__mul__(self, s)
