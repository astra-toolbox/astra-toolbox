# -----------------------------------------------------------------------
# Copyright: 2010-2016, iMinds-Vision Lab, University of Antwerp
#            2013-2016, CWI, Amsterdam
#
# Contact: astra@uantwerpen.be
# Website: http://sf.net/projects/astra-toolbox
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


# TODO: arithmetic (lincomb implemented with numexpr package?)

import astra
import astra.mpi_c as mpi
import numpy as np


def dot(id1, id2):
    """
    Compute dot product of the volumes id1, id2.

    Both IDs must be mpi/distributed data3d volumes of the same
    dimensions.

    :param id1: ID of first volume
    :type id1: :class:`int`
    :param id2: ID of second volume
    :type id2: :class:`int`
    :returns: :class:`float` -- The dot product of the two volumes
    """
    def _dot_internal(id1, id2):
        d1 = astra.data3d.get_shared_local(id1)
        d2 = astra.data3d.get_shared_local(id2)
        # Only look at the slices we're actually responsible
        # for, to avoid duplicating the overlapping slices.
        s = mpi.getObjectResponsibleSlices(id1)
        return np.dot(d1[s].ravel(), d2[s].ravel())
    return sum(mpi.run(_dot_internal, [id1, id2]))

def grad3(src_id, gradX_id, gradY_id, gradZ_id, scale=None):
    """
    Compute discrete gradients in X, Y, Z directions.

    All four IDs must be mpi/distributed data3d volumes of the same
    dimensions.

    If scale is specified, the output is multiplied by scale.

    :param src_id: ID of input volume
    :type src_id: :class:`int`
    :param gradX_id: ID of output volume for gradient in X direction
    :type gradX_id: :class:`int`
    :param gradY_id: ID of output volume for gradient in Y direction
    :type gradY_id: :class:`int`
    :param gradZ_id: ID of output volume for gradient in Z direction
    :type gradZ_id: :class:`int`
    :param scale: if specified, multiply output by this
    :type scale: :class:`float`
    """
    def _grad3_internal(src_id, gradX_id, gradY_id, gradZ_id, scale):
        dataS = astra.data3d.get_shared_local(src_id)
        dataDX = astra.data3d.get_shared_local(gradX_id)
        dataDY = astra.data3d.get_shared_local(gradY_id)
        dataDZ = astra.data3d.get_shared_local(gradZ_id)
        dataDX[:]= dataS
        dataDX[1:,:,:] -= dataS[0:-1,:,:]
        if scale is not None:
            dataDX *= scale
        dataDY[:]= dataS
        dataDY[1:,:,:] -= dataS[0:-1,:,:]
        if scale is not None:
            dataDY *= scale
        dataDZ[:]= dataS
        dataDZ[1:,:,:] -= dataS[0:-1,:,:]
        if scale is not None:
            dataDZ *= scale
    mpi.run(_grad3_internal, [src_id, gradX_id, gradY_id, gradZ_id, scale])
    astra.data3d.sync(gradX_id)
    astra.data3d.sync(gradY_id)
    astra.data3d.sync(gradZ_id)

def grad3_adj(dst_id, gradX_id, gradY_id, gradZ_id, scale=None):
    """
    Compute adjoint of grad3.

    All four IDs must be mpi/distributed data3d volumes of the same
    dimensions.

    If scale is specified, the output is multiplied by scale.

    :param dst_id: ID of output volume
    :type dst_id: :class:`int`
    :param gradX_id: ID of input volume with gradient in X direction
    :type gradX_id: :class:`int`
    :param gradY_id: ID of input volume with gradient in Y direction
    :type gradY_id: :class:`int`
    :param gradZ_id: ID of input volume with gradient in Z direction
    :type gradZ_id: :class:`int`
    :param scale: if specified, multiply output by this
    :type scale: :class:`float`
    """
    def _grad3_adj_internal(dst_id, gradX_id, gradY_id, gradZ_id, scale):
        dataSX = astra.data3d.get_shared_local(gradX_id)
        dataSY = astra.data3d.get_shared_local(gradY_id)
        dataSZ = astra.data3d.get_shared_local(gradZ_id)
        dataD = astra.data3d.get_shared_local(dst_id)
        dataD[:] = dataSX
        dataD[0:-1,:,:] -= dataSX[1:,:,:]
        dataD += dataSY
        dataD[:,0:-1,:] -= dataSY[:,1:,:]
        dataD += dataSZ
        dataD[:,:,0:-1] -= dataSZ[:,:,1:]
        if scale is not None:
            dataD *= scale
    mpi.run(_grad3_adj_internal, [dst_id, gradX_id, gradY_id, gradZ_id, scale])
    astra.data3d.sync(dst_id)

def linear_combination(out_id, a, id1, b, id2):
    """
    Evaluate out_id = a * id1 + b * id2

    All three IDs must be mpi/distributed data3d volumes of the same
    dimensions.

    out_id and/or id1 and/or id2 are allowed to be the same.
    """
    def _lincomb_internal(out_id, a, id1, b, id2):
        # TODO: Allow setting number of threads used by numexpr somehow
        import numexpr
        c = astra.data3d.get_shared_local(id1)
        d = astra.data3d.get_shared_local(id2)
        dst = astra.data3d.get_shared_local(out_id)
        numexpr.evaluate("a*c + b*d", out=dst)
    mpi.run(_lincomb_internal, [out_id, np.float32(a), id1, np.float32(b), id2])

lincomb = linear_combination
