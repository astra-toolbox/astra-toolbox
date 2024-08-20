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

# This example demonstrates using the FP and BP primitives with Matlab's lsqr
# solver. Calls to FP (astra.create_sino) and
# BP (astra.create_backprojection) are wrapped in a function astra_wrap,
# and a handle to this function is passed to lsqr.

# Because in this case the inputs/outputs of FP and BP have to be vectors
# instead of images (matrices), the calls require reshaping to and from vectors.

import astra
import numpy as np

# FP/BP wrapper class
class astra_wrap(object):
    def __init__(self,proj_geom,vol_geom):
        self.proj_id = astra.create_projector('cuda',proj_geom,vol_geom)
        self.shape = (proj_geom['DetectorCount']*len(proj_geom['ProjectionAngles']),vol_geom['GridColCount']*vol_geom['GridRowCount'])
        self.dtype = np.float32
    
    def matvec(self,v):
        sid, s = astra.create_sino(np.reshape(v,(vol_geom['GridRowCount'],vol_geom['GridColCount'])),self.proj_id)
        astra.data2d.delete(sid)
        return s.ravel()
    
    def rmatvec(self,v):
        bid, b = astra.create_backprojection(np.reshape(v,(len(proj_geom['ProjectionAngles']),proj_geom['DetectorCount'],)),self.proj_id)
        astra.data2d.delete(bid)
        return b.ravel()

vol_geom = astra.create_vol_geom(256, 256)
proj_geom = astra.create_proj_geom('parallel', 1.0, 384, np.linspace(0,np.pi,180,False))

# Create a 256x256 phantom image
phantom_id, P = astra.data2d.shepp_logan(vol_geom)

# Create a sinogram using the GPU.
proj_id = astra.create_projector('cuda',proj_geom,vol_geom)
sinogram_id, sinogram = astra.create_sino(P, proj_id)

# Reshape the sinogram into a vector
b = sinogram.ravel()

# Call lsqr with ASTRA FP and BP
import scipy.sparse.linalg
wrapper = astra_wrap(proj_geom,vol_geom)
result = scipy.sparse.linalg.lsqr(wrapper,b,atol=1e-4,btol=1e-4,iter_lim=25)

# Reshape the result into an image
Y = np.reshape(result[0],(vol_geom['GridRowCount'], vol_geom['GridColCount']));

import pylab
pylab.gray()
pylab.imshow(Y)
pylab.show()

astra.data2d.delete(sinogram_id)
astra.data2d.delete(phantom_id)
astra.projector.delete(proj_id)
astra.projector.delete(wrapper.proj_id)

