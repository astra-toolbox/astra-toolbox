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

import astra
import numpy as np
import scipy.sparse.linalg

vol_geom = astra.create_vol_geom(256, 256)
proj_geom = astra.create_proj_geom('parallel', 1.0, 384, np.linspace(0,np.pi,180,False))

# As before, create a sinogram from a phantom
phantom_id, P = astra.data2d.shepp_logan(vol_geom)
proj_id = astra.create_projector('cuda',proj_geom,vol_geom)

# construct the OpTomo object
W = astra.OpTomo(proj_id)

sinogram = W * P
sinogram = sinogram.reshape([180, 384])

import pylab
pylab.gray()
pylab.figure(1)
pylab.imshow(P)
pylab.figure(2)
pylab.imshow(sinogram)

# Run the lsqr linear solver
output = scipy.sparse.linalg.lsqr(W, sinogram.ravel(), iter_lim=150)
rec = output[0].reshape([256, 256])

pylab.figure(3)
pylab.imshow(rec)
pylab.show()

# Clean up.
astra.projector.delete(proj_id)
astra.data2d.delete(phantom_id)
