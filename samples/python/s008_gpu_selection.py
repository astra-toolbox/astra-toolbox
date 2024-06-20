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

# Switch GPU to GPU #1. The default is #0.
astra.set_gpu_index(1)

vol_geom = astra.create_vol_geom(256, 256)
proj_geom = astra.create_proj_geom('parallel', 1.0, 384, np.linspace(0,np.pi,180,False))
phantom_id = astra.data2d.shepp_logan(vol_geom, returnData=False)

proj_id = astra.create_projector('cuda',proj_geom,vol_geom)

# Create a sinogram from a phantom
sinogram_id, sinogram = astra.create_sino(P, proj_id)


# Set up the parameters for a reconstruction algorithm using the GPU
rec_id = astra.data2d.create('-vol', vol_geom)
cfg = astra.astra_dict('SIRT_CUDA')
cfg['ReconstructionDataId'] = rec_id
cfg['ProjectionDataId'] = sinogram_id

# Run 150 iterations of the algorithm
alg_id = astra.algorithm.create(cfg)
astra.algorithm.run(alg_id, 150)
rec = astra.data2d.get(rec_id)


# Clean up.
astra.algorithm.delete(alg_id)
astra.data2d.delete(rec_id)
astra.data2d.delete(sinogram_id)
astra.data2d.delete(phantom_id)
astra.projector.delete(proj_id)
