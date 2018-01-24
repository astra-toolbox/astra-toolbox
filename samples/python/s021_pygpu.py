# -----------------------------------------------------------------------
# Copyright: 2010-2018, imec Vision Lab, University of Antwerp
#            2013-2018, CWI, Amsterdam
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
import pygpu
import pylab

# Initialize pygpu
ctx = pygpu.init('cuda')
pygpu.set_default_context(ctx)

vol_geom = astra.create_vol_geom(128, 128, 128)
angles = np.linspace(0, 2 * np.pi, 180, False)
proj_geom = astra.create_proj_geom('cone', 1.0, 1.0, 128, 192, angles, 1000, 0)

# Create a simple hollow cube phantom, as a pygpu gpuarray
vol_gpuarr = pygpu.gpuarray.zeros(astra.geom_size(vol_geom), dtype='float32')
vol_gpuarr[17:113, 17:113, 17:113] = 1
vol_gpuarr[33:97, 33:97, 33:97] = 0

# Create a pygpu gpuarray for the output projection data
proj_gpuarr = pygpu.gpuarray.zeros(astra.geom_size(proj_geom), dtype='float32')

# Create the astra GPULink objects and create astra data3d objects from them
z, y, x = proj_gpuarr.shape
proj_data_link = astra.data3d.GPULink(proj_gpuarr.gpudata, x, y, z,
                                      proj_gpuarr.strides[-2])
z, y, x = vol_gpuarr.shape
vol_link = astra.data3d.GPULink(vol_gpuarr.gpudata, x, y, z,
                                vol_gpuarr.strides[-2])

proj_id = astra.data3d.link('-sino', proj_geom, proj_data_link)
vol_id = astra.data3d.link('-vol', vol_geom, vol_link)

# Run a 3D FP
cfg = astra.astra_dict('FP3D_CUDA')
cfg['VolumeDataId'] = vol_id
cfg['ProjectionDataId'] = proj_id
alg_id = astra.algorithm.create(cfg)
astra.algorithm.run(alg_id)

pylab.figure(1)
pylab.gray()
pylab.imshow(proj_gpuarr[:, 20, :])
pylab.show()

astra.algorithm.delete(alg_id)
astra.data3d.delete(vol_id)
astra.data3d.delete(proj_id)
