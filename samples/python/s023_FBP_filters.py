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
import scipy.io


# This sample script illustrates three ways of passing filters to FBP.
# They work with both the FBP (CPU) and the FBP_CUDA (GPU) algorithms.


N = 256

vol_geom = astra.create_vol_geom(N, N)
proj_geom = astra.create_proj_geom('parallel', 1.0, N, np.linspace(0,np.pi,180,False))

P = scipy.io.loadmat('phantom.mat')['phantom256']

proj_id = astra.create_projector('strip',proj_geom,vol_geom)
sinogram_id, sinogram = astra.create_sino(P, proj_id)

rec_id = astra.data2d.create('-vol', vol_geom)
cfg = astra.astra_dict('FBP')
cfg['ReconstructionDataId'] = rec_id
cfg['ProjectionDataId'] = sinogram_id
cfg['ProjectorId'] = proj_id
cfg['option'] = {}



# 1. Use a standard Ram-Lak filter
cfg['option']['FilterType'] = 'ram-lak'

alg_id = astra.algorithm.create(cfg)
astra.algorithm.run(alg_id)
rec_RL = astra.data2d.get(rec_id)
astra.algorithm.delete(alg_id)

# 2. Define a filter in Fourier space
# This is assumed to be symmetric, and ASTRA therefore expects only half

# The full filter size should be the smallest power of two that is at least
# twice the number of detector pixels.
fullFilterSize = 2*N
kernel = np.append( np.linspace(0, 1, fullFilterSize//2, endpoint=False), np.linspace(1, 0, fullFilterSize//2, endpoint=False) )
halfFilterSize = fullFilterSize // 2 + 1
filter = np.reshape(kernel[0:halfFilterSize], (1, halfFilterSize))

filter_geom = astra.create_proj_geom('parallel', 1.0, halfFilterSize, [0]);
filter_id = astra.data2d.create('-sino', filter_geom, filter);

cfg['option']['FilterType'] = 'projection'
cfg['option']['FilterSinogramId'] = filter_id
alg_id = astra.algorithm.create(cfg)
astra.algorithm.run(alg_id)
rec_filter = astra.data2d.get(rec_id)
astra.algorithm.delete(alg_id)


# 3. Define a (spatial) convolution kernel directly
# For a kernel of odd size 2*k+1, the central component is at kernel[k]
# For a kernel of even size 2*k, the central component is at kernel[k]
kernel = np.zeros((1, N))
for i in range(0,N//4):
    f = np.pi * (2*i + 1)
    val = -2.0 / (f * f)
    kernel[0, N//2 + (2*i+1)] = val
    kernel[0, N//2 - (2*i+1)] = val
kernel[0, N//2] = 0.5
kernel_geom = astra.create_proj_geom('parallel', 1.0, N, [0]);
kernel_id = astra.data2d.create('-sino', kernel_geom, kernel);

cfg['option']['FilterType'] = 'rprojection'
cfg['option']['FilterSinogramId'] = kernel_id
alg_id = astra.algorithm.create(cfg)
astra.algorithm.run(alg_id)
rec_kernel = astra.data2d.get(rec_id)
astra.algorithm.delete(alg_id)

import pylab
pylab.figure()
pylab.imshow(P)
pylab.figure()
pylab.imshow(rec_RL)
pylab.figure()
pylab.imshow(rec_filter)
pylab.figure()
pylab.imshow(rec_kernel)
pylab.show()

astra.data2d.delete(rec_id)
astra.data2d.delete(sinogram_id)
astra.projector.delete(proj_id)

