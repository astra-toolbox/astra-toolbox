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

try:
    from six.moves import range
except ImportError:
    # six 1.3.0
    from six.moves import xrange as range
import astra
import numpy as np


vol_geom = astra.create_vol_geom(256, 256)
proj_geom = astra.create_proj_geom('parallel', 1.0, 384, np.linspace(0,np.pi,180,False))

# As before, create a sinogram from a phantom
import scipy.io
P = scipy.io.loadmat('phantom.mat')['phantom256']
proj_id = astra.create_projector('cuda',proj_geom,vol_geom)
sinogram_id, sinogram = astra.create_sino(P, proj_id)

import pylab
pylab.gray()
pylab.figure(1)
pylab.imshow(P)
pylab.figure(2)
pylab.imshow(sinogram)

# Create a data object for the reconstruction
rec_id = astra.data2d.create('-vol', vol_geom)

# Set up the parameters for a reconstruction algorithm using the GPU
cfg = astra.astra_dict('SIRT_CUDA')
cfg['ReconstructionDataId'] = rec_id
cfg['ProjectionDataId'] = sinogram_id

# Create the algorithm object from the configuration structure
alg_id = astra.algorithm.create(cfg)

# Run 1500 iterations of the algorithm one at a time, keeping track of errors
nIters = 1500
phantom_error = np.zeros(nIters)
residual_error = np.zeros(nIters)
for i in range(nIters):
  # Run a single iteration
  astra.algorithm.run(alg_id, 1)
  residual_error[i] = astra.algorithm.get_res_norm(alg_id)
  rec = astra.data2d.get(rec_id)
  phantom_error[i] = np.sqrt(((rec - P)**2).sum())

# Get the result
rec = astra.data2d.get(rec_id)
pylab.figure(3)
pylab.imshow(rec)

pylab.figure(4)
pylab.plot(residual_error)
pylab.figure(5)
pylab.plot(phantom_error)

pylab.show()

# Clean up.
astra.algorithm.delete(alg_id)
astra.data2d.delete(rec_id)
astra.data2d.delete(sinogram_id)
astra.projector.delete(proj_id)
