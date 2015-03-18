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

import astra
import numpy as np

vol_geom = astra.create_vol_geom(256, 256)
proj_geom = astra.create_proj_geom('parallel', 1.0, 384, np.linspace(0,np.pi,180,False))
import scipy.io
P = scipy.io.loadmat('phantom.mat')['phantom256']

proj_id = astra.create_projector('cuda',proj_geom,vol_geom)

# Create a sinogram from a phantom, using GPU #1. (The default is #0)
sinogram_id, sinogram = astra.create_sino(P, proj_id, gpuIndex=1)


# Set up the parameters for a reconstruction algorithm using the GPU
rec_id = astra.data2d.create('-vol', vol_geom)
cfg = astra.astra_dict('SIRT_CUDA')
cfg['ReconstructionDataId'] = rec_id
cfg['ProjectionDataId'] = sinogram_id

# Use GPU #1 for the reconstruction. (The default is #0.)
cfg['option'] = {}
cfg['option']['GPUindex'] = 1

# Run 150 iterations of the algorithm
alg_id = astra.algorithm.create(cfg)
astra.algorithm.run(alg_id, 150)
rec = astra.data2d.get(rec_id)


# Clean up.
astra.algorithm.delete(alg_id)
astra.data2d.delete(rec_id)
astra.data2d.delete(sinogram_id)
astra.projector.delete(proj_id)
