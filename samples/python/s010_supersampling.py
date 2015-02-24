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
proj_geom = astra.create_proj_geom('parallel', 3.0, 128, np.linspace(0,np.pi,180,False))
import scipy.io
P = scipy.io.loadmat('phantom.mat')['phantom256']

# Because the astra.create_sino method does not have support for
# all possible algorithm options, we manually create a sinogram
phantom_id = astra.data2d.create('-vol', vol_geom, P)
sinogram_id = astra.data2d.create('-sino', proj_geom)
cfg = astra.astra_dict('FP_CUDA')
cfg['VolumeDataId'] = phantom_id
cfg['ProjectionDataId'] = sinogram_id

# Set up 3 rays per detector element
cfg['option'] = {}
cfg['option']['DetectorSuperSampling'] = 3

alg_id = astra.algorithm.create(cfg)
astra.algorithm.run(alg_id)
astra.algorithm.delete(alg_id)
astra.data2d.delete(phantom_id)

sinogram3 = astra.data2d.get(sinogram_id)

import pylab
pylab.gray()
pylab.figure(1)
pylab.imshow(P)
pylab.figure(2)
pylab.imshow(sinogram3)

# Create a reconstruction, also using supersampling
rec_id = astra.data2d.create('-vol', vol_geom)
cfg = astra.astra_dict('SIRT_CUDA')
cfg['ReconstructionDataId'] = rec_id
cfg['ProjectionDataId'] = sinogram_id
# Set up 3 rays per detector element
cfg['option'] = {}
cfg['option']['DetectorSuperSampling'] = 3

# There is also an option for supersampling during the backprojection step.
# This should be used if your detector pixels are smaller than the voxels.

# Set up 2 rays per image pixel dimension, for 4 rays total per image pixel.
# cfg['option']['PixelSuperSampling'] = 2


alg_id = astra.algorithm.create(cfg)
astra.algorithm.run(alg_id, 150)
astra.algorithm.delete(alg_id)

rec = astra.data2d.get(rec_id)
pylab.figure(3)
pylab.imshow(rec)
pylab.show()

