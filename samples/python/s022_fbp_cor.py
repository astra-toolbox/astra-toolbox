import astra
import numpy as np

cor_shift = 3.6

vol_geom = astra.create_vol_geom(256, 256)
proj_geom = astra.create_proj_geom('parallel', 1.0, 256, np.linspace(0,np.pi,180,False))

# Projection geometry with shifted center of rotation
proj_geom_cor = astra.geom_postalignment(proj_geom, cor_shift)

# As before, create a sinogram from a phantom, using the shifted center of rotation
import scipy.io
P = scipy.io.loadmat('phantom.mat')['phantom256']

proj_id_cor = astra.create_projector('cuda',proj_geom_cor,vol_geom)
sinogram_id, sinogram = astra.create_sino(P, proj_id_cor)

# Change the projection geometry metadata attached to the sinogram to standard geometry,
# and try to do a reconstruction, to show the misalignment artifacts caused by
# the shifted center of rotation
astra.data2d.change_geometry(sinogram_id, proj_geom)

import pylab
pylab.gray()
pylab.figure(1)
pylab.imshow(P)
pylab.figure(2)
pylab.imshow(sinogram)

# Create a data object for the reconstruction
rec_id = astra.data2d.create('-vol', vol_geom)

# Set up the parameters for a reconstruction algorithm using the GPU
cfg = astra.astra_dict('FBP_CUDA')
cfg['ReconstructionDataId'] = rec_id
cfg['ProjectionDataId'] = sinogram_id
alg_id = astra.algorithm.create(cfg)
astra.algorithm.run(alg_id)

# Get the result
rec = astra.data2d.get(rec_id)
pylab.figure(3)
pylab.imshow(rec)

astra.algorithm.delete(alg_id)
astra.data2d.delete(rec_id)

# Now change back to the proper, shifted geometry, and do another reconstruction
astra.data2d.change_geometry(sinogram_id, proj_geom_cor)
rec_id = astra.data2d.create('-vol', vol_geom)
cfg = astra.astra_dict('FBP_CUDA')
cfg['ReconstructionDataId'] = rec_id
cfg['ProjectionDataId'] = sinogram_id
alg_id = astra.algorithm.create(cfg)
astra.algorithm.run(alg_id)

# Get the result
rec = astra.data2d.get(rec_id)
pylab.figure(4)
pylab.imshow(rec)
pylab.show()



astra.data2d.delete(sinogram_id)
astra.projector.delete(proj_id_cor)
astra.algorithm.delete(alg_id)
astra.data2d.delete(rec_id)
