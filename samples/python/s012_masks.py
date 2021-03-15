import astra
import numpy as np

# In this example we will create a reconstruction in a circular region,
# instead of the usual rectangle.

# This is done by placing a circular mask on the square reconstruction volume:

c = np.linspace(-127.5,127.5,256)
x, y = np.meshgrid(c,c)
mask = np.array((x**2 + y**2 < 127.5**2),dtype=np.float)

import pylab
pylab.gray()
pylab.figure(1)
pylab.imshow(mask)

vol_geom = astra.create_vol_geom(256, 256)
proj_geom = astra.create_proj_geom('parallel', 1.0, 384, np.linspace(0,np.pi,50,False))

# As before, create a sinogram from a phantom
import scipy.io
P = scipy.io.loadmat('phantom.mat')['phantom256']
proj_id = astra.create_projector('cuda',proj_geom,vol_geom)
sinogram_id, sinogram = astra.create_sino(P, proj_id)

pylab.figure(2)
pylab.imshow(P)
pylab.figure(3)
pylab.imshow(sinogram)

# Create a data object for the reconstruction
rec_id = astra.data2d.create('-vol', vol_geom)

# Create a data object for the mask
mask_id = astra.data2d.create('-vol', vol_geom, mask)

# Set up the parameters for a reconstruction algorithm using the GPU
cfg = astra.astra_dict('SIRT_CUDA')
cfg['ReconstructionDataId'] = rec_id
cfg['ProjectionDataId'] = sinogram_id
cfg['option'] = {}
cfg['option']['ReconstructionMaskId'] = mask_id

# Create the algorithm object from the configuration structure
alg_id = astra.algorithm.create(cfg)

# Run 150 iterations of the algorithm
astra.algorithm.run(alg_id, 150)

# Get the result
rec = astra.data2d.get(rec_id)

pylab.figure(4)
pylab.imshow(rec)

pylab.show()

# Clean up. Note that GPU memory is tied up in the algorithm object,
# and main RAM in the data objects.
astra.algorithm.delete(alg_id)
astra.data2d.delete(mask_id)
astra.data2d.delete(rec_id)
astra.data2d.delete(sinogram_id)
astra.projector.delete(proj_id)
