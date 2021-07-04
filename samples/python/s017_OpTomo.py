import astra
import numpy as np
import scipy.sparse.linalg

vol_geom = astra.create_vol_geom(256, 256)
proj_geom = astra.create_proj_geom('parallel', 1.0, 384, np.linspace(0,np.pi,180,False))

# As before, create a sinogram from a phantom
import scipy.io
P = scipy.io.loadmat('phantom.mat')['phantom256']
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
