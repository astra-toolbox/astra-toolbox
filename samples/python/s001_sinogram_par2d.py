import astra
import numpy as np

# Create a basic 256x256 square volume geometry
vol_geom = astra.create_vol_geom(256, 256)

# Create a parallel beam geometry with 180 angles between 0 and pi, and
# 384 detector pixels of width 1.
# For more details on available geometries, see the online help of the
# function astra_create_proj_geom .
proj_geom = astra.create_proj_geom('parallel', 1.0, 384, np.linspace(0,np.pi,180,False))

# Load a 256x256 phantom image
import scipy.io
P = scipy.io.loadmat('phantom.mat')['phantom256']

# Create a sinogram using the GPU.
# Note that the first time the GPU is accessed, there may be a delay
# of up to 10 seconds for initialization.
proj_id = astra.create_projector('cuda',proj_geom,vol_geom)
sinogram_id, sinogram = astra.create_sino(P, proj_id)

import matplotlib.pyplot as plt
plt.subplots(1,2,1)
plt.imshow(P, cmap="gray")
plt.subplots(1,2,2)
plt.imshow(sinogram)
plt.show()


# Free memory
astra.data2d.delete(sinogram_id)
astra.projector.delete(proj_id)
