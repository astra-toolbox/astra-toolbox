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
