import astra
import numpy as np

# Set up multi-GPU usage.
# This only works for 3D GPU forward projection and back projection.
astra.set_gpu_index([0,1])

# Optionally, you can also restrict the amount of GPU memory ASTRA will use.
# The line commented below sets this to 1GB.
#astra.set_gpu_index([0,1], memory=1024*1024*1024)

vol_geom = astra.create_vol_geom(1024, 1024, 1024)

angles = np.linspace(0, np.pi, 1024,False)
proj_geom = astra.create_proj_geom('parallel3d', 1.0, 1.0, 1024, 1024, angles)

# Create a simple hollow cube phantom
cube = np.zeros((1024,1024,1024))
cube[128:895,128:895,128:895] = 1
cube[256:767,256:767,256:767] = 0

# Create projection data from this
proj_id, proj_data = astra.create_sino3d_gpu(cube, proj_geom, vol_geom)

# Backproject projection data
bproj_id, bproj_data = astra.create_backprojection3d_gpu(proj_data, proj_geom, vol_geom)

# Clean up. Note that GPU memory is tied up in the algorithm object,
# and main RAM in the data objects.
astra.data3d.delete(proj_id)
astra.data3d.delete(bproj_id)
