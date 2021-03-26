import astra
import numpy as np

vol_geom = astra.create_vol_geom(256, 256)

proj_geom = astra.create_proj_geom('parallel', 1.0, 384, np.linspace(0,np.pi,180,False))


# Create volumes

# initialized to zero
v0 = astra.data2d.create('-vol', vol_geom)

# initialized to 3.0
v1 = astra.data2d.create('-vol', vol_geom, 3.0)

# initialized to a matrix. A may be a single, double or logical (0/1) array.
import scipy.io
A = scipy.io.loadmat('phantom.mat')['phantom256']
v2 = astra.data2d.create('-vol', vol_geom, A)


# Projection data
s0 = astra.data2d.create('-sino', proj_geom)
# Initialization to a scalar or a matrix also works, exactly as with a volume.


# Update data

# set to zero
astra.data2d.store(v0, 0)

# set to a matrix
astra.data2d.store(v2, A)



# Retrieve data

R = astra.data2d.get(v2)

import matplotlib.pyplot as plt
plt.imshow(R, cmap="gray")
plt.show()


# Free memory
astra.data2d.delete(v0)
astra.data2d.delete(v1)
astra.data2d.delete(v2)
astra.data2d.delete(s0)