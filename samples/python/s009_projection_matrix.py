import astra
import numpy as np

vol_geom = astra.create_vol_geom(256, 256)
proj_geom = astra.create_proj_geom('parallel', 1.0, 384, np.linspace(0,np.pi,180,False))

# For CPU-based algorithms, a "projector" object specifies the projection
# model used. In this case, we use the "line" model.
proj_id = astra.create_projector('line', proj_geom, vol_geom)

# Generate the projection matrix for this projection model.
# This creates a matrix W where entry w_{i,j} corresponds to the
# contribution of volume element j to detector element i.
matrix_id = astra.projector.matrix(proj_id)

# Get the projection matrix as a Scipy sparse matrix.
W = astra.matrix.get(matrix_id)


# Manually use this projection matrix to do a projection:
import scipy.io
P = scipy.io.loadmat('phantom.mat')['phantom256']
s = W.dot(P.ravel())
s = np.reshape(s, (len(proj_geom['ProjectionAngles']),proj_geom['DetectorCount']))

import matplotlib.pyplot as plt
plt.imshow(s, cmap="gray")
plt.show()

# Each row of the projection matrix corresponds to a detector element.
# Detector t for angle p is for row 1 + t + p*proj_geom.DetectorCount.
# Each column corresponds to a volume pixel.
# Pixel (x,y) corresponds to column 1 + x + y*vol_geom.GridColCount.


astra.projector.delete(proj_id)
astra.matrix.delete(matrix_id)
