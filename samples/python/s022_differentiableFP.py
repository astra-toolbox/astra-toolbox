# -----------------------------------------------------------------------
# Copyright: 2010-2016, iMinds-Vision Lab, University of Antwerp
#            2013-2016, CWI, Amsterdam
#
# Contact: astra@uantwerpen.be
# Website: http://www.astra-toolbox.com/
#
# This file is part of the ASTRA Toolbox.
#
#
# The ASTRA Toolbox is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# The ASTRA Toolbox is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with the ASTRA Toolbox. If not, see <http://www.gnu.org/licenses/>.
#
# -----------------------------------------------------------------------

import astra
import numpy as np
import pylab as pl
from random import random


# Make hollow cube test object
N_obj = 256
cube = np.zeros((N_obj, N_obj, N_obj))
cube[N_obj/4:3*N_obj/4,N_obj/4:3*N_obj/4,N_obj/4:3*N_obj/4] = 1.0
cube[3*N_obj/8:5*N_obj/8,3*N_obj/8:5*N_obj/8,3*N_obj/8:5*N_obj/8] = 0.0


# Projection- and volume geometry
angle = 2*np.pi*random()    # Random tomographic angle
proj_geom = astra.create_proj_geom('parallel3d', 1.0, 1.0, N_obj, N_obj, angle)
#proj_geom = astra.create_proj_geom('cone', 2.0, 2.0, N_obj, N_obj, angle, 2*N_obj, 2*N_obj)    # uncomment for cone-beam test case
vol_geom = astra.create_vol_geom(N_obj, N_obj, N_obj)


# Compute forward projections
_, proj = astra.create_sino3d_gpu(cube, proj_geom, vol_geom)
_, proj_bicubic = astra.create_sino3d_gpu(cube, proj_geom, vol_geom, projectionKernel='bicubic')
_, proj_bicubic_dd1 = astra.create_sino3d_gpu(cube, proj_geom, vol_geom, projectionKernel='bicubic_derivative_1')
_, proj_bicubic_dd2 = astra.create_sino3d_gpu(cube, proj_geom, vol_geom, projectionKernel='bicubic_derivative_2')
s

# Plot results
fig_default = pl.figure(1)
fig_default.canvas.set_window_title("Forward projection with default bilinear texture-interpolation")
pl.imshow(proj[:,0,:])
pl.colorbar()

fig_bicubic = pl.figure(2)
fig_bicubic.canvas.set_window_title("Forward projection with bicubic texture-interpolation")
pl.imshow(proj_bicubic[:,0,:])
pl.colorbar()

fig_bicubic_dd1 = pl.figure(3)
fig_bicubic_dd1.canvas.set_window_title("Forward projection using bicubic differentiation along dimension 1")
pl.imshow(proj_bicubic_dd1[:,0,:])
pl.colorbar()

fig_bicubic_dd2 = pl.figure(4)
fig_bicubic_dd2.canvas.set_window_title("Forward projection using bicubic differentiation along dimension 2")
pl.imshow(proj_bicubic_dd2[:,0,:])
pl.colorbar()

pl.show()

