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
from astra.experimental import do_composite_FP

astra.log.setOutputScreen(astra.log.STDERR, astra.log.DEBUG)

# low res part (voxels of 4x4x4)
vol_geom1 = astra.create_vol_geom(32, 16, 32, -64, 0, -64, 64, -64, 64)

# high res part (voxels of 1x1x1)
vol_geom2 = astra.create_vol_geom(128, 64, 128, 0, 64, -64, 64, -64, 64)


# Split the output in two parts as well, for demonstration purposes
angles1 = np.linspace(0, np.pi/2, 90, False)
angles2 = np.linspace(np.pi/2, np.pi, 90, False)
proj_geom1 = astra.create_proj_geom('parallel3d', 1.0, 1.0, 128, 192, angles1)
proj_geom2 = astra.create_proj_geom('parallel3d', 1.0, 1.0, 128, 192, angles2)

# Create a simple hollow cube phantom
cube1 = np.zeros((32,32,16))
cube1[4:28,4:28,4:16] = 1

cube2 = np.zeros((128,128,64))
cube2[16:112,16:112,0:112] = 1
cube2[33:97,33:97,4:28] = 0

vol1 = astra.data3d.create('-vol', vol_geom1, cube1)
vol2 = astra.data3d.create('-vol', vol_geom2, cube2)

proj1 = astra.data3d.create('-proj3d', proj_geom1, 0)
proj2 = astra.data3d.create('-proj3d', proj_geom2, 0)

# The actual geometries don't matter for this composite FP/BP case
projector = astra.create_projector('cuda3d', proj_geom1, vol_geom1)

do_composite_FP(projector, [vol1, vol2], [proj1, proj2])

proj_data1 = astra.data3d.get(proj1)
proj_data2 = astra.data3d.get(proj2)

# Display a single projection image
import pylab
pylab.gray()
pylab.figure(1)
pylab.imshow(proj_data1[:,0,:])
pylab.figure(2)
pylab.imshow(proj_data2[:,0,:])
pylab.show()


# Clean up. Note that GPU memory is tied up in the algorithm object,
# and main RAM in the data objects.
astra.data3d.delete(vol1)
astra.data3d.delete(vol2)
astra.data3d.delete(proj1)
astra.data3d.delete(proj2)
astra.projector3d.delete(projector)
