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

# Set up multi-GPU usage.
# This only works for 3D GPU forward projection and back projection.
astra.astra.set_gpu_index([0,1])

# Optionally, you can also restrict the amount of GPU memory ASTRA will use.
# The line commented below sets this to 1GB.
#astra.astra.set_gpu_index([0,1], memory=1024*1024*1024)

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
