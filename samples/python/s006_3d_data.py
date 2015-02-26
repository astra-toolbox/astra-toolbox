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

# Create a 3D volume geometry.
# Parameter order: rows, colums, slices (y, x, z)
vol_geom = astra.create_vol_geom(64, 48, 32)


# Create volumes

# initialized to zero
v0 = astra.data3d.create('-vol', vol_geom)

# initialized to 3.0
v1 = astra.data3d.create('-vol', vol_geom, 3.0)

# initialized to a matrix. A may be a single or double array.
# Coordinate order: slice, row, column (z, y, x)
A = np.zeros((32, 64, 48))
v2 = astra.data3d.create('-vol', vol_geom, A)


# Projection data

# 2 projection directions, along x and y axis resp.
V = np.array([[ 1,0,0, 0,0,0,  0,1,0, 0,0,1],
      [0,1,0, 0,0,0, -1,0,0, 0,0,1]],dtype=np.float)
# 32 rows (v), 64 columns (u)
proj_geom = astra.create_proj_geom('parallel3d_vec', 32, 64, V)

s0 = astra.data3d.create('-proj3d', proj_geom)

# Initialization to a scalar or zero works exactly as with a volume.

# Initialized to a matrix:
# Coordinate order: row (v), angle, column (u)
A = np.zeros((32, 2, 64))
s1 = astra.data3d.create('-proj3d', proj_geom, A)


# Retrieve data:
R = astra.data3d.get(v1)


# Delete all created data objects
astra.data3d.delete(v0)
astra.data3d.delete(v1)
astra.data3d.delete(v2)
astra.data3d.delete(s0)
astra.data3d.delete(s1)
