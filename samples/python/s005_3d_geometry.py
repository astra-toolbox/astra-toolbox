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

try:
    from six.moves import range
except ImportError:
    # six 1.3.0
    from six.moves import xrange as range
import astra
import numpy as np

vol_geom = astra.create_vol_geom(64, 64, 64)


# There are two main 3d projection geometry types: cone beam and parallel beam.
# Each has a regular variant, and a 'vec' variant.
# The 'vec' variants are completely free in the placement of source/detector,
# while the regular variants assume circular trajectories around the z-axis.


# -------------
# Parallel beam
# -------------


# Circular

# Parameters: width of detector column, height of detector row, #rows, #columns
angles = np.linspace(0, 2*np.pi, 48, False)
proj_geom = astra.create_proj_geom('parallel3d', 1.0, 1.0, 32, 64, angles)


# Free

# We generate the same geometry as the circular one above. 
vectors = np.zeros((len(angles), 12))
for i in range(len(angles)):
  # ray direction
  vectors[i,0] = np.sin(angles[i])
  vectors[i,1] = -np.cos(angles[i])
  vectors[i,2] = 0

  # center of detector
  vectors[i,3:6] = 0

  # vector from detector pixel (0,0) to (0,1)
  vectors[i,6] = np.cos(angles[i])
  vectors[i,7] = np.sin(angles[i])
  vectors[i,8] = 0;

  # vector from detector pixel (0,0) to (1,0)
  vectors[i,9] = 0
  vectors[i,10] = 0
  vectors[i,11] = 1

# Parameters: #rows, #columns, vectors
proj_geom = astra.create_proj_geom('parallel3d_vec', 32, 64, vectors)

# ----------
# Cone beam
# ----------


# Circular

# Parameters: width of detector column, height of detector row, #rows, #columns,
#             angles, distance source-origin, distance origin-detector
angles = np.linspace(0, 2*np.pi, 48, False)
proj_geom = astra.create_proj_geom('cone', 1.0, 1.0, 32, 64, angles, 1000, 0)

# Free

vectors = np.zeros((len(angles), 12))
for i in range(len(angles)):
	# source
	vectors[i,0] = np.sin(angles[i]) * 1000
	vectors[i,1] = -np.cos(angles[i]) * 1000
	vectors[i,2] = 0

	# center of detector
	vectors[i,3:6] = 0

	# vector from detector pixel (0,0) to (0,1)
	vectors[i,6] = np.cos(angles[i])
	vectors[i,7] = np.sin(angles[i])
	vectors[i,8] = 0

	# vector from detector pixel (0,0) to (1,0)
	vectors[i,9] = 0
	vectors[i,10] = 0
	vectors[i,11] = 1		

# Parameters: #rows, #columns, vectors
proj_geom = astra.create_proj_geom('cone_vec', 32, 64, vectors)

