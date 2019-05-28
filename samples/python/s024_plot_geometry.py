# -----------------------------------------------------------------------
#   brief             example of usage for astra_plot_geom command
#   - last update     08.05.2019
# -----------------------------------------------------------------------
# Copyright: 2010-2019, imec Vision Lab, University of Antwerp
#            2013-2019, CWI, Amsterdam
#
# Contact: astra@astra-toolbox.com
# Website: http://www.astra-toolbox.com/
#
# This file is part of the ASTRA Toolbox.
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
import matplotlib.pyplot as plt

# proj_geom = astra.create_example('cone', 'normal')
# proj_geom = astra.create_example('cone', 'vec')
proj_geom = astra.create_example('cone', 'helix')
# proj_geom = astra.create_example('cone', 'deform_vec')
# proj_geom = astra.create_example('parallel3d', 'vec')
# proj_geom = astra.create_example('parallel3d')
# proj_geom = astra.create_example('fanflat', 'vec')
# proj_geom = astra.create_example('fanflat')
ax =  astra.plot_geom(proj_geom)

vol_magn = 20
phantom_size = 5.0
phantom_px = 1500.0
vx_size = phantom_size / phantom_px  # voxel size
vol_geom = astra.create_vol_geom(phantom_px, phantom_px, phantom_px)
astra.plot_geom(vol_geom, ax, vx_size=vx_size, Magnification=vol_magn, Color='r')

# the cad model plotting is not supported in python

plt.show()
