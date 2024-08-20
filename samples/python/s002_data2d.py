# -----------------------------------------------------------------------
# Copyright: 2010-2022, imec Vision Lab, University of Antwerp
#            2013-2022, CWI, Amsterdam
#
# Contact: astra@astra-toolbox.com
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

vol_geom = astra.create_vol_geom(256, 256)

proj_geom = astra.create_proj_geom('parallel', 1.0, 384, np.linspace(0,np.pi,180,False))


# Create volumes

# initialized to zero
v0 = astra.data2d.create('-vol', vol_geom)

# initialized to 3.0
v1 = astra.data2d.create('-vol', vol_geom, 3.0)

# initialized to a matrix. A may be a single, double or logical (0/1) array.
phantom_id, A = astra.data2d.shepp_logan(vol_geom)

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
import pylab
pylab.gray()
pylab.imshow(R)
pylab.show()


# Free memory
astra.data2d.delete(v0)
astra.data2d.delete(v1)
astra.data2d.delete(v2)
astra.data2d.delete(s0)
astra.data2d.delete(phantom_id)
