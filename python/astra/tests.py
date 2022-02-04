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

from __future__ import print_function, absolute_import

def _basic_par2d_fp(type):
  import astra
  import numpy as np
  vg = astra.create_vol_geom(2, 32)
  pg = astra.create_proj_geom('parallel', 1, 32, [0])
  proj_id = astra.create_projector(type, pg, vg)
  vol = np.random.rand(2, 32)
  (sino_id, sino) = astra.create_sino(vol, proj_id)
  astra.data2d.delete(sino_id)
  astra.projector.delete(proj_id)
  err = np.max(np.abs(sino[0,:] - np.sum(vol,axis=0)))
  return err < 1e-6

def _basic_par3d_fp():
  import astra
  import numpy as np
  vg = astra.create_vol_geom(2, 32, 32)
  pg = astra.create_proj_geom('parallel3d', 1, 1, 32, 32, [0])
  vol = np.random.rand(32, 2, 32)
  (sino_id, sino) = astra.create_sino3d_gpu(vol, pg, vg)
  astra.data3d.delete(sino_id)
  err = np.max(np.abs(sino[:,0,:] - np.sum(vol,axis=1)))
  return err < 1e-6


def _basic_par2d():
  print("Testing basic CPU 2D functionality... ", end="")
  if _basic_par2d_fp('line'):
    print("Ok")
    return True
  else:
    print("Error")
    return False

def _basic_par2d_cuda():
  print("Testing basic CUDA 2D functionality... ", end="")
  if _basic_par2d_fp('cuda'):
    print("Ok")
    return True
  else:
    print("Error")
    return False

def _basic_par3d_cuda():
  print("Testing basic CUDA 3D functionality... ", end="")
  if _basic_par3d_fp():
    print("Ok")
    return True
  else:
    print("Error")
    return False

def test_noCUDA():
  """Perform a very basic functionality test, without CUDA"""

  import astra
  print("ASTRA Toolbox v%s" % (astra.__version__,))
  ok = _basic_par2d()
  if not ok:
    raise RuntimeError("Test failed")

def test_CUDA():
  """Perform a very basic functionality test, including CUDA"""

  import astra
  print("ASTRA Toolbox v%s" % (astra.__version__,))
  print("Getting GPU info... ", end="")
  print(astra.get_gpu_info())
  ok1 = _basic_par2d()
  ok2 = _basic_par2d_cuda()
  ok3 = _basic_par3d_cuda()
  if not (ok1 and ok2 and ok3):
    raise RuntimeError("Test failed")

def test():
  """Perform a very basic functionality test"""

  import astra
  if astra.use_cuda():
    test_CUDA()
  else:
    print("No GPU support available")
    test_noCUDA()
