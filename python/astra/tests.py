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

def _test_filterData_none(rec_type='SIRT'):
  import astra
  import numpy as np
  vg = astra.create_vol_geom(64, 64)
  pg = astra.create_proj_geom('parallel', 1.0, 128,
    np.linspace(0, np.pi, 180, endpoint=False, dtype=np.float64))
  proj_id = astra.create_projector('linear', pg, vg)
  sino = np.zeros((180, 128), dtype=np.float32)
  rec_id = astra.create_reconstruction(
    rec_type, proj_id, sino, iterations=1,
    filterType=None, filterData=None, returnData=False)
  astra.algorithm.delete(rec_id)
  astra.projector.delete(proj_id)
  return True

def _test_filterData_ndarray(rec_type='SIRT'):
  import astra
  import numpy as np
  vg = astra.create_vol_geom(64, 64)
  pg = astra.create_proj_geom('parallel', 1.0, 128,
    np.linspace(0, np.pi, 180, endpoint=False, dtype=np.float64))
  proj_id = astra.create_projector('linear', pg, vg)
  sino = np.zeros((180, 128), dtype=np.float32)
  # Shape is (angles, filtSize) where filtSize = nexpow // 2 + 1
  filt_id = np.ones((180, 129), dtype=np.float32)
  rec_id = astra.create_reconstruction(
    rec_type, proj_id, sino, iterations=1,
    filterType='projection', filterData=filt_id, returnData=False)
  astra.algorithm.delete(rec_id)
  astra.projector.delete(proj_id)
  return True

def _test_filterData_filtSize_is_int():
  import astra
  import numpy as np
  import math
  pg = astra.create_proj_geom('parallel', 1.0, 128,
    np.linspace(0, np.pi, 180, endpoint=False, dtype=np.float64))
  nexpow = int(pow(2, math.ceil(math.log(2 * pg['DetectorCount'], 2))))
  filt_size = nexpow // 2 + 1
  filt_pg = astra.create_proj_geom('parallel', 1.0, filt_size, pg['ProjectionAngles'])
  assert isinstance(filt_size, int), "filtSize should be int, got %s" % type(filt_size)
  assert isinstance(filt_pg['DetectorCount'], int), \
    "DetectorCount should be int, got %s" % type(filt_pg['DetectorCount'])
  return True

def test_filterData():
  """Test filterData parameter handling in create_reconstruction"""

  import astra
  import numpy as np
  print("Testing filterData=None (no filter)... ", end="", flush=True)
  assert _test_filterData_none()
  print("Ok")
  print("Testing filterData=np.ndarray (custom filter)... ", end="", flush=True)
  try:
    assert _test_filterData_ndarray()
    print("Ok")
  except ValueError as e:
    print("FAILED: %s" % e)
    raise
  print("Testing filtSize integer type... ", end="", flush=True)
  assert _test_filterData_filtSize_is_int()
  print("Ok")


def test():
  """Perform a very basic functionality test"""

  import astra
  if astra.use_cuda():
    test_CUDA()
  else:
    print("No GPU support available")
    test_noCUDA()
