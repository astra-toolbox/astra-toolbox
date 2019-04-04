import numpy as np
import unittest
import astra
import math
import pylab

DISPLAY=False

def VolumeGeometries(is3D,noncube):
  if not is3D:
    for s in [0.8, 1.0, 1.25]:
      yield astra.create_vol_geom(128, 128, -64*s, 64*s, -64*s, 64*s)
  elif noncube:
    for sx in [0.8, 1.0]:
      for sy in [0.8, 1.0]:
        for sz in [0.8, 1.0]:
          yield astra.create_vol_geom(64, 64, 64, -32*sx, 32*sx, -32*sy, 32*sy, -32*sz, 32*sz)
  else:
    for s in [0.8, 1.0]:
      yield astra.create_vol_geom(64, 64, 64, -32*s, 32*s, -32*s, 32*s, -32*s, 32*s)


def ProjectionGeometries(type):
  if type == 'parallel':
    for dU in [0.8, 1.0, 1.25]:
      yield astra.create_proj_geom('parallel', dU, 256, np.linspace(0,np.pi,180,False))
  elif type == 'fanflat':
    for dU in [0.8, 1.0, 1.25]:
      for src in [500, 1000]:
        for det in [0, 250, 500]:
          yield astra.create_proj_geom('fanflat', dU, 256, np.linspace(0,2*np.pi,180,False), src, det)
  elif type == 'parallel3d':
    for dU in [0.8, 1.0]:
      for dV in [0.8, 1.0]:
        yield astra.create_proj_geom('parallel3d', dU, dV, 128, 128, np.linspace(0,np.pi,180,False))
  elif type == 'cone':
    for dU in [0.8, 1.0]:
      for dV in [0.8, 1.0]:
        for src in [500, 1000]:
          for det in [0, 250]:
            yield astra.create_proj_geom('cone', dU, dV, 128, 128, np.linspace(0,2*np.pi,180,False), src, det)


class TestRecScale(unittest.TestCase):
  def single_test(self, geom_type, proj_type, alg, iters):
    is3D = (geom_type in ['parallel3d', 'cone'])
    for vg in VolumeGeometries(is3D, 'FDK' not in alg):
      for pg in ProjectionGeometries(geom_type):
        if not is3D:
          vol = np.zeros((128,128),dtype=np.float32)
          vol[50:70,50:70] = 1
        else:
          vol = np.zeros((64,64,64),dtype=np.float32)
          vol[25:35,25:35,25:35] = 1
        proj_id = astra.create_projector(proj_type, pg, vg)
        if not is3D:
          sino_id, sinogram = astra.create_sino(vol, proj_id)
        else:
          sino_id, sinogram = astra.create_sino3d_gpu(vol, pg, vg)
        if not is3D:
          DATA = astra.data2d
        else:
          DATA = astra.data3d

        rec_id = DATA.create('-vol', vg, 0.0 if 'EM' not in alg else 1.0)

        cfg = astra.astra_dict(alg)
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = sino_id
        cfg['ProjectorId'] = proj_id
        alg_id = astra.algorithm.create(cfg)

        for i in range(iters):
            astra.algorithm.run(alg_id, 1)
        rec = DATA.get(rec_id)
        astra.astra.delete([sino_id, alg_id, alg_id, proj_id])
        if not is3D:
          val = np.sum(rec[55:65,55:65]) / 100.
        else:
          val = np.sum(rec[27:32,27:32,27:32]) / 125.
        TOL = 5e-2
        if DISPLAY and abs(val-1.0) >= TOL:
          print(geom_type, proj_type, alg, vg, pg)
          print(val)
          pylab.gray()
          if not is3D:
            pylab.imshow(rec)
          else:
            pylab.imshow(rec[:,32,:])
          pylab.show()
        #self.assertTrue(abs(val-1.0) < TOL)


__combinations = {
                   'parallel': [ 'line', 'linear', 'distance_driven', 'strip', 'cuda' ],
                   'fanflat': [ 'line_fanflat', 'strip_fanflat', 'cuda' ],
                   'parallel3d': [ 'cuda3d' ],
                   'cone': [ 'cuda3d' ],
                 }

__algs = {
   'SIRT': 50, 'SART': 10*180, 'CGLS': 30, 'FBP': 1
}

__algs_CUDA = {
  'SIRT_CUDA': 50, 'SART_CUDA': 10*180, 'CGLS_CUDA': 30, 'EM_CUDA': 50,
  'FBP_CUDA': 1
}

__algs_parallel3d = {
  'SIRT3D_CUDA': 200, 'CGLS3D_CUDA': 20,
}

__algs_cone = {
  'SIRT3D_CUDA': 200, 'CGLS3D_CUDA': 20,
  'FDK_CUDA': 1
}



for k, l in __combinations.items():
  for v in l:
    is3D = (k in ['parallel3d', 'cone'])
    if k == 'parallel3d':
      A = __algs_parallel3d
    elif k == 'cone':
      A = __algs_cone
    elif v == 'cuda':
      A = __algs_CUDA
    else:
      A = __algs
    for a, i in A.items():
      def f(k, v, a, i):
        return lambda self: self.single_test(k, v, a, i)
      setattr(TestRecScale, 'test_' + a + '_' + k + '_' + v, f(k,v,a,i))

if __name__ == '__main__':
  unittest.main()

