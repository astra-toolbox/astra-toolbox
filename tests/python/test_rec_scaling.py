import numpy as np
import unittest
import astra
import math
import pylab

DISPLAY=False

def VolumeGeometries():
  for s in [0.8, 1.0, 1.25]:
    yield astra.create_vol_geom(128, 128, -64*s, 64*s, -64*s, 64*s)

def ProjectionGeometries(type):
  if type == 'parallel':
    for dU in [0.8, 1.0, 1.25]:
      yield astra.create_proj_geom('parallel', dU, 256, np.linspace(0,np.pi,180,False))
  elif type == 'fanflat':
    for dU in [0.8, 1.0, 1.25]:
      for src in [500, 1000]:
        for det in [0, 250, 500]:
          yield astra.create_proj_geom('fanflat', dU, 256, np.linspace(0,2*np.pi,180,False), src, det)


class Test2DRecScale(unittest.TestCase):
  def single_test(self, geom_type, proj_type, alg, iters):
    for vg in VolumeGeometries():
      for pg in ProjectionGeometries(geom_type):
        vol = np.zeros((128,128))
        vol[50:70,50:70] = 1
        proj_id = astra.create_projector(proj_type, pg, vg)
        sino_id, sinogram = astra.create_sino(vol, proj_id)
        rec_id = astra.data2d.create('-vol', vg, 0.0 if 'EM' not in alg else 1.0)

        cfg = astra.astra_dict(alg)
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = sino_id
        cfg['ProjectorId'] = proj_id
        alg_id = astra.algorithm.create(cfg)

        astra.algorithm.run(alg_id, iters)
        rec = astra.data2d.get(rec_id)
        astra.astra.delete([sino_id, alg_id, alg_id, proj_id])
        val = np.sum(rec[55:65,55:65]) / 100.
        TOL = 5e-2
        if DISPLAY and abs(val-1.0) >= TOL:
          print(geom_type, proj_type, alg, vg, pg)
          print(val)
          pylab.gray()
          pylab.imshow(rec)
          pylab.show()
        self.assertTrue(abs(val-1.0) < TOL)


__combinations = {
                   'parallel': [ 'line', 'linear', 'distance_driven', 'strip', 'cuda' ],
                   'fanflat': [ 'line_fanflat', 'strip_fanflat', 'cuda' ],
#                   'fanflat': [ 'cuda' ],
                 }

__algs = {
   'SIRT': 50, 'SART': 10*180, 'CGLS': 30, 'FBP': 1
}

__algs_CUDA = {
  'SIRT_CUDA': 50, 'SART_CUDA': 10*180, 'CGLS_CUDA': 30, 'EM_CUDA': 50,
  'FBP_CUDA': 1
}

for k, l in __combinations.items():
  for v in l:
    A = __algs if v != 'cuda' else __algs_CUDA
    for a, i in A.items():
      def f(k, v, a, i):
        return lambda self: self.single_test(k, v, a, i)
      setattr(Test2DRecScale, 'test_' + a + '_' + k + '_' + v, f(k,v,a,i))
 
if __name__ == '__main__':
  unittest.main()

