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
  elif type == 'parallel3d_vec':
    for j in range(10):
       Vectors = np.zeros([180,12])
       wu = 0.6 + 0.8 * np.random.random()
       wv = 0.6 + 0.8 * np.random.random()
       for i in range(Vectors.shape[0]):
         l = 0.6 + 0.8 * np.random.random()
         angle1 = 2*np.pi*np.random.random()
         angle2 = angle1 + 0.5 * np.random.random()
         angle3 = 0.1*np.pi*np.random.random()
         detc = 10 * np.random.random(size=3)
         detu = [ math.cos(angle1) * wu, math.sin(angle1) * wu, 0 ]
         detv = [ -math.sin(angle1) * math.sin(angle3) * wv, math.cos(angle1) * math.sin(angle3) * wv, math.cos(angle3) * wv ]
         ray = [ math.sin(angle2) * l, -math.cos(angle2) * l, 0 ]
         Vectors[i, :] = [ ray[0], ray[1], ray[2], detc[0], detc[1], detc[2], detu[0], detu[1], detu[2], detv[0], detv[1], detv[2] ]
       pg = astra.create_proj_geom('parallel3d_vec', 128, 128, Vectors)
       yield pg
  elif type == 'cone':
    for dU in [0.8, 1.0]:
      for dV in [0.8, 1.0]:
        for src in [500, 1000]:
          for det in [0, 250]:
            yield astra.create_proj_geom('cone', dU, dV, 128, 128, np.linspace(0,2*np.pi,180,False), src, det)
  elif type == 'cone_vec':
    for j in range(10):
       Vectors = np.zeros([180,12])
       wu = 0.6 + 0.8 * np.random.random()
       wv = 0.6 + 0.8 * np.random.random()
       for i in range(Vectors.shape[0]):
         l = 256 * (0.5 * np.random.random())
         angle1 = 2*np.pi*np.random.random()
         angle2 = angle1 + 0.5 * np.random.random()
         angle3 = 0.1*np.pi*np.random.random()
         detc = 10 * np.random.random(size=3)
         detu = [ math.cos(angle1) * wu, math.sin(angle1) * wu, 0 ]
         detv = [ -math.sin(angle1) * math.sin(angle3) * wv, math.cos(angle1) * math.sin(angle3) * wv, math.cos(angle3) * wv ]
         src = [ math.sin(angle2) * l, -math.cos(angle2) * l, 0 ]
         Vectors[i, :] = [ src[0], src[1], src[2], detc[0], detc[1], detc[2], detu[0], detu[1], detu[2], detv[0], detv[1], detv[2] ]
       pg = astra.create_proj_geom('parallel3d_vec', 128, 128, Vectors)
       yield pg


class TestRecScale(unittest.TestCase):
  def single_test(self, geom_type, proj_type, alg, iters):
    if alg == 'FBP' and 'fanflat' in geom_type:
      self.skipTest('CPU FBP is parallel-beam only')
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
        self.assertTrue(abs(val-1.0) < TOL)

  def single_test_adjoint3D(self, geom_type, proj_type):
    for vg in VolumeGeometries(True, True):
      for pg in ProjectionGeometries(geom_type):
        for i in range(5):
          X = np.random.random(astra.geom_size(vg))
          Y = np.random.random(astra.geom_size(pg))
          proj_id, fX = astra.create_sino3d_gpu(X, pg, vg)
          bp_id, fTY = astra.create_backprojection3d_gpu(Y, pg, vg)

          astra.data3d.delete([proj_id, bp_id])

          da = np.dot(fX.ravel(), Y.ravel())
          db = np.dot(X.ravel(), fTY.ravel())
          m = np.abs(da - db)
          TOL = 1e-1
          if m / da >= TOL:
            print(vg)
            print(pg)
            print(m/da, da/db, da, db)
          self.assertTrue(m / da < TOL)





__combinations = {
  'parallel': [ 'line', 'linear', 'distance_driven', 'strip', 'cuda' ],
  'fanflat': [ 'line_fanflat', 'strip_fanflat', 'cuda' ],
  'parallel3d': [ 'cuda3d' ],
  'cone': [ 'cuda3d' ],
}

__combinations_adjoint = {
  'parallel3d': [ 'cuda3d' ],
  'cone': [ 'cuda3d' ],
  'parallel3d_vec': [ 'cuda3d' ],
  'cone_vec': [ 'cuda3d' ],
}

__algs = {
   'SIRT': 50, 'SART': 10*180, 'CGLS': 30,
   'FBP': 1
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

for k, l in __combinations_adjoint.items():
  for v in l:
    def g(k, v):
      return lambda self: self.single_test_adjoint3D(k, v)
    setattr(TestRecScale, 'test_adjoint_' + k + '_' + v, g(k,v))

if __name__ == '__main__':
  unittest.main()

