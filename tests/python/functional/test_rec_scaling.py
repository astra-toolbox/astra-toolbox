import numpy as np
import astra
import astra.experimental
import math
import pytest

DISPLAY=False

def VolumeGeometries(is3D,noncube,singleslice):
  if not is3D:
    for s in [0.8, 1.0, 1.25]:
      yield astra.create_vol_geom(128, 128, -64*s, 64*s, -64*s, 64*s)
  elif noncube:
    for sx in [0.8, 1.0]:
      for sy in [0.8, 1.0]:
        for sz in [0.8, 1.0]:
          yield astra.create_vol_geom(64, 64, 64, -32*sx, 32*sx, -32*sy, 32*sy, -32*sz, 32*sz)
          if singleslice:
            yield astra.create_vol_geom(64, 64, 1, -32*sx, 32*sx, -32*sy, 32*sy, -0.5*sz, 0.5*sz)
  else:
    for s in [0.8, 1.0]:
      yield astra.create_vol_geom(64, 64, 64, -32*s, 32*s, -32*s, 32*s, -32*s, 32*s)


def ProjectionGeometries(type,shortscan,singleslice):
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
        if singleslice:
          yield astra.create_proj_geom('parallel3d', dU, dV, 1, 128, np.linspace(0,np.pi,180,False))
        else:
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
    A = [1.5, 2] if shortscan else [ 2 ]
    for dU in [0.8, 1.0]:
      for dV in [0.8, 1.0]:
        for src in [500, 1000]:
          for det in [0, 250]:
            for a in A:
              if singleslice:
                yield astra.create_proj_geom('cone', dU, dV, 1, 128, np.linspace(0,a*np.pi,180,False), src, det)
              else:
                yield astra.create_proj_geom('cone', dU, dV, 128, 128, np.linspace(0,a*np.pi,180,False), src, det)
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

@pytest.mark.slow
class TestRecScale:
  def single_test(self, geom_type, proj_type, alg, iters, vss, dss):
    if alg == 'FBP' and 'fanflat' in geom_type:
      pytest.skip('CPU FBP is parallel-beam only')
    is3D = (geom_type in ['parallel3d', 'cone'])
    for vg in VolumeGeometries(is3D, 'FDK' not in alg, False):
      for pg in ProjectionGeometries(geom_type, 'FDK' in alg, False):
        if not is3D:
          vol = np.zeros((128,128),dtype=np.float32)
          vol[50:70,50:70] = 1
        else:
          vol = np.zeros((64,64,64),dtype=np.float32)
          vol[25:35,25:35,25:35] = 1
        options = {}
        if vss > 1:
          options["VoxelSuperSampling"] = vss
        if dss > 1:
          options["DetectorSuperSampling"] = vss
        proj_id = astra.create_projector(proj_type, pg, vg, options=options)
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
        if 'FDK' in alg and geom_type == "cone" and pg["ProjectionAngles"][-1] < 1.8*np.pi:
          cfg['option'] = { 'ShortScan': True }
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
          import pylab
          print(geom_type, proj_type, alg, vg, pg)
          print(val)
          pylab.gray()
          if not is3D:
            pylab.imshow(rec)
          else:
            pylab.imshow(rec[:,32,:])
          pylab.show()
        assert abs(val-1.0) < TOL

  def single_test_adjoint3D(self, geom_type, proj_type):
    for vg in VolumeGeometries(True, True, 'vec' not in geom_type):
      for pg in ProjectionGeometries(geom_type, False, vg['GridSliceCount'] == 1):
        for i in range(5):
          X = np.random.random(astra.geom_size(vg)).astype(np.float32)
          Y = np.random.random(astra.geom_size(pg)).astype(np.float32)
          projector_cfg = astra.astra_dict('cuda3d')
          projector_cfg['ProjectionGeometry'] = pg
          projector_cfg['VolumeGeometry'] = vg
          if vg['GridSliceCount'] == 1:
            projector_cfg['ProjectionKernel'] = '2d_weighting'
          projector_id = astra.projector3d.create(projector_cfg)
          fX = np.zeros(astra.geom_size(pg), dtype=np.float32)
          astra.experimental.direct_FP3D(projector_id, X, fX)
          fTY = np.zeros(astra.geom_size(vg), dtype=np.float32)
          astra.experimental.direct_BP3D(projector_id, fTY, Y)

          astra.projector3d.delete(projector_id)

          da = np.dot(fX.ravel(), Y.ravel())
          db = np.dot(X.ravel(), fTY.ravel())
          m = np.abs(da - db)
          TOL = 1e-1
          if m / da >= TOL:
            print(vg)
            print(pg)
            print(m/da, da/db, da, db)
          assert m / da < TOL





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


__combinations_ss = {
  'parallel': [ { 'projector': 'cuda', 'alg': 'SIRT_CUDA', 'iters': 50 } ],
  'fanflat': [ { 'projector': 'cuda', 'alg': 'SIRT_CUDA', 'iters': 50 } ],
  'parallel3d': [ { 'projector': 'cuda3d', 'alg': 'SIRT3D_CUDA', 'iters': 200 } ],
  'cone': [ { 'projector': 'cuda3d', 'alg': 'SIRT3D_CUDA', 'iters': 200 } ]
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
        return lambda self: self.single_test(k, v, a, i, 1, 1)
      setattr(TestRecScale, 'test_' + a + '_' + k + '_' + v, f(k,v,a,i))

for k, l in __combinations_adjoint.items():
  for v in l:
    def g(k, v):
      return lambda self: self.single_test_adjoint3D(k, v)
    setattr(TestRecScale, 'test_adjoint_' + k + '_' + v, g(k,v))

for k, l in __combinations_ss.items():
  for A in l:
    for vss in [1, 2]:
      for dss in [1, 2]:
        def h(k, v, a, i, vss, dss):
          return lambda self: self.single_test(k, v, a, i, vss, dss)
        setattr(TestRecScale, 'test_ss_' + a + '_' + k + '_' + v + '_' + str(vss) + '_' + str(dss), h(k, A['projector'], A['alg'], A['iters'], vss, dss))

