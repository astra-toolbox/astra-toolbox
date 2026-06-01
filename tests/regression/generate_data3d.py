import numpy as np
import astra
import os
import msgpack
import msgpack_numpy

FORMAT_VERSION = 3

msgpack_numpy.patch()

def gen_vol_mask(name, vol):
    if name is None:
        return None

    if name == 'sphere':
        c = np.linspace(-0.5*vol+0.5, 0.5*vol-0.5, vol)
        x, y, z = np.meshgrid(c, c, c)
        mask = np.array((x**2 + y**2 + z**2 < (0.5*vol-0.5)**2), dtype=np.float32)
    elif name == 'half':
        mask = np.zeros((vol, vol, vol), dtype=np.float32)
        mask[0:(vol//2),:,:] = 1.0
    else:
        raise RuntimeError("Invalid vol_mask name")
    return mask

def gen_proj_mask(name, vol, ang):
    if name is None:
        return None

    mask = np.zeros((vol, ang, vol), dtype=np.float32)
    if name == 'center':
        mask[:,:,(vol//4):(3*vol//4)] = 1.0
    elif name == 'limited-angle':
        mask[:,0:(ang//3),:] = 1.0
    else:
        raise RuntimeError("Invalid proj_mask name")
    return mask


def gen_cuda_rec3d(*, vol=128, ang=128, geom='par', alg=None, opt=None, iters=None, vol_mask=None, proj_mask=None, suffix=''):
    if alg is None:
        raise ValueError()

    vol_geom = astra.create_vol_geom(vol, vol, vol)
    angles = np.linspace(0, 2*np.pi, ang, endpoint=False)
    if geom == 'par':
        proj_geom = astra.create_proj_geom('parallel3d', 1.0, 1.0, vol, vol, angles)
    elif geom == 'cone':
        proj_geom = astra.create_proj_geom('cone', 1.0, 1.0, vol, vol, angles, vol*4, 0)
    else:
        raise ValueError()

    proj_cfg = astra.astra_dict('cuda3d')
    proj_cfg['ProjectionGeometry'] = proj_geom
    proj_cfg['VolumeGeometry'] = vol_geom

    projector_id = astra.projector3d.create(proj_cfg)

    phantom_id = astra.data3d.shepp_logan(vol_geom, returnData=False)

    proj_id, proj = astra.create_sino3d_gpu(phantom_id, proj_geom, vol_geom)

    astra.projector3d.delete(projector_id)
    astra.data3d.delete([proj_id, phantom_id])

    rec_cfg = astra.astra_dict(alg)
    if opt is not None:
        rec_cfg['option'] = opt
    else:
        rec_cfg['option'] = {}

    if alg == 'SART' or alg == 'SART_CUDA':
        if 'ProjectionOrder' not in rec_cfg['option']:
            rec_cfg['option']['ProjectionOrder'] = 'sequential'

    vol_mask_data = gen_vol_mask(vol_mask, vol)
    proj_mask_data = gen_proj_mask(proj_mask, vol, ang)

    if alg == 'EM' or alg == 'EM_CUDA':
        rec_x0 = 1.0
    else:
        rec_x0 = 0.0

    data = {}
    data['version'] = FORMAT_VERSION
    data['type'] = 'in_rec3d'
    data['rec_proj_cfg'] = proj_cfg
    data['rec_cfg'] = rec_cfg
    data['rec_in'] = proj
    data['rec_x0'] = rec_x0
    if iters is not None:
        data['rec_iters'] = iters
    elif alg == 'FDK_CUDA':
        data['rec_iters'] = 1
    else:
        data['rec_iters'] = 10
    if vol_mask_data is not None:
        data['rec_vol_mask'] = vol_mask_data
    if proj_mask_data is not None:
        data['rec_proj_mask'] = proj_mask_data

    name = alg.lower() + '_' + geom + str(vol) + suffix + '.mp'
    print(name)
    #print(data)

    with open(os.path.join('inputs/rec3d', name), 'wb') as F:
      msgpack.pack(data, F, default=msgpack_numpy.encode)

def gen_cuda3d(*, alg, vol=128, volscale=None, projscale=1.0, ang=128, geom='par', opt=None, suffix=''):
    try:
        _ = vol[0]
        name = geom + '_cuda3d' + suffix + '.mp'
    except TypeError:
        name = geom + '_cuda3d_' + str(vol) + suffix + '.mp'
        vol = (vol, vol, vol)

    if volscale is not None:
        vol_geom = astra.create_vol_geom(vol[1], vol[2], vol[0], -vol[2]*volscale*0.5, vol[2]*volscale*0.5, -vol[1]*volscale*0.5, vol[1]*volscale*0.5, -vol[0]*volscale*0.5, vol[0]*volscale*0.5)
    else:
        vol_geom = astra.create_vol_geom(vol[1], vol[2], vol[0])
    angles = np.linspace(0, 2*np.pi, ang, endpoint=False)
    if geom == 'par':
        proj_geom = astra.create_proj_geom('parallel3d', projscale, projscale, vol[0], max(vol[1], vol[2]), angles)
    elif geom == 'parvec':
        proj_geom = astra.create_proj_geom('parallel3d', projscale, projscale, vol[0], max(vol[1], vol[2]), angles)
        proj_geom = astra.geom_2vec(proj_geom)
    elif geom == 'parvecX':
        proj_geom = astra.create_proj_geom('parallel3d', projscale, projscale, vol[0], max(vol[1], vol[2]), angles)
        proj_geom = astra.geom_2vec(proj_geom)
        proj_geom['Vectors'][:,:] = proj_geom['Vectors'][:,(2,1,0,5,4,3,8,7,6,11,10,9)]
    elif geom == 'parvecY':
        proj_geom = astra.create_proj_geom('parallel3d', projscale, projscale, vol[0], max(vol[1], vol[2]), angles)
        proj_geom = astra.geom_2vec(proj_geom)
        proj_geom['Vectors'][:,:] = proj_geom['Vectors'][:,(0,2,1,3,5,4,6,8,7,9,11,10)]
    elif geom == 'cone':
        proj_geom = astra.create_proj_geom('cone', projscale, projscale, vol[0], max(vol[1], vol[2]), angles, max(vol[1],vol[2])*4, 0)
    elif geom == 'conevec':
        proj_geom = astra.create_proj_geom('cone', projscale, projscale, vol[0], max(vol[1], vol[2]), angles, max(vol[1],vol[2])*4, 0)
        proj_geom = astra.geom_2vec(proj_geom)
    elif geom == 'conevecX':
        proj_geom = astra.create_proj_geom('cone', projscale, projscale, vol[0], max(vol[1], vol[2]), angles, max(vol[1],vol[2])*4, 0)
        proj_geom = astra.geom_2vec(proj_geom)
        proj_geom['Vectors'][:,:] = proj_geom['Vectors'][:,(2,1,0,5,4,3,8,7,6,11,10,9)]
    elif geom == 'conevecY':
        proj_geom = astra.create_proj_geom('cone', projscale, projscale, vol[0], max(vol[1], vol[2]), angles, max(vol[1],vol[2])*4, 0)
        proj_geom = astra.geom_2vec(proj_geom)
        proj_geom['Vectors'][:,:] = proj_geom['Vectors'][:,(0,2,1,3,5,4,6,8,7,9,11,10)]
    elif geom == 'cylconevec':
        proj_geom = astra.create_proj_geom('cone', projscale*1.5, projscale*1.5, vol[0], max(vol[1], vol[2]), angles, max(vol[1],vol[2])*4, max(vol[1],vol[2])*2)
        proj_geom = astra.geom_2vec(proj_geom)
        V = np.concatenate((proj_geom['Vectors'], np.ones((ang,1)) * max(vol[1],vol[2])*6),axis=1)
        proj_geom = astra.create_proj_geom('cyl_cone_vec', vol[0], max(vol[1], vol[2]), V)
    else:
        raise ValueError()

    proj_cfg = astra.astra_dict('cuda3d')
    proj_cfg['ProjectionGeometry'] = proj_geom
    proj_cfg['VolumeGeometry'] = vol_geom
    if opt is not None:
        proj_cfg['option'] = opt
    else:
        proj_cfg['option'] = {}


    projector_id = astra.projector3d.create(proj_cfg)

    phantom_id = astra.data3d.shepp_logan(vol_geom, returnData=False)

    if alg == 'fp3d':
        phantom = astra.data3d.get(phantom_id)
        data = {}
        data['version'] = FORMAT_VERSION
        data['type'] = 'in_fp3d'
        data['fp_proj_cfg'] = proj_cfg
        data['fp_in'] = phantom

        print(name)
        #print(data)

        with open(os.path.join('inputs/fp3d', name), 'wb') as F:
            msgpack.pack(data, F, default=msgpack_numpy.encode)

    elif alg == 'bp3d':
        proj_id, proj = astra.create_sino3d_gpu(phantom_id, proj_geom, vol_geom)

        astra.data3d.delete(proj_id)

        data = {}
        data['version'] = FORMAT_VERSION
        data['type'] = 'in_bp3d'
        data['bp_proj_cfg'] = proj_cfg
        data['bp_in'] = proj

        print(name)
        #print(data)

        with open(os.path.join('inputs/bp3d', name), 'wb') as F:
            msgpack.pack(data, F, default=msgpack_numpy.encode)
    else:
        raise RuntimeError()

    astra.projector3d.delete(projector_id)
    astra.data3d.delete(phantom_id)



if True:
    os.makedirs('inputs/rec3d', exist_ok=True)
    gen_cuda_rec3d(alg='FDK_CUDA', geom='cone')
    gen_cuda_rec3d(alg='FDK_CUDA', geom='cone', vol=64, ang=100000, suffix='_100k_angles')
    gen_cuda_rec3d(alg='SIRT3D_CUDA')
    gen_cuda_rec3d(alg='SIRT3D_CUDA', opt = { 'Relaxation': 2.0 }, suffix='_lam2')
    gen_cuda_rec3d(alg='SIRT3D_CUDA', geom='cone')
    gen_cuda_rec3d(alg='CGLS3D_CUDA')
    gen_cuda_rec3d(alg='CGLS3D_CUDA', iters=[3,3,3,3], suffix='_reinit')

    gen_cuda_rec3d(alg='SIRT3D_CUDA', vol_mask='sphere', proj_mask='limited-angle', suffix='_masks')
    gen_cuda_rec3d(alg='SIRT3D_CUDA', vol_mask='half', suffix='_vmask')
    gen_cuda_rec3d(alg='SIRT3D_CUDA', proj_mask='center', suffix='_pmask')

    gen_cuda_rec3d(alg='CGLS3D_CUDA', vol_mask='half', suffix='_vmask')

if True:
    os.makedirs('inputs/bp3d', exist_ok=True)
    gen_cuda3d(alg='bp3d',vol=(20,20,10000),ang=3,geom='cone',suffix='_10k_x')
    gen_cuda3d(alg='bp3d',vol=(20,10000,20),ang=3,geom='cone',suffix='_10k_y')
    gen_cuda3d(alg='bp3d',vol=(10000,20,20),ang=3,geom='cone',suffix='_10k_z')
    gen_cuda3d(alg='bp3d',vol=256,ang=256)
    gen_cuda3d(alg='bp3d',vol=256,ang=256,projscale=1.5,suffix='_scaledproj')
    gen_cuda3d(alg='bp3d',vol=256,ang=256,geom='parvec')
    gen_cuda3d(alg='bp3d',vol=256,ang=256,volscale=0.5,suffix='_scaledvol')
    gen_cuda3d(alg='bp3d',vol=128,ang=128,geom='parvecX')
    gen_cuda3d(alg='bp3d',vol=128,ang=128,geom='parvecY')
    gen_cuda3d(alg='bp3d',vol=600,ang=600)
    gen_cuda3d(alg='bp3d',vol=256,ang=256,geom='cone')
    gen_cuda3d(alg='bp3d',vol=256,ang=256,geom='cone',projscale=1.5,suffix='_scaledproj')
    gen_cuda3d(alg='bp3d',vol=256,ang=256,geom='conevec')
    gen_cuda3d(alg='bp3d',vol=256,ang=256,geom='cone',volscale=0.5,suffix='_scaledvol')
    gen_cuda3d(alg='bp3d',vol=600,ang=600,geom='cone')
    gen_cuda3d(alg='bp3d',vol=128,ang=128,geom='conevecX')
    gen_cuda3d(alg='bp3d',vol=128,ang=128,geom='conevecY')
    gen_cuda3d(alg='bp3d',vol=256,ang=256,geom='cylconevec')

if True:
    os.makedirs('inputs/fp3d', exist_ok=True)
    gen_cuda3d(alg='fp3d',vol=(20,20,10000),ang=3,geom='cone',suffix='_10k_x')
    gen_cuda3d(alg='fp3d',vol=(20,10000,20),ang=3,geom='cone',suffix='_10k_y')
    gen_cuda3d(alg='fp3d',vol=(10000,20,20),ang=3,geom='cone',suffix='_10k_z')
    gen_cuda3d(alg='fp3d',vol=256,ang=256)
    gen_cuda3d(alg='fp3d',vol=256,ang=256,projscale=1.5,suffix='_scaledproj')
    gen_cuda3d(alg='fp3d',vol=256,ang=256,geom='parvec')
    gen_cuda3d(alg='fp3d',vol=256,ang=256,volscale=0.5,suffix='_scaledvol')
    gen_cuda3d(alg='fp3d',vol=128,ang=128,geom='parvecX')
    gen_cuda3d(alg='fp3d',vol=128,ang=128,geom='parvecY')
    gen_cuda3d(alg='fp3d',vol=600,ang=600)
    gen_cuda3d(alg='fp3d',vol=256,ang=256,geom='cone')
    gen_cuda3d(alg='fp3d',vol=256,ang=256,geom='cone',projscale=1.5,suffix='_scaledproj')
    gen_cuda3d(alg='fp3d',vol=256,ang=256,geom='conevec')
    gen_cuda3d(alg='fp3d',vol=256,ang=256,geom='cone',volscale=0.5,suffix='_scaledvol')
    gen_cuda3d(alg='fp3d',vol=600,ang=600,geom='cone')
    gen_cuda3d(alg='fp3d',vol=128,ang=128,geom='conevecX')
    gen_cuda3d(alg='fp3d',vol=128,ang=128,geom='conevecY')
    gen_cuda3d(alg='fp3d',vol=256,ang=256,geom='cylconevec')
