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

    if name == 'circle':
        c = np.linspace(-0.5*vol+0.5, 0.5*vol-0.5, vol)
        x, y = np.meshgrid(c, c)
        mask = np.array((x**2 + y**2 < (0.5*vol-0.5)**2), dtype=np.float32)
    elif name == 'half':
        mask = np.zeros((vol, vol), dtype=np.float32)
        mask[0:(vol//2),:] = 1.0
    else:
        raise RuntimeError("Invalid vol_mask name")
    return mask

def gen_proj_mask(name, vol, ang):
    if name is None:
        return None

    mask = np.zeros((ang, vol), dtype=np.float32)
    if name == 'center':
        mask[:,(vol//4):(3*vol//4)] = 1.0
    elif name == 'limited-angle':
        mask[0:(ang//3),:] = 1.0
    else:
        raise RuntimeError("Invalid proj_mask name")
    return mask


def gen_cuda_rec2d(*, vol=128, ang=128, geom='par', alg=None, opt=None, iters=None, vol_mask=None, proj_mask=None, suffix=''):
    if alg is None:
        raise ValueError()

    vol_geom = astra.create_vol_geom(vol, vol)
    angles = np.linspace(0, 2*np.pi, ang, endpoint=False)
    if geom == 'par':
        proj_geom = astra.create_proj_geom('parallel', 1.0, vol, angles)
    elif geom == 'fan':
        proj_geom = astra.create_proj_geom('fanflat', 1.0, vol, angles, vol*4, 0)
    else:
        raise ValueError()

    proj_cfg = astra.astra_dict('cuda')
    proj_cfg['ProjectionGeometry'] = proj_geom
    proj_cfg['VolumeGeometry'] = vol_geom

    projector_id = astra.projector.create(proj_cfg)

    phantom_id = astra.data2d.shepp_logan(vol_geom, returnData=False)

    proj_id, proj = astra.create_sino(phantom_id, projector_id)

    astra.projector.delete(projector_id)
    astra.data2d.delete([proj_id, phantom_id])

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
    data['type'] = 'in_rec2d'
    data['rec_proj_cfg'] = proj_cfg
    data['rec_cfg'] = rec_cfg
    data['rec_in'] = proj
    data['rec_x0'] = rec_x0
    if iters is not None:
        data['rec_iters'] = iters
    elif alg == 'FBP' or alg == 'FBP_CUDA' or alg == 'FDK_CUDA':
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

    with open(os.path.join('inputs/rec2d', name), 'wb') as F:
      msgpack.pack(data, F, default=msgpack_numpy.encode)

def gen_2d(*, alg, projector, vol=128, volscale=None, projscale=1.0, ang=128, geom='par', opt=None, suffix=''):
    try:
        _ = vol[0]
        name = geom + '_' + projector + suffix + '.mp'
    except TypeError:
        name = geom + '_' + projector + '_' + str(vol) + suffix + '.mp'
        vol = (vol, vol)

    if volscale is not None:
        vol_geom = astra.create_vol_geom(vol[0], vol[1], -vol[1]*volscale*0.5, vol[1]*volscale*0.5, -vol[0]*volscale*0.5, vol[0]*volscale*0.5)
    else:
        vol_geom = astra.create_vol_geom(vol[0], vol[1])
    angles = np.linspace(0, 2*np.pi, ang, endpoint=False)
    if geom == 'par':
        proj_geom = astra.create_proj_geom('parallel', projscale, max(vol[0], vol[1]), angles)
    elif geom == 'parvec':
        proj_geom = astra.create_proj_geom('parallel', projscale, max(vol[0], vol[1]), angles)
        proj_geom = astra.geom_2vec(proj_geom)
    elif geom == 'fan':
        proj_geom = astra.create_proj_geom('fanflat', projscale, max(vol[0], vol[1]), angles, max(vol[0],vol[1])*4, 0)
    elif geom == 'fanvec':
        proj_geom = astra.create_proj_geom('fanflat', projscale, max(vol[0], vol[1]), angles, max(vol[0],vol[1])*4, 0)
        proj_geom = astra.geom_2vec(proj_geom)
    else:
        raise ValueError()

    proj_cfg = astra.astra_dict(projector)
    proj_cfg['ProjectionGeometry'] = proj_geom
    proj_cfg['VolumeGeometry'] = vol_geom
    if opt is not None:
        proj_cfg['option'] = opt
    else:
        proj_cfg['option'] = {}

    projector_id = astra.projector.create(proj_cfg)
    phantom_id = astra.data2d.shepp_logan(vol_geom, returnData=False)

    if alg == 'fp2d':
        phantom = astra.data2d.get(phantom_id)
        data = {}
        data['version'] = FORMAT_VERSION
        data['type'] = 'in_fp2d'
        data['fp_proj_cfg'] = proj_cfg
        data['fp_in'] = phantom

        print(name)
        #print(data)

        with open(os.path.join('inputs/fp2d', name), 'wb') as F:
            msgpack.pack(data, F, default=msgpack_numpy.encode)

    elif alg == 'bp2d':
        proj_id, proj = astra.create_sino(phantom_id, projector_id)

        astra.data2d.delete(proj_id)

        data = {}
        data['version'] = FORMAT_VERSION
        data['type'] = 'in_bp2d'
        data['bp_proj_cfg'] = proj_cfg
        data['bp_in'] = proj

        print(name)
        #print(data)

        with open(os.path.join('inputs/bp2d', name), 'wb') as F:
            msgpack.pack(data, F, default=msgpack_numpy.encode)
    else:
        raise RuntimeError()

    astra.projector.delete(projector_id)
    astra.data2d.delete(phantom_id)


if True:
    os.makedirs("inputs/rec2d", exist_ok=True)
    gen_cuda_rec2d(alg='FBP_CUDA')
    gen_cuda_rec2d(alg='FBP_CUDA', geom='fan')
    gen_cuda_rec2d(alg='FBP_CUDA', ang=10000, suffix='_10k_angles')
    gen_cuda_rec2d(alg='FBP_CUDA', geom='fan', ang=10000, suffix='_10k_angles')
    gen_cuda_rec2d(alg='SIRT_CUDA')
    gen_cuda_rec2d(alg='SIRT_CUDA', opt = { 'Relaxation': 2.0 }, suffix='_lam2')
    gen_cuda_rec2d(alg='SIRT_CUDA', vol=1024)
    gen_cuda_rec2d(alg='SIRT_CUDA', geom='fan')
    gen_cuda_rec2d(alg='CGLS_CUDA')
    gen_cuda_rec2d(alg='CGLS_CUDA', iters=[3,3,3,3], suffix='_reinit')
    gen_cuda_rec2d(alg='EM_CUDA')
    gen_cuda_rec2d(alg='SART_CUDA')
    gen_cuda_rec2d(alg='SART_CUDA', geom='fan')
    gen_cuda_rec2d(alg='SART_CUDA', opt = { 'Relaxation': 2.0 }, suffix='_lam2')

    gen_cuda_rec2d(alg='SIRT_CUDA', vol_mask='circle', proj_mask='limited-angle', suffix='_masks')
    gen_cuda_rec2d(alg='SIRT_CUDA', vol_mask='half', suffix='_vmask')
    gen_cuda_rec2d(alg='SIRT_CUDA', proj_mask='center', suffix='_pmask')

    gen_cuda_rec2d(alg='SART_CUDA', vol_mask='half', suffix='_vmask')
    gen_cuda_rec2d(alg='CGLS_CUDA', vol_mask='half', suffix='_vmask')

    gen_cuda_rec2d(alg='FBP', vol=64)
    gen_cuda_rec2d(alg='SIRT', vol=64)
    gen_cuda_rec2d(alg='CGLS', vol=64)
    gen_cuda_rec2d(alg='SART', vol=64)

if True:
    os.makedirs("inputs/bp2d", exist_ok=True)
    gen_2d(alg='bp2d', projector='cuda')
    gen_2d(alg='bp2d', projector='cuda', geom='fan')
    gen_2d(alg='bp2d', projector='cuda', ang=10000, suffix='_10k_angles')
    gen_2d(alg='bp2d', projector='cuda', geom='fan', ang=10000, suffix='_10k_angles')
    gen_2d(alg='bp2d', projector='cuda', geom='parvec')
    gen_2d(alg='bp2d', projector='cuda', geom='fanvec')
    gen_2d(alg='bp2d', projector='cuda', vol=2048)
    gen_2d(alg='bp2d', projector='line')
    gen_2d(alg='bp2d', projector='linear')
    gen_2d(alg='bp2d', projector='strip')
    gen_2d(alg='bp2d', projector='distance_driven')

if True:
    os.makedirs("inputs/fp2d", exist_ok=True)
    gen_2d(alg='fp2d', projector='cuda')
    gen_2d(alg='fp2d', projector='cuda', geom='fan')
    gen_2d(alg='fp2d', projector='cuda', ang=10000, suffix='_10k_angles')
    gen_2d(alg='fp2d', projector='cuda', geom='fan', ang=10000, suffix='_10k_angles')
    gen_2d(alg='fp2d', projector='cuda', geom='parvec')
    gen_2d(alg='fp2d', projector='cuda', geom='fanvec')
    gen_2d(alg='fp2d', projector='cuda', vol=2048)
    gen_2d(alg='fp2d', projector='line')
    gen_2d(alg='fp2d', projector='linear')
    gen_2d(alg='fp2d', projector='strip')
    gen_2d(alg='fp2d', projector='distance_driven')
