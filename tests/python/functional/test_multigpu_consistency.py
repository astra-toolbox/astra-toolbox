import astra
import numpy as np
import pytest

@pytest.mark.slow
def test_multigpu_consistency():

    def print_diff(a, b):
        x = np.abs((a-b).reshape(-1))
        print(np.linalg.norm(x, ord=2), np.max(x), np.max(np.abs(a)), np.max(np.abs(b)))

    N = 1024

    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    vg = astra.create_vol_geom(N, N, N)
    pg = astra.create_proj_geom('cone', 1.0, 1.0, N, N, angles, 10*N, 0)

    projector_id = astra.create_projector('cuda3d', pg, vg)
    W = astra.OpTomo(projector_id)

    phantom_id = astra.data3d.shepp_logan(vg, returnData=False)

    astra.set_gpu_index(0)

    pid, projdata_single = astra.create_sino3d_gpu(phantom_id, pg, vg, returnData=True)
    astra.data3d.delete(pid)

    rec_single = W.reconstruct('FDK_CUDA', projdata_single)

    gpu = 0
    gpus = [0]

    while True:
        print("Now using GPUs " + ", ".join(str(i) for i in gpus))

        for m in ( 0, 100000000 ):
            astra.set_gpu_index(gpus, memory=m)

            pid, projdata_multi = astra.create_sino3d_gpu(phantom_id, pg, vg, returnData=True)
            astra.data3d.delete(pid)

            print_diff(projdata_single, projdata_multi)
            assert(np.allclose(projdata_multi, projdata_single, rtol=1e-3, atol=1e-1))

            rec_multi = W.reconstruct('FDK_CUDA', projdata_single)

            print_diff(rec_single, rec_multi)
            assert(np.allclose(rec_multi, rec_single, rtol=1e-3, atol=1e-3))

        gpu = gpu + 1
        if 'Invalid' in astra.get_gpu_info(gpu):
            print(f"No GPU #%s. Aborting." % (gpu,))
            break

        gpus.append(gpu)

