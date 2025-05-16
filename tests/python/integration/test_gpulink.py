import astra
import astra.experimental
import numpy as np
import pytest

DET_SPACING_X = 1.0
DET_SPACING_Y = 1.0
DET_ROW_COUNT = 20
DET_COL_COUNT = 45
N_ANGLES = 180
ANGLES = np.linspace(0, 2 * np.pi, N_ANGLES, endpoint=False)
N_ROWS = 40
N_COLS = 30
N_SLICES = 50
VOL_GEOM = astra.create_vol_geom(N_ROWS, N_COLS, N_SLICES)
PROJ_GEOM = astra.create_proj_geom('parallel3d', DET_SPACING_X, DET_SPACING_Y,
                                   DET_ROW_COUNT, DET_COL_COUNT, ANGLES)
DATA_INIT_VALUE = 1.0


def _convert_to_backend(data, backend):
    if backend == 'pytorch_cuda':
        import torch
        return torch.tensor(data, device='cuda')
    elif backend == 'cupy':
        import cupy as cp
        return cp.array(data)


def _make_GPULink(data, backend):
    if backend == 'pytorch_cuda':
        ptr = data.data_ptr()
    elif backend == 'cupy':
        ptr = data.data.ptr
    return astra.data3d.GPULink(ptr, data.shape[2], data.shape[1], data.shape[0], 4*data.shape[2])


@pytest.fixture
def projector():
    projector_id = astra.create_projector('cuda3d', PROJ_GEOM, VOL_GEOM)
    yield projector_id
    astra.projector3d.delete(projector_id)


@pytest.fixture
def vol_data(backend):
    data = np.full([N_SLICES, N_ROWS, N_COLS], DATA_INIT_VALUE, dtype=np.float32)
    return _convert_to_backend(data, backend)


@pytest.fixture
def proj_data(backend):
    data = np.full([DET_ROW_COUNT, N_ANGLES, DET_COL_COUNT], DATA_INIT_VALUE, dtype=np.float32)
    return _convert_to_backend(data, backend)


@pytest.fixture
def reference_fp():
    vol_data_id = astra.data3d.create('-vol', VOL_GEOM, DATA_INIT_VALUE)
    data_id, data = astra.create_sino3d_gpu(vol_data_id, PROJ_GEOM, VOL_GEOM)
    astra.data3d.delete(data_id)
    return data


@pytest.fixture
def reference_bp():
    proj_data_id = astra.data3d.create('-sino', PROJ_GEOM, DATA_INIT_VALUE)
    data_id, data = astra.create_backprojection3d_gpu(proj_data_id, PROJ_GEOM, VOL_GEOM)
    astra.data3d.delete(data_id)
    return data


@pytest.mark.parametrize('backend', ['pytorch_cuda', 'cupy'])
class TestAll:
    def test_backends_fp(self, backend, projector, vol_data, proj_data, reference_fp):
        vol_data_link = _make_GPULink(vol_data, backend)
        proj_data_link = _make_GPULink(proj_data, backend)
        astra.experimental.direct_FP3D(projector, vol_data_link, proj_data_link)
        if backend == 'pytorch_cuda':
            proj_data = proj_data.cpu()
        assert np.allclose(proj_data, reference_fp)

    def test_backends_bp(self, backend, projector, vol_data, proj_data, reference_bp):
        vol_data_link = _make_GPULink(vol_data, backend)
        proj_data_link = _make_GPULink(proj_data, backend)
        astra.experimental.direct_BP3D(projector, vol_data_link, proj_data_link)
        if backend == 'pytorch_cuda':
            vol_data = vol_data.cpu()
        assert np.allclose(vol_data, reference_bp)
