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
DATA_INIT_VALUE = 1.0


def _convert_to_backend(data, backend):
    if backend == 'numpy':
        return data
    elif backend == 'pytorch_cpu':
        import torch
        return torch.tensor(data, device='cpu')
    elif backend == 'pytorch_cuda':
        import torch
        return torch.tensor(data, device='cuda')
    elif backend == 'cupy':
        import cupy as cp
        return cp.array(data)
    elif backend == 'jax_cpu':
        import jax
        return jax.device_put(data, device=jax.devices('cpu')[0])
    elif backend == 'jax_cuda':
        import jax
        return jax.device_put(data, device=jax.devices('cuda')[0])


@pytest.fixture(params=['full', 'singleton_rows', 'singleton_cols', 'singleton_slices',
                        'singleton_rows_cols', 'singleton_rows_slices', 'singleton_cols_slices',
                        'singleton_rows_cols_slices'])
def vol_geom(request):
    if request.param == 'full':
        return astra.create_vol_geom(N_ROWS, N_COLS, N_SLICES)
    elif request.param.startswith('singleton'):
        dims = [N_SLICES, N_ROWS, N_COLS]
        for dim in request.param.split('_')[1:]:
            if dim == 'rows':
                dims[1] = 1
            elif dim == 'cols':
                dims[2] = 1
            elif dim == 'slices':
                dims[0] = 1
        return astra.create_vol_geom(*dims)


@pytest.fixture(params=['full', 'singleton_rows', 'singleton_angles', 'singleton_cols',
                        'singleton_rows_angles', 'singleton_rows_cols', 'singleton_angles_cols',
                        'singleton_rows_angles_cols'])
def proj_geom(request):
    if request.param == 'full':
        return astra.create_proj_geom('parallel3d', DET_SPACING_X, DET_SPACING_Y,
                                      DET_ROW_COUNT, DET_COL_COUNT, ANGLES)
    elif request.param.startswith('singleton'):
        rows, n_angles, cols = DET_ROW_COUNT, N_ANGLES, DET_COL_COUNT
        for dim in request.param.split('_')[1:]:
            if dim == 'rows':
                rows = 1
            elif dim == 'angles':
                n_angles = 1
            elif dim == 'cols':
                cols = 1
        return astra.create_proj_geom('parallel3d', DET_SPACING_X, DET_SPACING_Y,
                                      rows, cols, ANGLES[:n_angles])


@pytest.fixture
def vol_data(backend, vol_geom):
    shape = astra.geom_size(vol_geom)
    data = np.full(shape, DATA_INIT_VALUE, dtype=np.float32)
    return _convert_to_backend(data, backend)


@pytest.fixture
def proj_data(backend, proj_geom):
    shape = astra.geom_size(proj_geom)
    data = np.full(shape, DATA_INIT_VALUE, dtype=np.float32)
    return _convert_to_backend(data, backend)


@pytest.fixture
def projector(vol_geom, proj_geom):
    projector_id = astra.create_projector('cuda3d', proj_geom, vol_geom)
    yield projector_id
    astra.projector3d.delete(projector_id)


@pytest.fixture
def reference_fp(vol_geom, proj_geom):
    vol_data_id = astra.data3d.create('-vol', vol_geom, DATA_INIT_VALUE)
    data_id, data = astra.create_sino3d_gpu(vol_data_id, proj_geom, vol_geom)
    astra.data3d.delete(data_id)
    return data


@pytest.fixture
def reference_bp(vol_geom, proj_geom):
    proj_data_id = astra.data3d.create('-sino', proj_geom, DATA_INIT_VALUE)
    data_id, data = astra.create_backprojection3d_gpu(proj_data_id, proj_geom, vol_geom)
    astra.data3d.delete(data_id)
    return data


@pytest.fixture
def proj_data_non_contiguous(backend, proj_geom):
    shape = astra.geom_size(proj_geom)
    data = np.full(shape, DATA_INIT_VALUE, dtype=np.float32)
    return _convert_to_backend(data, backend).swapaxes(0, 1)


@pytest.mark.parametrize('backend', ['numpy', 'pytorch_cpu', 'pytorch_cuda', 'cupy',
                                     'jax_cpu', 'jax_cuda'])
class TestAll:
    def test_backends_fp(self, backend, projector, vol_data, proj_data, reference_fp):
        astra.experimental.direct_FP3D(projector, vol_data, proj_data)
        if backend.startswith('pytorch'):
            proj_data = proj_data.cpu()
        assert np.allclose(proj_data, reference_fp)

    def test_backends_bp(self, backend, projector, vol_data, proj_data, reference_bp):
        astra.experimental.direct_BP3D(projector, vol_data, proj_data)
        if backend.startswith('pytorch'):
            vol_data = vol_data.cpu()
        assert np.allclose(vol_data, reference_bp)
    
    @pytest.mark.parametrize('proj_geom', ['full'], indirect=True)
    def test_non_contiguous(self, backend, proj_geom, proj_data_non_contiguous):
        if backend.startswith('jax'):
            # JAX should not produce non-contiguous tensors, so nothing to test
            return
        with pytest.raises(ValueError):
            astra.data3d.link('-sino', proj_geom, proj_data_non_contiguous)        


@pytest.mark.parametrize('backend', ['numpy'])
@pytest.mark.parametrize('vol_geom', ['full'], indirect=True)
def test_read_only(backend, vol_geom, vol_data):
    vol_data.flags['WRITEABLE'] = False
    # BufferError for numpy < 2 which doesn't support exporting read-only arrays
    # ValueError for numpy >= 2 where astra rejects the read-only array
    with pytest.raises((ValueError, BufferError)):
        astra.data3d.link('-vol', vol_geom, vol_data)
