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


@pytest.fixture
def proj_data_non_contiguous(backend):
    data = np.full([N_ANGLES, DET_ROW_COUNT, DET_COL_COUNT], DATA_INIT_VALUE, dtype=np.float32)
    return _convert_to_backend(data, backend).swapaxes(0, 1)


@pytest.fixture
def vol_geom_singular_dim(singular_dims):
    dims = [N_ROWS, N_COLS, N_SLICES]
    for dim in singular_dims.split('-'):
        if dim == 'rows':
            dims[0] = 1
        elif dim == 'cols':
            dims[1] = 1
        elif dim == 'slices':
            dims[2] = 1
    return astra.create_vol_geom(*dims)


@pytest.fixture
def vol_data_singular_dim(singular_dims, backend):
    shape = [N_SLICES, N_ROWS, N_COLS]
    for dim in singular_dims.split('-'):
        if dim == 'rows':
            shape[1] = 1
        elif dim == 'cols':
            shape[2] = 1
        elif dim == 'slices':
            shape[0] = 1
    data = np.full(shape, DATA_INIT_VALUE, dtype=np.float32)
    return _convert_to_backend(data, backend)


@pytest.fixture
def vol_data_slice(singular_dims, backend):
    data = np.full([N_SLICES, N_ROWS, N_COLS], DATA_INIT_VALUE, dtype=np.float32)
    data = _convert_to_backend(data, backend)
    for dim in singular_dims.split('-'):
        if dim == 'rows':
            data = data[:, 0:1, :]
        elif dim == 'cols':
            data = data[:, :, 0:1]
        elif dim == 'slices':
            data = data[0:1, :, :]
    return data


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

    def test_non_contiguous(self, backend, proj_data_non_contiguous):
        if backend.startswith('jax'):
            # JAX should not produce non-contiguous tensors, so nothing to test
            return
        with pytest.raises(ValueError):
            astra.data3d.link('-sino', PROJ_GEOM, proj_data_non_contiguous)

    @pytest.mark.parametrize('singular_dims', [
        'rows', 'cols', 'slices', 'rows-cols', 'rows-slices', 'cols-slices', 'rows-cols-slices'
    ])
    def test_singular_dimensions(self, backend, singular_dims, vol_geom_singular_dim,
                                 vol_data_singular_dim):
        astra.data3d.link('-vol', vol_geom_singular_dim, vol_data_singular_dim)


@pytest.mark.parametrize('backend', ['pytorch_cuda', 'cupy'])
@pytest.mark.parametrize('singular_dims', ['rows'])
def test_allow_pitched(backend, singular_dims, vol_geom_singular_dim, vol_data_slice):
    astra.data3d.link('-vol', vol_geom_singular_dim, vol_data_slice)


@pytest.mark.parametrize('backend', ['numpy'])
def test_read_only(backend, vol_data):
    vol_data.flags['WRITEABLE'] = False
    # BufferError for numpy < 2 which doesn't support exporting read-only arrays
    # ValueError for numpy >= 2 where astra rejects the read-only array
    with pytest.raises((ValueError, BufferError)):
        astra.data3d.link('-vol', VOL_GEOM, vol_data)
