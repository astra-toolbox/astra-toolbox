import astra
import pytest
import numpy as np
import scipy

DET_SPACING_X = 1.0
DET_SPACING_Y = 1.0
DET_ROW_COUNT = 20
DET_COL_COUNT = 45
N_ANGLES = 180
ANGLES = np.linspace(0, 2 * np.pi, N_ANGLES, endpoint=False)
N_ROWS = 40
N_COLS = 30
N_SLICES = 50


@pytest.fixture
def op_tomo(dimensionality):
    if dimensionality == '2d':
        vol_geom = astra.create_vol_geom(N_ROWS, N_COLS)
        proj_geom = astra.create_proj_geom('parallel', DET_SPACING_Y, DET_COL_COUNT, ANGLES)
        projector_id = astra.create_projector('cuda', proj_geom, vol_geom)
    elif dimensionality == '3d':
        vol_geom = astra.create_vol_geom(N_ROWS, N_COLS, N_SLICES)
        proj_geom = astra.create_proj_geom('parallel3d', DET_SPACING_X, DET_SPACING_Y,
                                           DET_ROW_COUNT, DET_COL_COUNT, ANGLES)
        projector_id = astra.create_projector('cuda3d', proj_geom, vol_geom)
    yield astra.OpTomo(projector_id)
    astra.projector.delete(projector_id)


@pytest.fixture
def vol_data(dimensionality):
    if dimensionality == '2d':
        return np.ones([N_ROWS, N_COLS], dtype=np.float32)
    elif dimensionality == '3d':
        return np.ones([N_SLICES, N_ROWS, N_COLS], dtype=np.float32)


@pytest.fixture
def vol_buffer(dimensionality):
    if dimensionality == '2d':
        return np.zeros([N_ROWS, N_COLS], dtype=np.float32)
    elif dimensionality == '3d':
        return np.zeros([N_SLICES, N_ROWS, N_COLS], dtype=np.float32)


@pytest.fixture
def proj_data(dimensionality):
    if dimensionality == '2d':
        return np.ones([N_ANGLES, DET_COL_COUNT], dtype=np.float32)
    elif dimensionality == '3d':
        return np.ones([DET_ROW_COUNT, N_ANGLES, DET_COL_COUNT], dtype=np.float32)


@pytest.fixture
def proj_buffer(dimensionality):
    if dimensionality == '2d':
        return np.zeros([N_ANGLES, DET_COL_COUNT], dtype=np.float32)
    elif dimensionality == '3d':
        return np.zeros([DET_ROW_COUNT, N_ANGLES, DET_COL_COUNT], dtype=np.float32)


@pytest.fixture
def algorithm(dimensionality):
    if dimensionality == '2d':
         return 'SIRT_CUDA'
    elif dimensionality == '3d':
        return 'SIRT3D_CUDA'


@pytest.mark.parametrize('dimensionality', ['2d', '3d'])
class TestAll:
    def test_fp(self, dimensionality, op_tomo, vol_data):
        fp = op_tomo.FP(vol_data)
        assert not np.allclose(fp, 0.0)

    def test_fp_flattened(self, dimensionality, op_tomo, vol_data):
        fp = op_tomo.FP(vol_data.flatten())
        assert not np.allclose(fp, 0.0)

    def test_fp_out_arg(self, dimensionality, op_tomo, vol_data, proj_buffer):
        op_tomo.FP(vol_data, out=proj_buffer)
        assert not np.allclose(proj_buffer, 0.0)

    def test_bp(self, dimensionality, op_tomo, proj_data):
        bp = op_tomo.BP(proj_data)
        assert not np.allclose(bp, 0.0)

    def test_bp_flattened(self, dimensionality, op_tomo, proj_data):
        bp = op_tomo.BP(proj_data.flatten())
        assert not np.allclose(bp, 0.0)

    def test_bp_out_arg(self, dimensionality, op_tomo, proj_data, vol_buffer):
        op_tomo.BP(proj_data, out=vol_buffer)
        assert not np.allclose(vol_buffer, 0.0)

    def test_matvec(self, dimensionality, op_tomo, vol_data):
        fp = op_tomo(vol_data)
        assert not np.allclose(fp, 0.0)

    def test_rmatvec(self, dimensionality, op_tomo, proj_data):
        bp = op_tomo.T(proj_data)
        assert not np.allclose(bp, 0.0)

    def test_mul(self, dimensionality, op_tomo, vol_data):
        fp = op_tomo * vol_data
        assert not np.allclose(fp, 0.0)

    def test_reconstruct(self, dimensionality, op_tomo, proj_data, algorithm):
        rec = op_tomo.reconstruct(algorithm, proj_data, iterations=2,
                                  extraOptions={'MinConstraint': 0.0})
        assert not np.allclose(rec, 0.0)

    def test_scipy_solver(self, dimensionality, op_tomo, proj_data):
        result = scipy.sparse.linalg.lsqr(op_tomo, proj_data.flatten(), iter_lim=2)
        rec = result[0]
        assert not np.allclose(rec, 0.0)
