import astra
import astra.experimental
import numpy as np
import pytest

MODE_ADD = 0
MODE_SET = 1

DET_SPACING = 1.0
DET_COUNT = 45
N_ANGLES = 180
ANGLES = np.linspace(0, 2 * np.pi, N_ANGLES, endpoint=False)
SOURCE_ORIGIN = 100
ORIGIN_DET = 100
N_ROWS = 40
N_COLS = 30
N_SLICES = 50
VOL_SHIFT = 1, 2
VOL_GEOM = astra.create_vol_geom(
    N_ROWS, N_COLS,
    -N_COLS/2 + VOL_SHIFT[0], N_COLS/2 + VOL_SHIFT[0],
    -N_ROWS/2 + VOL_SHIFT[1], N_ROWS/2 + VOL_SHIFT[1])


@pytest.fixture(scope='module')
def proj_geom(request):
    if request.param == 'parallel':
        return astra.create_proj_geom('parallel', DET_SPACING, DET_COUNT, ANGLES)
    elif request.param == 'parallel_vec':
        geom = astra.create_proj_geom('parallel', DET_SPACING, DET_COUNT, ANGLES)
        return astra.geom_2vec(geom)
    elif request.param == 'fanflat':
        return astra.create_proj_geom('fanflat', DET_SPACING,
                                      DET_COUNT, ANGLES,
                                      SOURCE_ORIGIN, ORIGIN_DET)
    elif request.param == 'fanflat_vec':
        geom = astra.create_proj_geom('fanflat', DET_SPACING,
                                      DET_COUNT, ANGLES,
                                      SOURCE_ORIGIN, ORIGIN_DET)
        return astra.geom_2vec(geom)
    elif request.param == 'short_scan':
        cone_angle = np.arctan2(0.5 * DET_COUNT * DET_SPACING, SOURCE_ORIGIN + ORIGIN_DET)
        angles = np.linspace(0, np.pi + 2 * cone_angle, 180)
        return astra.create_proj_geom('fanflat', DET_SPACING, DET_COUNT, angles,
                                      SOURCE_ORIGIN, ORIGIN_DET)


@pytest.fixture
def projector(proj_geom, proj_type):
    if proj_type == 'gpu':
        projector_id = astra.create_projector('cuda', proj_geom, VOL_GEOM)
    elif 'fanflat' in proj_geom['type']:
        projector_id = astra.create_projector('line_fanflat', proj_geom, VOL_GEOM)
    else:
        projector_id = astra.create_projector('linear', proj_geom, VOL_GEOM)

    yield projector_id
    astra.projector.delete(projector_id)


@pytest.fixture
def vol_data():
    return np.ones([N_ROWS, N_COLS], dtype=np.float32)


@pytest.fixture
def vol_buffer():
    return np.zeros([N_ROWS, N_COLS], dtype=np.float32)


@pytest.fixture
def proj_data():
    return np.ones([N_ANGLES, DET_COUNT], dtype=np.float32)


@pytest.fixture
def proj_buffer():
    return np.zeros([N_ANGLES, DET_COUNT], dtype=np.float32)


@pytest.mark.parametrize(
    'proj_geom', ['parallel', 'parallel_vec', 'fanflat', 'fanflat_vec'], indirect=True
)
@pytest.mark.parametrize('proj_type,', ['cpu', 'gpu'])
class TestAll:
    def test_direct_FP2D(self, proj_geom, projector, vol_data, proj_buffer):
        astra.experimental.direct_FP2D(projector, vol_data, proj_buffer)
        assert not np.allclose(proj_buffer, 0.0)
        proj_data_id, proj_data_ref = astra.create_sino(vol_data, projector)
        astra.data2d.delete(proj_data_id)
        assert np.allclose(proj_buffer, proj_data_ref)

    def test_direct_BP2D(self, proj_geom, projector, proj_data, vol_buffer):
        astra.experimental.direct_BP2D(projector, vol_buffer, proj_data)
        assert not np.allclose(vol_buffer, 0.0)
        vol_data_id, vol_data_ref = astra.create_backprojection(proj_data, projector)
        astra.data2d.delete(vol_data_id)
        assert np.allclose(vol_buffer, vol_data_ref)

    def test_direct_FP(self, proj_geom, projector, vol_data, proj_buffer):
        astra.projector.direct_FP(projector, vol_data, out=proj_buffer)
        assert not np.allclose(proj_buffer, 0.0)
        proj_data_id, proj_data_ref = astra.create_sino(vol_data, projector)
        astra.data2d.delete(proj_data_id)
        assert np.allclose(proj_buffer, proj_data_ref)

    def test_direct_BP(self, proj_geom, projector, proj_data, vol_buffer):
        astra.projector.direct_BP(projector, proj_data, out=vol_buffer)
        assert not np.allclose(vol_buffer, 0.0)
        vol_data_id, vol_data_ref = astra.create_backprojection(proj_data, projector)
        astra.data2d.delete(vol_data_id)
        assert np.allclose(vol_buffer, vol_data_ref)
