import astra
import astra.experimental
import numpy as np
import pytest

MODE_ADD = 0
MODE_SET = 1

DET_SPACING_X = 1.0
DET_SPACING_Y = 1.0
DET_ROW_COUNT = 20
DET_COL_COUNT = 45
N_ANGLES = 180
ANGLES = np.linspace(0, 2 * np.pi, N_ANGLES, endpoint=False)
SOURCE_ORIGIN = 100
ORIGIN_DET = 100
N_ROWS = 40
N_COLS = 30
N_SLICES = 50
VOL_SHIFT = 1, 2, 3
VOL_GEOM = astra.create_vol_geom(
    N_ROWS, N_COLS, N_SLICES,
    -N_COLS/2 + VOL_SHIFT[0], N_COLS/2 + VOL_SHIFT[0],
    -N_ROWS/2 + VOL_SHIFT[1], N_ROWS/2 + VOL_SHIFT[1],
    -N_SLICES/2 + VOL_SHIFT[2], N_SLICES/2 + VOL_SHIFT[2]
)


@pytest.fixture(scope='module')
def proj_geom(request):
    if request.param == 'parallel3d':
        return astra.create_proj_geom('parallel3d', DET_SPACING_X, DET_SPACING_Y,
                                      DET_ROW_COUNT, DET_COL_COUNT, ANGLES)
    elif request.param == 'parallel3d_vec':
        geom = astra.create_proj_geom('parallel3d', DET_SPACING_X, DET_SPACING_Y,
                                      DET_ROW_COUNT, DET_COL_COUNT, ANGLES)
        return astra.geom_2vec(geom)
    elif request.param == 'cone':
        return astra.create_proj_geom('cone', DET_SPACING_X, DET_SPACING_Y,
                                      DET_ROW_COUNT, DET_COL_COUNT, ANGLES,
                                      SOURCE_ORIGIN, ORIGIN_DET)
    elif request.param == 'cone_vec':
        geom = astra.create_proj_geom('cone', DET_SPACING_X, DET_SPACING_Y,
                                      DET_ROW_COUNT, DET_COL_COUNT, ANGLES,
                                      SOURCE_ORIGIN, ORIGIN_DET)
        return astra.geom_2vec(geom)
    elif request.param == 'short_scan':
        cone_angle = np.arctan2(0.5 * DET_COL_COUNT * DET_SPACING_X, SOURCE_ORIGIN + ORIGIN_DET)
        angles = np.linspace(0, np.pi + 2 * cone_angle, 180)
        return astra.create_proj_geom('cone', DET_SPACING_X, DET_SPACING_Y,
                                      DET_ROW_COUNT, DET_COL_COUNT, angles,
                                      SOURCE_ORIGIN, ORIGIN_DET)


@pytest.fixture
def projector(proj_geom):
    projector_id = astra.create_projector('cuda3d', proj_geom, VOL_GEOM)
    yield projector_id
    astra.projector.delete(projector_id)


@pytest.fixture
def vol_data():
    return np.ones([N_SLICES, N_ROWS, N_COLS], dtype=np.float32)


@pytest.fixture
def vol_buffer():
    return np.zeros([N_SLICES, N_ROWS, N_COLS], dtype=np.float32)


@pytest.fixture
def proj_data():
    return np.ones([DET_ROW_COUNT, N_ANGLES, DET_COL_COUNT], dtype=np.float32)


@pytest.fixture
def proj_buffer():
    return np.zeros([DET_ROW_COUNT, N_ANGLES, DET_COL_COUNT], dtype=np.float32)


@pytest.mark.parametrize(
    'proj_geom,', ['parallel3d', 'parallel3d_vec', 'cone', 'cone_vec'], indirect=True
)
class TestAll:
    def test_direct_FP3D(self, proj_geom, projector, vol_data, proj_buffer):
        astra.experimental.direct_FPBP3D(projector, vol_data, proj_buffer, MODE_SET, 'FP')
        proj_buffer_iter2 = proj_buffer.copy()
        astra.experimental.direct_FPBP3D(projector, vol_data, proj_buffer_iter2, MODE_ADD, 'FP')
        assert not np.allclose(proj_buffer, 0.0)
        assert not np.allclose(proj_buffer_iter2, 0.0)
        assert not np.allclose(proj_buffer, proj_buffer_iter2)

    def test_direct_BP3D(self, proj_geom, projector, proj_data, vol_buffer):
        astra.experimental.direct_FPBP3D(projector, vol_buffer, proj_data, MODE_SET, 'BP')
        vol_buffer_iter2 = vol_buffer.copy()
        astra.experimental.direct_FPBP3D(projector, vol_buffer_iter2, proj_data, MODE_ADD, 'BP')
        assert not np.allclose(vol_buffer, 0.0)
        assert not np.allclose(vol_buffer_iter2, 0.0)
        assert not np.allclose(vol_buffer, vol_buffer_iter2)

    def test_composite_FP3D(self, proj_geom, projector, vol_data, proj_buffer):
        vol_ids = [astra.data3d.create('-vol', VOL_GEOM, vol_data) for _ in range(2)]
        proj_ids = [astra.data3d.create('-sino', proj_geom, proj_buffer) for _ in range(2)]
        astra.experimental.do_composite(projector, vol_ids, proj_ids, MODE_SET, 'FP')
        proj_data_iter1 = [astra.data3d.get(x) for x in proj_ids]
        astra.experimental.do_composite(projector, vol_ids, proj_ids, MODE_ADD, 'FP')
        proj_data_iter2 = [astra.data3d.get(x) for x in proj_ids]
        assert all([not np.allclose(x, 0.0) for x in proj_data_iter1])
        assert all([not np.allclose(x, 0.0) for x in proj_data_iter2])
        assert all([not np.allclose(x, y) for x, y in zip(proj_data_iter1, proj_data_iter2)])

    def test_composite_BP3D(self, proj_geom, projector, proj_data, vol_buffer):
        vol_ids = [astra.data3d.create('-vol', VOL_GEOM, vol_buffer) for _ in range(2)]
        proj_ids = [astra.data3d.create('-sino', proj_geom, proj_data) for _ in range(2)]
        astra.experimental.do_composite(projector, vol_ids, proj_ids, MODE_SET, 'BP')
        vol_data_iter1 = [astra.data3d.get(x) for x in vol_ids]
        astra.experimental.do_composite(projector, vol_ids, proj_ids, MODE_ADD, 'BP')
        vol_data_iter2 = [astra.data3d.get(x) for x in vol_ids]
        assert all([not np.allclose(x, 0.0) for x in vol_data_iter1])
        assert all([not np.allclose(x, 0.0) for x in vol_data_iter2])
        assert all([not np.allclose(x, y) for x, y in zip(vol_data_iter1, vol_data_iter2)])

    def test_accumulate_FDK(self, proj_geom, projector, proj_data, vol_buffer):
        if proj_geom['type'].startswith('parallel'):
            pytest.xfail('Not implemented')
        proj_id = astra.data3d.link('-sino', proj_geom, proj_data)
        vol_id = astra.data3d.link('-vol', VOL_GEOM, vol_buffer)
        astra.experimental.accumulate_FDK(projector, vol_id, proj_id)
        vol_buffer_iter1 = vol_buffer.copy()
        astra.experimental.accumulate_FDK(projector, vol_id, proj_id)
        assert not np.allclose(vol_buffer, 0.0)
        assert not np.allclose(vol_buffer_iter1, 0.0)
        assert not np.allclose(vol_buffer, vol_buffer_iter1)

    def test_getProjectedBBox(self, proj_geom):
        minv, maxv = astra.experimental.getProjectedBBox(proj_geom, minx=0, maxx=1, miny=2, maxy=3,
                                                         minz=4, maxz=5)

    def test_projectPoint(self, proj_geom):
        u, v = astra.experimental.projectPoint(proj_geom, x=0, y=1, z=2, angle=3)
