import astra
import numpy as np

DET_SPACING_X = 1.0
DET_SPACING_Y = 1.0
DET_ROW_COUNT = 20
DET_COL_COUNT = 45
N_ANGLES = 180
ANGLES = np.linspace(0, 2 * np.pi, 180, endpoint=False)
SOURCE_ORIGIN = 100
ORIGIN_DET = 100
N_ROWS = 40
N_COLS = 60
N_SLICES = 50


def get_data_dimensions(type, geom):
    data_id = astra.data3d.create(type, geom)
    data = astra.data3d.get(data_id)
    astra.data3d.delete(data_id)
    return data.shape


class TestVolumeGeometry:
    def test_arguments(self):
        geom = astra.create_vol_geom(N_ROWS, N_COLS, N_SLICES)
        assert get_data_dimensions('-vol', geom) == (N_SLICES, N_ROWS, N_COLS)
        geom = astra.create_vol_geom(N_ROWS, N_COLS, N_SLICES)
        assert get_data_dimensions('-vol', geom) == (N_SLICES, N_ROWS, N_COLS)

    def test_window(self):
        geom = astra.create_vol_geom(N_ROWS, N_COLS, N_SLICES, 0, 1, 2, 3, 4, 5)
        assert geom['option'] == {'WindowMinX': 0, 'WindowMaxX': 1,
                                  'WindowMinY': 2, 'WindowMaxY': 3,
                                  'WindowMinZ': 4, 'WindowMaxZ': 5}

    def test_default_window(self):
        geom = astra.create_vol_geom(N_ROWS, N_COLS, N_SLICES)
        assert geom['option'] == {'WindowMinX': -N_COLS/2, 'WindowMaxX': N_COLS/2,
                                  'WindowMinY': -N_ROWS/2, 'WindowMaxY': N_ROWS/2,
                                  'WindowMinZ': -N_SLICES/2, 'WindowMaxZ': N_SLICES/2}


class TestProjectionGeometries:
    def test_parallel3d(self):
        geom = astra.create_proj_geom('parallel3d', DET_SPACING_X, DET_SPACING_Y, DET_ROW_COUNT,
                                      DET_COL_COUNT, ANGLES)
        assert get_data_dimensions('-sino', geom) == (DET_ROW_COUNT, N_ANGLES, DET_COL_COUNT)

    def test_cone(self):
        geom = astra.create_proj_geom('cone', DET_SPACING_X, DET_SPACING_Y, DET_ROW_COUNT,
                                      DET_COL_COUNT, ANGLES, SOURCE_ORIGIN, ORIGIN_DET)
        assert get_data_dimensions('-sino', geom) == (DET_ROW_COUNT, N_ANGLES, DET_COL_COUNT)

    def test_parallel3d_vec(self):
        vectors = np.random.rand(N_ANGLES, 12)
        geom = astra.create_proj_geom('parallel3d_vec', DET_ROW_COUNT, DET_COL_COUNT, vectors)
        assert get_data_dimensions('-sino', geom) == (DET_ROW_COUNT, N_ANGLES, DET_COL_COUNT)

    def test_cone_vec(self):
        vectors = np.random.rand(N_ANGLES, 12)
        geom = astra.create_proj_geom('cone_vec', DET_ROW_COUNT, DET_COL_COUNT, vectors)
        assert get_data_dimensions('-sino', geom) == (DET_ROW_COUNT, N_ANGLES, DET_COL_COUNT)
