import astra
import numpy as np
import scipy

DET_SPACING = 1.0
DET_COUNT = 40
N_ANGLES = 180
ANGLES = np.linspace(0, 2 * np.pi, N_ANGLES, endpoint=False)
SOURCE_ORIGIN = 100
ORIGIN_DET = 100
N_ROWS = 50
N_COLS = 60


def get_data_dimensions(type, geom):
    data_id = astra.data2d.create(type, geom)
    data = astra.data2d.get(data_id)
    astra.data2d.delete(data_id)
    return data.shape


class TestVolumeGeometry:
    def test_arguments(self):
        geom = astra.create_vol_geom(N_ROWS)
        assert get_data_dimensions('-vol', geom) == (N_ROWS, N_ROWS)
        geom = astra.create_vol_geom([N_ROWS, N_COLS])
        assert get_data_dimensions('-vol', geom) == (N_ROWS, N_COLS)
        geom = astra.create_vol_geom(N_ROWS, N_COLS)
        assert get_data_dimensions('-vol', geom) == (N_ROWS, N_COLS)

    def test_window(self):
        geom = astra.create_vol_geom(N_ROWS, N_COLS, 0, 1, 2, 3)
        assert geom['option'] == {'WindowMinX': 0, 'WindowMaxX': 1,
                                  'WindowMinY': 2, 'WindowMaxY': 3}

    def test_default_window(self):
        geom = astra.create_vol_geom(N_ROWS, N_COLS)
        assert geom['option'] == {'WindowMinX': -N_COLS/2, 'WindowMaxX': N_COLS/2,
                                  'WindowMinY': -N_ROWS/2, 'WindowMaxY': N_ROWS/2}


class TestProjectionGeometries:
    def test_parallel(self):
        geom = astra.create_proj_geom('parallel', DET_SPACING, DET_COUNT, ANGLES)
        assert get_data_dimensions('-sino', geom) == (N_ANGLES, DET_COUNT)

    def test_fanflat(self):
        geom = astra.create_proj_geom('fanflat', DET_SPACING, DET_COUNT, ANGLES,
                                      SOURCE_ORIGIN, ORIGIN_DET)
        assert get_data_dimensions('-sino', geom) == (N_ANGLES, DET_COUNT)

    def test_parallel_vec(self):
        vectors = np.random.rand(N_ANGLES, 6)
        geom = astra.create_proj_geom('parallel_vec', DET_COUNT, vectors)
        assert get_data_dimensions('-sino', geom) == (N_ANGLES, DET_COUNT)

    def test_fanflat_vec(self):
        vectors = np.random.rand(N_ANGLES, 6)
        geom = astra.create_proj_geom('fanflat_vec', DET_COUNT, vectors)
        assert get_data_dimensions('-sino', geom) == (N_ANGLES, DET_COUNT)

    def test_sparse_matrix(self):
        matrix = scipy.sparse.csc_array(np.zeros([DET_COUNT * N_ANGLES, N_ROWS * N_COLS]))
        matrix_id = astra.matrix.create(matrix)
        geom = astra.create_proj_geom('sparse_matrix', DET_SPACING, DET_COUNT, ANGLES, matrix_id)
        assert get_data_dimensions('-sino', geom) == (N_ANGLES, DET_COUNT)
        astra.matrix.delete(matrix_id)
