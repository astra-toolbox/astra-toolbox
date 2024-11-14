import astra
import numpy as np
import pytest

DET_SPACING = 1.0
DET_COUNT = 40
N_ANGLES = 180
ANGLES = np.linspace(0, 2 * np.pi, N_ANGLES, endpoint=False)
N_ROWS = 50
N_COLS = 60


@pytest.fixture
def geometry(geometry_type):
    if geometry_type == '-vol':
        return astra.create_vol_geom(N_ROWS, N_COLS)
    elif geometry_type == '-sino':
        return astra.create_proj_geom('parallel', DET_SPACING, DET_COUNT, ANGLES)


@pytest.fixture
def matrix_initializer(geometry_type):
    if geometry_type == '-vol':
        return np.random.rand(N_ROWS, N_COLS)
    elif geometry_type == '-sino':
        return np.random.rand(N_ANGLES, DET_COUNT)


@pytest.mark.parametrize('geometry_type', ['-vol', '-sino'])
class TestAll:
    def test_default_initializer(self, geometry_type, geometry):
        data_id = astra.data2d.create(geometry_type, geometry)
        data = astra.data2d.get(data_id)
        astra.data2d.delete(data_id)
        assert np.allclose(data, 0.0)

    def test_scalar_initializer(self, geometry_type, geometry):
        data_id = astra.data2d.create(geometry_type, geometry, 1.0)
        data = astra.data2d.get(data_id)
        astra.data2d.delete(data_id)
        assert np.allclose(data, 1.0)

    def test_matrix_initializer(self, geometry_type, geometry, matrix_initializer):
        data_id = astra.data2d.create(geometry_type, geometry, matrix_initializer)
        data = astra.data2d.get(data_id)
        astra.data2d.delete(data_id)
        assert np.allclose(data, matrix_initializer)

    @pytest.mark.parametrize('dtype', [np.float32, np.float64, np.int64, bool])
    def test_dtypes(self, geometry_type, geometry, matrix_initializer, dtype):
        matrix_initializer = matrix_initializer.astype(dtype)
        data_id = astra.data2d.create(geometry_type, geometry, matrix_initializer)
        data = astra.data2d.get(data_id)
        astra.data2d.delete(data_id)
        assert data.dtype == np.float32
        assert np.array_equal(data, matrix_initializer.astype(np.float32))

    def test_link(self, geometry_type, geometry, matrix_initializer):
        linked_array = matrix_initializer.astype(np.float32)
        data_id = astra.data2d.link(geometry_type, geometry, linked_array)
        # Assert writing to shared ndarray writes to Astra object
        linked_array[:] = 0.0
        astra_object_contents = astra.data2d.get(data_id)
        assert np.allclose(astra_object_contents, linked_array)
        # Assert writing to Astra object writes to shared ndarray
        astra.data2d.store(data_id, 2.0)
        assert np.allclose(linked_array, 2.0)
        # Assert Astra object contents is a copy not a link
        astra_object_contents[:] = 1.0
        assert not np.allclose(astra_object_contents, linked_array)
        astra.data2d.delete(data_id)

    def test_get_shared(self, geometry_type, geometry, matrix_initializer):
        data_id = astra.data2d.create(geometry_type, geometry, matrix_initializer)
        shared_array = astra.data2d.get_shared(data_id)
        # Assert writing to shared ndarray writes to Astra object
        shared_array[:] = 0.0
        astra_object_contents = astra.data2d.get(data_id)
        assert np.allclose(astra_object_contents, shared_array)
        # Assert writing to Astra object writes to shared ndarray
        astra.data2d.store(data_id, 2.0)
        assert np.allclose(shared_array, 2.0)
        # Assert Astra object contents is a copy not a link
        astra_object_contents[:] = 1.0
        assert not np.allclose(astra_object_contents, shared_array)
        astra.data2d.delete(data_id)

    def test_get_geometry(self, geometry_type, geometry):
        data_id = astra.data2d.create(geometry_type, geometry)
        geometry_in = geometry.copy()  # To safely use `pop` later
        geometry_out = astra.data2d.get_geometry(data_id)
        astra.data2d.delete(data_id)
        if geometry_type == '-sino':
            assert np.allclose(geometry_in.pop('ProjectionAngles'),
                               geometry_out.pop('ProjectionAngles'))
            assert geometry_in == geometry_out
        elif geometry_type == 'vol':
            # `option`, `options`, `Option` and `Options` are synonymous in astra configs
            assert geometry_in.pop('option') == geometry_out.pop('options')
            assert geometry_in == geometry_out

    def test_change_geometry(self, geometry_type, geometry):
        data_id = astra.data2d.create(geometry_type, geometry)
        if geometry_type == '-sino':
            new_spacing = 2 * DET_SPACING
            new_angles = np.random.rand(N_ANGLES)
            new_geometry = astra.create_proj_geom('parallel', new_spacing, DET_COUNT, new_angles)
            astra.data2d.change_geometry(data_id, new_geometry)
            changed_geometry = astra.data2d.get_geometry(data_id)
            astra.data2d.delete(data_id)
            assert changed_geometry['DetectorWidth'] == new_spacing
            assert np.allclose(changed_geometry['ProjectionAngles'], new_angles)
        elif geometry_type == '-vol':
            new_geometry = astra.create_vol_geom(N_ROWS, N_COLS, 0, 1, 2, 3)
            astra.data2d.change_geometry(data_id, new_geometry)
            changed_geometry = astra.data2d.get_geometry(data_id)
            astra.data2d.delete(data_id)
            assert changed_geometry['options'] == {'WindowMinX': 0, 'WindowMaxX': 1,
                                                   'WindowMinY': 2, 'WindowMaxY': 3}

    def test_delete(self, geometry_type, geometry):
        data_id = astra.data2d.create(geometry_type, geometry)
        astra.data2d.delete(data_id)
        with pytest.raises(astra.log.AstraError):
            astra.data2d.get(data_id)

    def test_clear(self, geometry_type, geometry):
        data_id1 = astra.data2d.create(geometry_type, geometry)
        data_id2 = astra.data2d.create(geometry_type, geometry)
        astra.data2d.clear()
        with pytest.raises(astra.log.AstraError):
            astra.data2d.get(data_id1)
        with pytest.raises(astra.log.AstraError):
            astra.data2d.get(data_id2)

    def test_info(self, geometry_type, geometry, capsys):
        get_n_info_objects = lambda: len(capsys.readouterr().out.split('\n')) - 5
        data_id = astra.data2d.create(geometry_type, geometry)
        astra.data2d.info()
        assert get_n_info_objects() == 1
        astra.data2d.delete(data_id)
        astra.data2d.info()
        assert get_n_info_objects() == 0


@pytest.mark.parametrize('modified', [True, False])
def test_shepp_logan(modified):
    geometry = astra.create_vol_geom(N_ROWS, N_COLS)
    data_id, data = astra.data2d.shepp_logan(geometry, modified)
    astra.data2d.delete(data_id)
    assert not np.allclose(data, 0.0)
