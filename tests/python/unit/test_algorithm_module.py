import astra
import pytest
import numpy as np


@pytest.fixture
def algorithm_config():
    proj_geom = astra.create_proj_geom('parallel3d', 1.0, 1.0, 20, 20, np.linspace(0, 1, 30))
    vol_geom = astra.create_vol_geom(10, 10, 10)
    proj_data_id = astra.data3d.create('-sino', proj_geom)
    vol_data_id = astra.data3d.create('-vol', vol_geom)
    config = astra.astra_dict('FP3D_CUDA')
    config['ProjectionDataId'] = proj_data_id
    config['VolumeDataId'] = vol_data_id
    yield config
    astra.data3d.delete(proj_data_id)
    astra.data3d.delete(vol_data_id)


def test_create_run(algorithm_config):
    algorithm_id = astra.algorithm.create(algorithm_config)
    astra.algorithm.run(algorithm_id)
    astra.algorithm.delete(algorithm_id)


def test_delete(algorithm_config):
    algorithm_id = astra.algorithm.create(algorithm_config)
    astra.algorithm.delete(algorithm_id)
    with pytest.raises(astra.log.AstraError):
        astra.algorithm.run(algorithm_id)


def test_clear(algorithm_config):
    algorithm_id1 = astra.algorithm.create(algorithm_config)
    algorithm_id2 = astra.algorithm.create(algorithm_config)
    astra.algorithm.clear()
    with pytest.raises(astra.log.AstraError):
        astra.algorithm.run(algorithm_id1)
    with pytest.raises(astra.log.AstraError):
        astra.algorithm.run(algorithm_id2)


def test_info(algorithm_config, capsys):
    get_n_info_objects = lambda: len(capsys.readouterr().out.split('\n')) - 5
    algorithm_id = astra.algorithm.create(algorithm_config)
    astra.algorithm.info()
    assert get_n_info_objects() == 1
    astra.algorithm.delete(algorithm_id)
    astra.algorithm.info()
    assert get_n_info_objects() == 0
