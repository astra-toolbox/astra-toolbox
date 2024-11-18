import astra
import pytest


@pytest.fixture
def vol_data_id():
    vol_geom = astra.create_vol_geom(10, 10, 10)
    vol_data_id = astra.data3d.create('-vol', vol_geom)
    yield vol_data_id
    astra.data3d.delete(vol_data_id)


def test_credits(capsys):
    astra.astra.credits()
    assert len(capsys.readouterr().out) > 0


def test_delete(vol_data_id):
    astra.astra.delete(vol_data_id)
    with pytest.raises(astra.log.AstraError):
        astra.data3d.get(vol_data_id)


def test_info(vol_data_id, capsys):
    get_n_info_objects = lambda: len(capsys.readouterr().out.split('\n')) - 1
    astra.astra.info(vol_data_id)
    assert get_n_info_objects() == 1
    astra.astra.delete(vol_data_id)
    assert get_n_info_objects() == 0


def test_get_gpu_info():
    assert isinstance(astra.get_gpu_info(), str)


def test_has_feature():
    for feature in ['cuda', 'random_string']:
        assert isinstance(astra.astra.has_feature(feature), bool)


def test_use_cuda():
    assert isinstance(astra.astra.use_cuda(), bool)
