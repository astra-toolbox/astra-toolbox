import astra
import numpy as np
import pytest

DET_SPACING = 1.0
DET_COUNT = 40
N_ANGLES= 180
ANGLES = np.linspace(0, 2 * np.pi, N_ANGLES, endpoint=False)
PROJ_GEOM = astra.create_proj_geom('parallel', DET_SPACING, DET_COUNT, ANGLES)
N_ROWS = 50
N_COLS = 60
VOL_GEOM = astra.create_vol_geom(N_ROWS, N_COLS)


def test_get_volume_geometry():
    projector_id = astra.create_projector('linear', PROJ_GEOM, VOL_GEOM)
    geometry_in = VOL_GEOM.copy()  # To safely use `pop` later
    geometry_out = astra.projector.volume_geometry(projector_id)
    astra.projector.delete(projector_id)
    # `option`, `options`, `Option` and `Options` are synonymous in astra configs
    assert geometry_in.pop('option') == geometry_out.pop('options')
    assert geometry_in == geometry_out


def test_get_projection_geometry():
    projector_id = astra.create_projector('linear', PROJ_GEOM, VOL_GEOM)
    geometry_in = PROJ_GEOM.copy()  # To safely use `pop` later
    geometry_out = astra.projector.projection_geometry(projector_id)
    astra.projector.delete(projector_id)
    assert np.allclose(geometry_in.pop('ProjectionAngles'), geometry_out.pop('ProjectionAngles'))
    assert geometry_in == geometry_out


def test_is_cuda():
    projector_id = astra.create_projector('cuda', PROJ_GEOM, VOL_GEOM)
    assert astra.projector.is_cuda(projector_id)
    astra.projector.delete(projector_id)

    projector_id = astra.create_projector('linear', PROJ_GEOM, VOL_GEOM)
    assert not astra.projector.is_cuda(projector_id)
    astra.projector.delete(projector_id)


def test_delete():
    projector_id = astra.create_projector('linear', PROJ_GEOM, VOL_GEOM)
    astra.projector.delete(projector_id)
    with pytest.raises(astra.log.AstraError):
        astra.projector.is_cuda(projector_id)


def test_clear():
    projector_id1 = astra.create_projector('linear', PROJ_GEOM, VOL_GEOM)
    projector_id2 = astra.create_projector('linear', PROJ_GEOM, VOL_GEOM)
    astra.projector.clear()
    with pytest.raises(astra.log.AstraError):
        astra.projector.is_cuda(projector_id1)
    with pytest.raises(astra.log.AstraError):
        astra.projector.is_cuda(projector_id2)


def test_info(capsys):
    get_n_info_objects = lambda: len(capsys.readouterr().out.split('\n')) - 5
    projector_id = astra.create_projector('linear', PROJ_GEOM, VOL_GEOM)
    astra.projector.info()
    assert get_n_info_objects() == 1
    astra.projector.delete(projector_id)
    astra.projector.info()
    assert get_n_info_objects() == 0
