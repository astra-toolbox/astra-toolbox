import astra
import pytest
import scipy
import numpy as np
import scipy.sparse


@pytest.fixture
def scipy_matrix():
    data = np.random.rand(10, 10).astype(np.float32)
    return scipy.sparse.csr_array(data)


def test_create_get(scipy_matrix):
    matrix_id = astra.matrix.create(scipy_matrix)
    astra_matrix = astra.matrix.get(matrix_id)
    astra.matrix.delete(matrix_id)
    assert np.array_equal(astra_matrix.todense(), scipy_matrix.todense())


def test_get_size(scipy_matrix):
    matrix_id = astra.matrix.create(scipy_matrix)
    astra_matrix_size = astra.matrix.get_size(matrix_id)
    astra.matrix.delete(matrix_id)
    assert astra_matrix_size == scipy_matrix.shape


def test_store(scipy_matrix):
    matrix_id = astra.matrix.create(scipy_matrix)
    astra.matrix.store(matrix_id, -scipy_matrix)
    astra_matrix = astra.matrix.get(matrix_id)
    astra.matrix.delete(matrix_id)
    assert np.array_equal(astra_matrix.todense(), -scipy_matrix.todense())


def test_delete(scipy_matrix):
    matrix_id = astra.matrix.create(scipy_matrix)
    astra.matrix.delete(matrix_id)
    with pytest.raises(astra.log.AstraError):
        astra.matrix.get(matrix_id)


def test_delete(scipy_matrix):
    matrix_id1 = astra.matrix.create(scipy_matrix)
    matrix_id2 = astra.matrix.create(scipy_matrix)
    astra.matrix.clear()
    with pytest.raises(astra.log.AstraError):
        astra.matrix.get(matrix_id1)
    with pytest.raises(astra.log.AstraError):
        astra.matrix.get(matrix_id2)


def test_info(scipy_matrix, capsys):
    get_n_info_objects = lambda: len(capsys.readouterr().out.split('\n')) - 5
    matrix_id = astra.matrix.create(scipy_matrix)
    astra.matrix.info()
    assert get_n_info_objects() == 1
    astra.matrix.delete(matrix_id)
    astra.matrix.info()
    assert get_n_info_objects() == 0
