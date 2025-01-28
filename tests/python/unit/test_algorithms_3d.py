import astra
import numpy as np
import pytest

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
DATA_INIT_VALUE = 1.0


@pytest.fixture
def proj_geom(request):
    geometry_type = request.param
    if geometry_type == 'parallel3d':
        return astra.create_proj_geom('parallel3d', DET_SPACING_X, DET_SPACING_Y,
                                      DET_ROW_COUNT, DET_COL_COUNT, ANGLES)
    elif geometry_type == 'parallel3d_vec':
        geom = astra.create_proj_geom('parallel3d', DET_SPACING_X, DET_SPACING_Y,
                                      DET_ROW_COUNT, DET_COL_COUNT, ANGLES)
        return astra.geom_2vec(geom)
    elif geometry_type == 'cone':
        return astra.create_proj_geom('cone', DET_SPACING_X, DET_SPACING_Y,
                                      DET_ROW_COUNT, DET_COL_COUNT, ANGLES,
                                      SOURCE_ORIGIN, ORIGIN_DET)
    elif geometry_type == 'cone_vec':
        geom = astra.create_proj_geom('cone', DET_SPACING_X, DET_SPACING_Y,
                                      DET_ROW_COUNT, DET_COL_COUNT, ANGLES,
                                      SOURCE_ORIGIN, ORIGIN_DET)
        return astra.geom_2vec(geom)
    elif geometry_type == 'short_scan':
        cone_angle = np.arctan2(0.5 * DET_COL_COUNT * DET_SPACING_X, SOURCE_ORIGIN + ORIGIN_DET)
        angles = np.linspace(0, np.pi + 2 * cone_angle, 180)
        return astra.create_proj_geom('cone', DET_SPACING_X, DET_SPACING_Y,
                                      DET_ROW_COUNT, DET_COL_COUNT, angles,
                                      SOURCE_ORIGIN, ORIGIN_DET)


def _fourier_space_filter(proj_geom):
    # The full filter size should be the smallest power of two that is at least
    # twice the number of detector pixels
    full_filter_size = int(2 ** np.ceil(np.log2(2 * proj_geom['DetectorColCount'])))
    half_filter_size = full_filter_size // 2 + 1
    return np.linspace(0, 1, half_filter_size).reshape(1, -1)


def _real_space_filter(proj_geom):
    n = proj_geom['DetectorColCount']
    kernel = np.zeros([1, n])
    for i in range(n//4):
        f = np.pi * (2*i + 1)
        val = -2.0 / (f * f)
        kernel[0, n//2 + (2*i+1)] = val
        kernel[0, n//2 - (2*i+1)] = val
    kernel[0, n//2] = 0.5
    return kernel


@pytest.fixture
def custom_filter(proj_geom, request):
    filter_type = request.param
    if filter_type == 'projection':
        kernel = _fourier_space_filter(proj_geom)
    elif filter_type == 'sinogram':
        weights = np.random.rand(N_ANGLES)
        kernel = np.outer(weights, _fourier_space_filter(proj_geom))
    elif filter_type == 'rprojection':
        kernel = _real_space_filter(proj_geom)
    elif filter_type == 'rsinogram':
        weights = np.random.rand(N_ANGLES)
        kernel = np.outer(weights, _real_space_filter(proj_geom))
    dummy_geom = astra.create_proj_geom('parallel', 1, kernel.shape[1], np.zeros(kernel.shape[0]))
    filter_data_id = astra.data2d.create('-sino', dummy_geom, kernel)
    yield filter_type, filter_data_id
    astra.data2d.delete(filter_data_id)


@pytest.fixture
def sinogram_mask(proj_geom):
    mask = np.random.rand(DET_ROW_COUNT, N_ANGLES, DET_COL_COUNT) > 0.1
    mask_data_id = astra.data3d.create('-sino', proj_geom, mask)
    yield mask_data_id
    astra.data3d.delete(mask_data_id)


@pytest.fixture
def reconstruction_mask():
    mask = np.random.rand(N_SLICES, N_ROWS, N_COLS) > 0.1
    mask_data_id = astra.data3d.create('-vol', VOL_GEOM, mask)
    yield mask_data_id
    astra.data3d.delete(mask_data_id)


def make_algorithm_config(algorithm_type, proj_geom, options=None):
    algorithm_config = astra.astra_dict(algorithm_type)
    vol_data_id = astra.data3d.create('-vol', VOL_GEOM, DATA_INIT_VALUE)
    if algorithm_type.startswith('FP'):
        algorithm_config['VolumeDataId'] = vol_data_id
        proj_data_id = astra.data3d.create('-sino', proj_geom, DATA_INIT_VALUE)
    else:
        algorithm_config['ReconstructionDataId'] = vol_data_id
        # Make reconstruction contain negative and large numbers for testing
        # min/max constraint options
        proj_data = -10 * np.ones([DET_ROW_COUNT, N_ANGLES, DET_COL_COUNT])
        proj_data[DET_ROW_COUNT//4:-DET_ROW_COUNT//4:, DET_COL_COUNT//4:-DET_COL_COUNT//4] = 10
        proj_data_id = astra.data3d.create('-sino', proj_geom, proj_data)
    algorithm_config['ProjectionDataId'] = proj_data_id
    if options is not None:
        algorithm_config['option'] = options
    return algorithm_config


def get_algorithm_output(algorithm_config, n_iter=None):
    if n_iter is None:
        if algorithm_config['type'] in ['SIRT3D_CUDA', 'CGLS3D_CUDA']:
            n_iter = 2
        else:
            n_iter = 1
    algorithm_id = astra.algorithm.create(algorithm_config)
    astra.algorithm.run(algorithm_id, n_iter)
    if algorithm_config['type'].startswith('FP'):
        output = astra.data3d.get(algorithm_config['ProjectionDataId'])
        astra.data3d.delete(algorithm_config['VolumeDataId'])
    else:
        output = astra.data3d.get(algorithm_config['ReconstructionDataId'])
        astra.data3d.delete(algorithm_config['ReconstructionDataId'])
    astra.data3d.delete(algorithm_config['ProjectionDataId'])
    astra.algorithm.delete(algorithm_id)
    return output


@pytest.mark.parametrize(
    'proj_geom,', ['parallel3d', 'parallel3d_vec', 'cone', 'cone_vec'], indirect=True
)
@pytest.mark.parametrize(
    'algorithm_type', ['FP3D_CUDA', 'BP3D_CUDA', 'FDK_CUDA', 'SIRT3D_CUDA', 'CGLS3D_CUDA'],
)
def test_algorithms(proj_geom, algorithm_type):
    if algorithm_type == 'FDK_CUDA' and proj_geom['type'] not in ['cone', 'cone_vec']:
        pytest.xfail('Not implemented')
    algorithm_config = make_algorithm_config(algorithm_type, proj_geom)
    output = get_algorithm_output(algorithm_config)
    assert not np.allclose(output, DATA_INIT_VALUE)


class TestOptions:
    @pytest.mark.parametrize('proj_geom,', ['parallel3d', 'cone'], indirect=True)
    @pytest.mark.parametrize('algorithm_type', ['FP3D_CUDA', 'SIRT3D_CUDA', 'CGLS3D_CUDA'])
    def test_detector_supersampling_fp(self, proj_geom, algorithm_type):
        if algorithm_type == 'FP3D_CUDA':
            pytest.xfail('Known bug')
        algorithm_no_supersampling = make_algorithm_config(algorithm_type, proj_geom)
        algorithm_with_supersampling = make_algorithm_config(algorithm_type, proj_geom,
                                                             options={'DetectorSuperSampling': 3})
        output_no_supersampling = get_algorithm_output(algorithm_no_supersampling)
        output_with_supersampling = get_algorithm_output(algorithm_with_supersampling)
        assert not np.allclose(output_with_supersampling, DATA_INIT_VALUE)
        assert not np.allclose(output_with_supersampling, output_no_supersampling)

    @pytest.mark.parametrize('proj_geom,', ['parallel3d', 'cone'], indirect=True)
    @pytest.mark.parametrize(
        'algorithm_type', ['BP3D_CUDA', 'FDK_CUDA', 'SIRT3D_CUDA', 'CGLS3D_CUDA']
    )
    def test_voxel_supersampling(self, proj_geom, algorithm_type):
        if algorithm_type in ['BP3D_CUDA', 'FDK_CUDA']:
            pytest.xfail('Known bug')
        if algorithm_type == 'FDK_CUDA' and proj_geom['type'] == 'parallel3d':
            pytest.xfail('Not implemented')
        algorithm_no_supersampling = make_algorithm_config(algorithm_type, proj_geom)
        algorithm_with_supersampling = make_algorithm_config(algorithm_type, proj_geom,
                                                             options={'VoxelSuperSampling': 3})
        reconstruction_no_supersampling = get_algorithm_output(algorithm_no_supersampling)
        reconstruction_with_supersampling = get_algorithm_output(algorithm_with_supersampling)
        assert not np.allclose(reconstruction_with_supersampling, DATA_INIT_VALUE)
        assert not np.allclose(reconstruction_with_supersampling, reconstruction_no_supersampling)

    @pytest.mark.parametrize('proj_geom', ['cone'], indirect=True)
    @pytest.mark.parametrize('filter_type', ['ram-lak', 'none'])
    def test_fbp_filters_basic(self, proj_geom, filter_type):
        algorithm_config = make_algorithm_config(algorithm_type='FDK_CUDA', proj_geom=proj_geom,
                                                 options={'FilterType': filter_type})
        reconstruction = get_algorithm_output(algorithm_config)
        assert not np.allclose(reconstruction, DATA_INIT_VALUE)

    @pytest.mark.parametrize('proj_geom', ['cone'], indirect=True)
    @pytest.mark.parametrize('filter_type', ['tukey', 'gaussian', 'blackman', 'kaiser'])
    def test_fbp_filter_parameter(self, proj_geom, filter_type):
        algorithm_config = make_algorithm_config(
            algorithm_type='FDK_CUDA', proj_geom=proj_geom,
            options={'FilterType': filter_type, 'FilterParameter': -1.0}
        )
        reconstruction = get_algorithm_output(algorithm_config)
        assert not np.allclose(reconstruction, DATA_INIT_VALUE)

    @pytest.mark.parametrize('proj_geom', ['cone'], indirect=True)
    @pytest.mark.parametrize('filter_type', ['shepp-logan', 'cosine', 'hamming', 'hann'])
    def test_fbp_filter_d(self, proj_geom, filter_type):
        algorithm_config = make_algorithm_config(
            algorithm_type='FDK_CUDA', proj_geom=proj_geom,
            options={'FilterType': filter_type, 'FilterD': 1.0}
        )
        reconstruction = get_algorithm_output(algorithm_config)
        assert not np.allclose(reconstruction, DATA_INIT_VALUE)

    @pytest.mark.parametrize('proj_geom', ['cone'], indirect=True)
    @pytest.mark.parametrize(
        'custom_filter', ['projection', 'sinogram', 'rprojection', 'rsinogram'], indirect=True
    )
    def test_fbp_custom_filters(self, proj_geom, custom_filter):
        filter_type, filter_data_id = custom_filter
        algorithm_config = make_algorithm_config(
            algorithm_type='FDK_CUDA', proj_geom=proj_geom,
            options={'FilterType': filter_type, 'FilterSinogramId': filter_data_id}
        )
        reconstruction = get_algorithm_output(algorithm_config)
        assert not np.allclose(reconstruction, DATA_INIT_VALUE)

    @pytest.mark.parametrize('proj_geom', ['short_scan'], indirect=True)
    def test_short_scan(self, proj_geom):
        algorithm_no_short_scan = make_algorithm_config('FDK_CUDA', proj_geom)
        algorithm_with_short_scan = make_algorithm_config('FDK_CUDA', proj_geom,
                                                          options={'ShortScan': True})
        reconstruction_no_short_scan = get_algorithm_output(algorithm_no_short_scan)
        reconstruction_with_short_scan = get_algorithm_output(algorithm_with_short_scan)
        assert not np.allclose(reconstruction_with_short_scan, DATA_INIT_VALUE)
        assert not np.allclose(reconstruction_with_short_scan, reconstruction_no_short_scan)

    @pytest.mark.parametrize('proj_geom,', ['parallel3d'], indirect=True)
    def test_min_max_constraint(self, proj_geom):
        algorithm_no_constrains = make_algorithm_config('SIRT3D_CUDA', proj_geom)
        algorithm_with_constrains = make_algorithm_config(
            'SIRT3D_CUDA', proj_geom, options={'MinConstraint': 0.0, 'MaxConstraint': 0.125}
        )
        reconstruction_no_constrains = get_algorithm_output(algorithm_no_constrains)
        reconstruction_with_constrains = get_algorithm_output(algorithm_with_constrains)
        assert reconstruction_no_constrains.min() < 0.0
        assert reconstruction_no_constrains.max() > 0.125
        assert reconstruction_with_constrains.min() == 0.0
        assert reconstruction_with_constrains.max() == 0.125

    @pytest.mark.parametrize('proj_geom,', ['parallel3d'], indirect=True)
    @pytest.mark.parametrize('algorithm_type', ['SIRT3D_CUDA', 'CGLS3D_CUDA'])
    def test_reconstruction_mask(self, proj_geom,  reconstruction_mask, algorithm_type):
        algorithm_config = make_algorithm_config(
            algorithm_type, proj_geom, options={'ReconstructionMaskId': reconstruction_mask}
        )
        reconstruction = get_algorithm_output(algorithm_config)
        assert not np.allclose(reconstruction, DATA_INIT_VALUE)
        mask = (astra.data3d.get(reconstruction_mask) > 0)
        assert np.allclose(reconstruction[~mask], DATA_INIT_VALUE)

    @pytest.mark.parametrize('proj_geom,', ['parallel3d'], indirect=True)
    def test_sinogram_mask(self, proj_geom, sinogram_mask):
        algorithm_no_mask = make_algorithm_config('SIRT3D_CUDA', proj_geom)
        algorithm_with_sino_mask = make_algorithm_config('SIRT3D_CUDA', proj_geom,
                                                         options={'SinogramMaskId': sinogram_mask})
        reconstruction_no_mask = get_algorithm_output(algorithm_no_mask)
        reconstruction_with_sino_mask = get_algorithm_output(algorithm_with_sino_mask)
        assert not np.allclose(reconstruction_with_sino_mask, DATA_INIT_VALUE)
        assert not np.allclose(reconstruction_with_sino_mask, reconstruction_no_mask)

    @pytest.mark.parametrize('proj_geom,', ['parallel3d'], indirect=True)
    @pytest.mark.parametrize('algorithm_type', ['SIRT3D_CUDA', 'CGLS3D_CUDA'])
    def test_get_res_norm(self, proj_geom, algorithm_type):
        algorithm_config = make_algorithm_config(algorithm_type, proj_geom)
        algorithm_id = astra.algorithm.create(algorithm_config)
        astra.algorithm.run(algorithm_id, 2)
        res_norm = astra.algorithm.get_res_norm(algorithm_id)
        astra.algorithm.delete(algorithm_id)
        assert res_norm > 0.0
