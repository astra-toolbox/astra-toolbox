import astra
import numpy as np
import pytest

DET_SPACING = 1.0
DET_COUNT = 40
N_ANGLES= 180
ANGLES = np.linspace(0, 2 * np.pi, N_ANGLES, endpoint=False)
SOURCE_ORIGIN = 100
ORIGIN_DET = 100
N_ROWS = 50
N_COLS = 60
VOL_SHIFT = 0, 0
VOL_GEOM = astra.create_vol_geom(
    N_ROWS, N_COLS,
    -N_COLS/2 + VOL_SHIFT[0], N_COLS/2 + VOL_SHIFT[0],
    -N_ROWS/2 + VOL_SHIFT[1], N_ROWS/2 + VOL_SHIFT[1]
)
DATA_INIT_VALUE = 1.0


@pytest.fixture
def proj_geom(request):
    geometry_type = request.param
    if geometry_type == 'parallel':
        yield astra.create_proj_geom('parallel', DET_SPACING, DET_COUNT, ANGLES)
    elif geometry_type == 'parallel_vec':
        geom = astra.create_proj_geom('parallel', DET_SPACING, DET_COUNT, ANGLES)
        yield astra.geom_2vec(geom)
    elif geometry_type == 'fanflat':
        yield astra.create_proj_geom('fanflat', DET_SPACING, DET_COUNT, ANGLES,
                                      SOURCE_ORIGIN, ORIGIN_DET)
    elif geometry_type == 'fanflat_vec':
        geom = astra.create_proj_geom('fanflat', DET_SPACING, DET_COUNT, ANGLES,
                                      SOURCE_ORIGIN, ORIGIN_DET)
        yield astra.geom_2vec(geom)
    elif geometry_type == 'sparse_matrix':
        dummy_proj_geom = astra.create_proj_geom('parallel', DET_SPACING, DET_COUNT, ANGLES)
        dummy_proj_id = astra.create_projector('linear', dummy_proj_geom, VOL_GEOM)
        matrix_id = astra.projector.matrix(dummy_proj_id)
        yield astra.create_proj_geom('sparse_matrix', DET_SPACING, DET_COUNT, ANGLES, matrix_id)
        astra.matrix.delete(matrix_id)
    elif geometry_type == 'short_scan':
        cone_angle = np.arctan2(0.5 * DET_COUNT * DET_SPACING, SOURCE_ORIGIN + ORIGIN_DET)
        angles = np.linspace(0, np.pi + 2 * cone_angle, 180)
        yield astra.create_proj_geom('fanflat', DET_SPACING, DET_COUNT, angles,
                                      SOURCE_ORIGIN, ORIGIN_DET)


@pytest.fixture
def projector(proj_geom, request):
    projector_type = request.param
    projector_id = astra.create_projector(projector_type, proj_geom, VOL_GEOM)
    yield projector_id
    astra.projector.delete(projector_id)


def _fourier_space_filter(proj_geom):
    # The full filter size should be the smallest power of two that is at least
    # twice the number of detector pixels
    full_filter_size = int(2 ** np.ceil(np.log2(2 * proj_geom['DetectorCount'])))
    half_filter_size = full_filter_size // 2 + 1
    return np.linspace(0, 1, half_filter_size).reshape(1, -1)


def _real_space_filter(proj_geom):
    n = proj_geom['DetectorCount']
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
    mask = np.random.rand(N_ANGLES, DET_COUNT) > 0.1
    mask_data_id = astra.data2d.create('-sino', proj_geom, mask)
    yield mask_data_id
    astra.data2d.delete(mask_data_id)


@pytest.fixture
def reconstruction_mask():
    mask = np.random.rand(N_ROWS, N_COLS) > 0.1
    mask_data_id = astra.data2d.create('-vol', VOL_GEOM, mask)
    yield mask_data_id
    astra.data2d.delete(mask_data_id)


def make_algorithm_config(algorithm_type, proj_geom, projector=None, options=None):
    algorithm_config = astra.astra_dict(algorithm_type)
    vol_data_id = astra.data2d.create('-vol', VOL_GEOM, DATA_INIT_VALUE)
    if algorithm_type.startswith('FP'):
        algorithm_config['VolumeDataId'] = vol_data_id
        proj_data_id = astra.data2d.create('-sino', proj_geom, DATA_INIT_VALUE)
    else:
        algorithm_config['ReconstructionDataId'] = vol_data_id
        # Make reconstruction contain negative and large numbers for testing
        # min/max constraint options
        proj_data = -10 * np.ones([N_ANGLES, DET_COUNT])
        proj_data[:, DET_COUNT//4:-DET_COUNT//4] = 10
        proj_data_id = astra.data2d.create('-sino', proj_geom, proj_data)
    algorithm_config['ProjectionDataId'] = proj_data_id
    if projector is not None:
        algorithm_config['ProjectorId'] = projector
    if options is not None:
        algorithm_config['option'] = options
    return algorithm_config


def get_algorithm_output(algorithm_config, n_iter=None):
    algorithm_id = astra.algorithm.create(algorithm_config)
    if n_iter is None:
        if algorithm_config['type'].startswith(('SIRT', 'CGLS', 'EM')):
            n_iter = 2
        elif algorithm_config['type'].startswith('SART'):
            n_iter = N_ANGLES
        elif algorithm_config['type'].startswith('ART'):
            n_iter = N_ANGLES * DET_COUNT
        else:
            n_iter = 1
    astra.algorithm.run(algorithm_id, n_iter)
    if algorithm_config['type'].startswith('FP'):
        output = astra.data2d.get(algorithm_config['ProjectionDataId'])
        astra.data2d.delete(algorithm_config['VolumeDataId'])
    else:
        output = astra.data2d.get(algorithm_config['ReconstructionDataId'])
        astra.data2d.delete(algorithm_config['ReconstructionDataId'])
    astra.data2d.delete(algorithm_config['ProjectionDataId'])
    astra.algorithm.delete(algorithm_id)
    return output


@pytest.mark.parametrize('proj_geom, projector', [
    ('parallel', 'line'),
    ('parallel', 'strip'),
    ('parallel', 'linear'),
    ('parallel_vec', 'line'),
    ('parallel_vec', 'strip'),
    ('parallel_vec', 'linear'),
    ('fanflat', 'line_fanflat'),
    ('fanflat', 'strip_fanflat'),
    ('fanflat_vec', 'line_fanflat'),
    ('sparse_matrix', 'sparse_matrix')
], indirect=True)
@pytest.mark.parametrize('algorithm_type', ['FP', 'BP', 'FBP', 'SIRT', 'SART', 'ART', 'CGLS'])
def test_cpu_algorithms(projector, proj_geom, algorithm_type):
    if algorithm_type == 'FBP' and proj_geom['type'] != 'parallel':
        pytest.xfail('Not implemented')
    algorithm_config = make_algorithm_config(algorithm_type, proj_geom, projector)
    output = get_algorithm_output(algorithm_config)
    assert not np.allclose(output, DATA_INIT_VALUE)


@pytest.mark.parametrize(
    'proj_geom,', ['parallel', 'parallel_vec', 'fanflat', 'fanflat_vec'], indirect=True
)
@pytest.mark.parametrize(
    'algorithm_type',
    ['FP_CUDA', 'BP_CUDA', 'FBP_CUDA', 'SIRT_CUDA', 'SART_CUDA', 'CGLS_CUDA', 'EM_CUDA']
)
def test_gpu_algorithms(proj_geom, algorithm_type):
    algorithm_config = make_algorithm_config(algorithm_type, proj_geom)
    output = get_algorithm_output(algorithm_config)
    assert not np.allclose(output, DATA_INIT_VALUE)


@pytest.mark.parametrize('proj_geom', ['parallel'], indirect=True)
@pytest.mark.parametrize('projector', ['linear'], indirect=True)
class TestOptionsCPU:
    def test_fp_sinogram_mask(self, proj_geom, projector, sinogram_mask):
        pytest.xfail('Known bug')
        algorithm_config = make_algorithm_config(
            algorithm_type='FP', proj_geom=proj_geom, projector=projector,
            options={'SinogramMaskId': sinogram_mask}
        )
        projection = get_algorithm_output(algorithm_config)
        assert not np.allclose(projection, DATA_INIT_VALUE)
        mask = (astra.data2d.get(sinogram_mask) > 0)
        assert np.allclose(projection[~mask], DATA_INIT_VALUE)

    def test_fp_volume_mask(self, proj_geom, projector, reconstruction_mask):
        pytest.xfail('Known bug')
        algorithm_no_mask = make_algorithm_config(
            algorithm_type='FP', proj_geom=proj_geom, projector=projector
        )
        algorithm_with_mask = make_algorithm_config(
            algorithm_type='FP', proj_geom=proj_geom, projector=projector,
            options={'VolumeMaskId': reconstruction_mask}
        )
        projection_no_mask = get_algorithm_output(algorithm_no_mask)
        projection_with_mask = get_algorithm_output(algorithm_with_mask)
        assert not np.allclose(projection_with_mask, DATA_INIT_VALUE)
        assert not np.allclose(projection_with_mask, projection_no_mask)

    @pytest.mark.parametrize('algorithm_type', ['BP', 'SIRT', 'SART', 'ART', 'CGLS'])
    def test_sinogram_mask(self, proj_geom, projector, algorithm_type, sinogram_mask):
        algorithm_no_mask = make_algorithm_config(algorithm_type, proj_geom, projector)
        algorithm_with_mask = make_algorithm_config(algorithm_type, proj_geom, projector,
                                                    options={'SinogramMaskId': sinogram_mask})
        reconstruction_no_mask = get_algorithm_output(algorithm_no_mask)
        reconstruction_with_mask = get_algorithm_output(algorithm_with_mask)
        assert not np.allclose(reconstruction_with_mask, DATA_INIT_VALUE)
        assert not np.allclose(reconstruction_with_mask, reconstruction_no_mask)

    @pytest.mark.parametrize('algorithm_type', ['BP', 'SIRT', 'SART', 'ART', 'CGLS'])
    def test_reconstruction_mask(self, proj_geom, projector, algorithm_type, reconstruction_mask):
        if algorithm_type == 'BP':
            pytest.xfail('Known bug')
        algorithm_config = make_algorithm_config(
            algorithm_type, proj_geom, projector,
            options={'ReconstructionMaskId': reconstruction_mask}
        )
        reconstruction = get_algorithm_output(algorithm_config)
        assert not np.allclose(reconstruction, DATA_INIT_VALUE)
        mask = (astra.data2d.get(reconstruction_mask) > 0)
        assert np.allclose(reconstruction[~mask], DATA_INIT_VALUE)

    @pytest.mark.parametrize('projection_order', ['random', 'sequential', 'custom'])
    def test_sart_projection_order(self, proj_geom, projector, projection_order):
        options = {'ProjectionOrder': projection_order}
        if projection_order == 'custom':
            pytest.xfail('Known bug')
            # Set projection order to 0, 5, 10, ..., 175, 1, 6, 11, ... , 176, 2, 7, ...
            proj_order_list = np.arange(N_ANGLES).reshape(-1, 5).T.flatten()
            options['ProjectionOrderList'] = proj_order_list
        algorithm_config = make_algorithm_config(algorithm_type='SART', proj_geom=proj_geom,
                                                 projector=projector, options=options)
        reconstruction = get_algorithm_output(algorithm_config)
        assert not np.allclose(reconstruction, DATA_INIT_VALUE)

    @pytest.mark.parametrize('ray_order', ['sequential', 'custom'])
    def test_art_ray_order(self, proj_geom, projector, ray_order):
        options = {'RayOrder': ray_order}
        if ray_order == 'custom':
            pytest.xfail('Known bug')
            # Every combination of (projection_id, detector_id)
            all_rays = np.mgrid[:N_ANGLES, :DET_COUNT].T.reshape(-1, 2)
            ray_order_list = np.random.permutation(all_rays).flatten()
            options['ProjectionOrderList'] = ray_order_list
        algorithm_config = make_algorithm_config(algorithm_type='ART', proj_geom=proj_geom,
                                                 projector=projector, options=options)
        reconstruction = get_algorithm_output(algorithm_config)
        assert not np.allclose(reconstruction, DATA_INIT_VALUE)

    @pytest.mark.parametrize('filter_type', ['ram-lak', 'none'])
    def test_fbp_filters_basic(self, proj_geom, projector, filter_type):
        algorithm_config = make_algorithm_config(algorithm_type='FBP', proj_geom=proj_geom,
                                                 projector=projector,
                                                 options={'FilterType': filter_type})
        reconstruction = get_algorithm_output(algorithm_config)
        assert not np.allclose(reconstruction, DATA_INIT_VALUE)

    @pytest.mark.parametrize('filter_type', ['tukey', 'gaussian', 'blackman', 'kaiser'])
    def test_fbp_filter_parameter(self, proj_geom, projector, filter_type):
        algorithm_config = make_algorithm_config(
            algorithm_type='FBP', proj_geom=proj_geom, projector=projector,
            options={'FilterType': filter_type, 'FilterParameter': -1.0}
        )
        reconstruction = get_algorithm_output(algorithm_config)
        assert not np.allclose(reconstruction, DATA_INIT_VALUE)

    @pytest.mark.parametrize('filter_type', ['shepp-logan', 'cosine', 'hamming', 'hann'])
    def test_fbp_filter_d(self, proj_geom, projector, filter_type):
        algorithm_config = make_algorithm_config(
            algorithm_type='FBP', proj_geom=proj_geom, projector=projector,
            options={'FilterType': filter_type, 'FilterD': 1.0}
        )
        reconstruction = get_algorithm_output(algorithm_config)
        assert not np.allclose(reconstruction, DATA_INIT_VALUE)

    @pytest.mark.parametrize(
        'custom_filter', ['projection', 'sinogram', 'rprojection', 'rsinogram'], indirect=True
    )
    def test_fbp_custom_filters(self, proj_geom, projector, custom_filter):
        filter_type, filter_data_id = custom_filter
        algorithm_config = make_algorithm_config(
            algorithm_type='FBP', proj_geom=proj_geom, projector=projector,
            options={'FilterType': filter_type, 'FilterSinogramId': filter_data_id}
        )
        reconstruction = get_algorithm_output(algorithm_config)
        assert not np.allclose(reconstruction, DATA_INIT_VALUE)


class TestOptionsGPU:
    @pytest.mark.parametrize('proj_geom,', ['parallel', 'fanflat'], indirect=True)
    def test_detector_supersampling_fp(self, proj_geom):
        algorithm_no_supersampling = make_algorithm_config(
            algorithm_type='FP_CUDA', proj_geom=proj_geom
        )
        algorithm_with_supersampling = make_algorithm_config(
            algorithm_type='FP_CUDA', proj_geom=proj_geom, options={'DetectorSuperSampling': 3}
        )
        proj_no_supersampling = get_algorithm_output(algorithm_no_supersampling)
        proj_with_supersampling = get_algorithm_output(algorithm_with_supersampling)
        assert not np.allclose(proj_with_supersampling, DATA_INIT_VALUE)
        assert not np.allclose(proj_with_supersampling, proj_no_supersampling)

    @pytest.mark.parametrize('proj_geom,', ['parallel', 'fanflat'], indirect=True)
    @pytest.mark.parametrize('algorithm_type', ['SIRT_CUDA', 'SART_CUDA', 'CGLS_CUDA', 'EM_CUDA'])
    def test_detector_supersampling_iterative(self, proj_geom, algorithm_type):
        algorithm_no_supersampling = make_algorithm_config(algorithm_type, proj_geom)
        algorithm_with_supersampling = make_algorithm_config(
            algorithm_type, proj_geom, options={'DetectorSuperSampling': 3}
        )
        rec_no_supersampling = get_algorithm_output(algorithm_no_supersampling)
        rec_with_supersampling = get_algorithm_output(algorithm_with_supersampling)
        assert not np.allclose(rec_with_supersampling, DATA_INIT_VALUE)
        assert not np.allclose(rec_with_supersampling, rec_no_supersampling)

    @pytest.mark.parametrize('proj_geom,', ['parallel', 'fanflat'], indirect=True)
    @pytest.mark.parametrize(
        'algorithm_type', ['BP_CUDA', 'SIRT_CUDA', 'SART_CUDA', 'CGLS_CUDA', 'EM_CUDA']
    )
    def test_pixel_supersampling(self, proj_geom, algorithm_type):
        algorithm_no_supersampling = make_algorithm_config(algorithm_type, proj_geom)
        algorithm_with_supersampling = make_algorithm_config(
            algorithm_type, proj_geom=proj_geom, options={'PixelSuperSampling': 3}
        )
        rec_no_supersampling = get_algorithm_output(algorithm_no_supersampling)
        rec_with_supersampling = get_algorithm_output(algorithm_with_supersampling)
        assert not np.allclose(rec_with_supersampling, DATA_INIT_VALUE)
        assert not np.allclose(rec_with_supersampling, rec_no_supersampling)

    @pytest.mark.parametrize('proj_geom', ['parallel', 'fanflat'], indirect=True)
    @pytest.mark.parametrize('filter_type', ['ram-lak', 'none'])
    def test_fbp_filters_basic(self, proj_geom, filter_type):
        algorithm_config = make_algorithm_config(algorithm_type='FBP_CUDA', proj_geom=proj_geom,
                                                 options={'FilterType': filter_type})
        reconstruction = get_algorithm_output(algorithm_config)
        assert not np.allclose(reconstruction, DATA_INIT_VALUE)

    @pytest.mark.parametrize('proj_geom', ['parallel', 'fanflat'], indirect=True)
    @pytest.mark.parametrize('filter_type', ['tukey', 'gaussian', 'blackman', 'kaiser'])
    def test_fbp_filter_parameter(self, proj_geom, filter_type):
        algorithm_config = make_algorithm_config(
            algorithm_type='FBP_CUDA', proj_geom=proj_geom,
            options={'FilterType': filter_type, 'FilterParameter': -1.0}
        )
        reconstruction = get_algorithm_output(algorithm_config)
        assert not np.allclose(reconstruction, DATA_INIT_VALUE)

    @pytest.mark.parametrize('proj_geom', ['parallel', 'fanflat'], indirect=True)
    @pytest.mark.parametrize('filter_type', ['shepp-logan', 'cosine', 'hamming', 'hann'])
    def test_fbp_filter_d(self, proj_geom, filter_type):
        algorithm_config = make_algorithm_config(
            algorithm_type='FBP_CUDA', proj_geom=proj_geom,
            options={'FilterType': filter_type, 'FilterD': 1.0}
        )
        reconstruction = get_algorithm_output(algorithm_config)
        assert not np.allclose(reconstruction, DATA_INIT_VALUE)

    @pytest.mark.parametrize('proj_geom', ['parallel', 'fanflat'], indirect=True)
    @pytest.mark.parametrize(
        'custom_filter', ['projection', 'sinogram', 'rprojection', 'rsinogram'], indirect=True
    )
    def test_fbp_custom_filters(self, proj_geom, custom_filter):
        filter_type, filter_data_id = custom_filter
        algorithm_config = make_algorithm_config(
            algorithm_type='FBP_CUDA', proj_geom=proj_geom,
            options={'FilterType': filter_type, 'FilterSinogramId': filter_data_id}
        )
        reconstruction = get_algorithm_output(algorithm_config)
        assert not np.allclose(reconstruction, DATA_INIT_VALUE)

    @pytest.mark.parametrize('proj_geom', ['short_scan'], indirect=True)
    def test_short_scan(self, proj_geom):
        algorithm_no_short_scan = make_algorithm_config(
            algorithm_type='FBP_CUDA', proj_geom=proj_geom
        )
        algorithm_with_short_scan = make_algorithm_config(
            algorithm_type='FBP_CUDA', proj_geom=proj_geom, options={'ShortScan': True}
        )
        reconstruction_no_short_scan = get_algorithm_output(algorithm_no_short_scan)
        reconstruction_with_short_scan = get_algorithm_output(algorithm_with_short_scan)
        assert not np.allclose(reconstruction_with_short_scan, DATA_INIT_VALUE)
        assert not np.allclose(reconstruction_with_short_scan, reconstruction_no_short_scan)

    @pytest.mark.parametrize('proj_geom,', ['parallel'], indirect=True)
    @pytest.mark.parametrize('algorithm_type', ['SIRT_CUDA', 'SART_CUDA'])
    def test_min_max_constraint(self, proj_geom, algorithm_type):
        algorithm_no_constrains = make_algorithm_config(algorithm_type, proj_geom)
        algorithm_with_constrains = make_algorithm_config(
            algorithm_type, proj_geom, options={'MinConstraint': 0.0, 'MaxConstraint': 0.125}
        )
        reconstruction_no_constrains = get_algorithm_output(algorithm_no_constrains)
        reconstruction_with_constrains = get_algorithm_output(algorithm_with_constrains)
        assert reconstruction_no_constrains.min() < 0.0
        assert reconstruction_no_constrains.max() > 0.125
        assert reconstruction_with_constrains.min() == 0.0
        assert reconstruction_with_constrains.max() == 0.125

    @pytest.mark.parametrize('proj_geom,', ['parallel'], indirect=True)
    @pytest.mark.parametrize('algorithm_type', ['SIRT_CUDA', 'SART_CUDA', 'CGLS_CUDA'])
    def test_reconstruction_mask(self, proj_geom, algorithm_type, reconstruction_mask):
        algorithm_config = make_algorithm_config(
            algorithm_type, proj_geom, options={'ReconstructionMaskId': reconstruction_mask}
        )
        reconstruction = get_algorithm_output(algorithm_config)
        assert not np.allclose(reconstruction, DATA_INIT_VALUE)
        mask = (astra.data2d.get(reconstruction_mask) > 0)
        assert np.allclose(reconstruction[~mask], DATA_INIT_VALUE)

    @pytest.mark.parametrize('proj_geom,', ['parallel'], indirect=True)
    def test_sinogram_mask(self, proj_geom, sinogram_mask):
        algorithm_no_mask = make_algorithm_config(algorithm_type='SIRT_CUDA', proj_geom=proj_geom)
        algorithm_with_sino_mask = make_algorithm_config(
            algorithm_type='SIRT_CUDA', proj_geom=proj_geom,
            options={'SinogramMaskId': sinogram_mask}
        )
        reconstruction_no_mask = get_algorithm_output(algorithm_no_mask)
        reconstruction_with_sino_mask = get_algorithm_output(algorithm_with_sino_mask)
        assert not np.allclose(reconstruction_with_sino_mask, DATA_INIT_VALUE)
        assert not np.allclose(reconstruction_with_sino_mask, reconstruction_no_mask)

    @pytest.mark.parametrize('proj_geom,', ['parallel'], indirect=True)
    @pytest.mark.parametrize('projection_order', ['random', 'sequential', 'custom'])
    def test_sart_projection_order(self, proj_geom, projection_order):
        options = {'ProjectionOrder': projection_order}
        if projection_order == 'custom':
            pytest.xfail('Known bug')
            # Set projection order to 0, 5, 10, ..., 175, 1, 6, 11, ... , 176, 2, 7, ...
            proj_order_list = np.arange(N_ANGLES).reshape(-1, 5).T.flatten()
            options['ProjectionOrderList'] = proj_order_list
        algorithm_config = make_algorithm_config(
            algorithm_type='SART_CUDA', proj_geom=proj_geom, options=options
        )
        reconstruction = get_algorithm_output(algorithm_config)
        assert not np.allclose(reconstruction, DATA_INIT_VALUE)

    @pytest.mark.parametrize('proj_geom,', ['parallel'], indirect=True)
    @pytest.mark.parametrize('algorithm_type', ['SIRT_CUDA', 'SART_CUDA', 'CGLS_CUDA', 'EM_CUDA'])
    def test_get_res_norm(self, proj_geom, algorithm_type):
        algorithm_config = make_algorithm_config(algorithm_type, proj_geom)
        algorithm_id = astra.algorithm.create(algorithm_config)
        astra.algorithm.run(algorithm_id, 2)
        res_norm = astra.algorithm.get_res_norm(algorithm_id)
        astra.algorithm.delete(algorithm_id)
        assert res_norm > 0.0
