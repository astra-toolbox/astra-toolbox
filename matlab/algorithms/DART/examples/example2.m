clear all;

addpath('..');

%
% Example 2: cone beam, full cube.
%

% Configuration
det_count = 128;
proj_count = 45;
slice_count = det_count;
dart_iterations = 20;
outdir = './';
prefix = 'example2';
rho = [0 0.5 1];
tau = [0.25 0.75];
gpu_core = 0;

% Create phantom.
% I = phantom3d([1 0.9 0.9 0.9 0 0 0 0 0 0; -0.5 0.8 0.8 0.8 0 0 0 0 0 0; -0.5 0.3 0.3 0.3 0 0 0 0 0 0], det_count);
% save('phantom3d', 'I');
load('phantom3d'); % Loads I.

% Create projection and volume geometries.
angles = linspace(0, pi - pi / proj_count, proj_count);
proj_geom = astra_create_proj_geom('cone', 1, 1, slice_count, det_count, angles, 500, 0);
vol_geom = astra_create_vol_geom(det_count, det_count, slice_count);

% Create sinogram.
[sinogram_id, sinogram] = astra_create_sino3d_cuda(I, proj_geom, vol_geom);
astra_mex_data3d('delete', sinogram_id);

%
% DART
%

base.sinogram = sinogram;
base.proj_geom = proj_geom;

D                      = DARTalgorithm(base);

D.tomography           = TomographyDefault3D();
D.tomography.t0        = 100;
D.tomography.t         = 10;
D.tomography.method    = 'SIRT3D_CUDA';
D.tomography.gpu_core  = gpu_core;
D.tomography.use_minc  = 'yes';
% D.tomography.maxc      = 0.003; % Not a sensible value, just for demonstration.

D.segmentation.rho     = rho;
D.segmentation.tau     = tau;

D.smoothing.b          = 0.1;
D.smoothing.full3d     = 'yes';
D.smoothing.gpu_core   = gpu_core;
 
D.masking.random       = 0.1;
D.masking.conn         = 6;
D.masking.gpu_core     = gpu_core;

D.output.directory     = outdir;
D.output.pre           = [prefix '_'];
D.output.save_images   = 'no';
D.output.save_results  = {'stats', 'settings', 'S', 'V'};
D.output.save_interval = dart_iterations;
D.output.verbose       = 'yes';

D.statistics.proj_diff = 'no';

D.initialize();

disp([D.output.directory D.output.pre]);

D.iterate(dart_iterations);

% Convert middle slice of final iteration to png.
load([outdir '/' prefix '_results_' num2str(dart_iterations) '.mat']);
imwritesc(D.S(:, :, round(slice_count / 2)), [outdir '/' prefix '_slice_2_S.png']);
imwritesc(D.V(:, :, round(slice_count / 2)), [outdir '/' prefix '_slice_2_V.png']);
