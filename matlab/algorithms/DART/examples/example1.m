clear all;

addpath('..');

%
% Example 1: parallel beam, three slices.
%

% Configuration
proj_count = 20;
slice_count = 3;
dart_iterations = 20;
filename = 'cylinders.png';
outdir = './';
prefix = 'example1';
rho = [0, 1];
tau = 0.5;
gpu_core = 0;

% Load phantom.
I = double(imread(filename)) / 255;

% Create projection and volume geometries.
det_count = size(I, 1);
angles = linspace(0, pi - pi / proj_count, proj_count);
proj_geom = astra_create_proj_geom('parallel3d', 1, 1, slice_count, det_count, angles);
vol_geom = astra_create_vol_geom(det_count, det_count, 1);

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
