%--------------------------------------------------------------------------
% This file is part of the ASTRA Toolbox
%
% Copyright: 2010-2014, iMinds-Vision Lab, University of Antwerp
%                 2014, CWI, Amsterdam
% License: Open Source under GPLv3
% Contact: astra@uantwerpen.be
% Website: http://sf.net/projects/astra-toolbox
%--------------------------------------------------------------------------

clear all;

addpath('../');

%%
% Example 1: 2D parallel beam, cuda
%

% Configuration
proj_count		= 20;
dart_iterations = 20;
filename		= 'cylinders.png';
outdir			= './';
prefix			= 'example1';
rho				= [0, 255];
tau				= 128;
gpu_core		= 0;

% Load phantom
I = imreadgs(filename);

% Create projection and volume geometries
det_count = size(I, 1);
angles = linspace2(0, pi, proj_count);
proj_geom = astra_create_proj_geom('parallel', 1, det_count, angles);
vol_geom = astra_create_vol_geom(det_count, det_count, 1);

% Create sinogram.
[sinogram_id, sinogram] = astra_create_sino_cuda(I, proj_geom, vol_geom);
astra_mex_data2d('delete', sinogram_id);

%%
% DART
%

%base.sinogram = sinogram;
%base.proj_geom = proj_geom;

D						= DARTalgorithm(sinogram, proj_geom);
D.t0					= 100;
D.t						= 10;

D.tomography.method		= 'SIRT_CUDA';
D.tomography.gpu_core	= gpu_core;
D.tomography.use_minc	= 'yes';

D.segmentation.rho		= rho;
D.segmentation.tau		= tau;

D.smoothing.b			= 0.1;
D.smoothing.gpu_core	= gpu_core;
 
D.masking.random		= 0.1;
D.masking.gpu_core		= gpu_core;

D.output.directory     = outdir;
D.output.pre           = [prefix '_'];
D.output.save_images   = 'no';
D.output.save_results  = {'stats', 'settings', 'S', 'V'};
D.output.save_interval = dart_iterations;
D.output.verbose       = 'yes';

D.statistics.proj_diff = 'no';

D.initialize();

D.iterate(dart_iterations);

%%
% Convert middle slice of final iteration to png.
%
imwritesc(D.S, [outdir '/' prefix '_S.png']);
imwritesc(D.V, [outdir '/' prefix '_V.png']);
