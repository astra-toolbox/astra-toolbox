%--------------------------------------------------------------------------
% This file is part of the ASTRA Toolbox
%
% Copyright: 2010-2014, iMinds-Vision Lab, University of Antwerp
%                 2014, CWI, Amsterdam
% License: Open Source under GPLv3
% Contact: astra@uantwerpen.be
% Website: http://sf.net/projects/astra-toolbox
%--------------------------------------------------------------------------

addpath('../');

%% Example 1: 2D parallel beam, cuda

% configuration
proj_count		= 20;
dart_iterations = 20;
filename		= 'cylinders.png';
outdir			= './';
prefix			= 'example1';
rho				= [0, 255];
tau				= 128;
gpu_core		= 0;

% load phantom
I = imreadgs(filename);

% create projection and volume geometries
det_count = size(I, 1);
angles = linspace2(0, pi, proj_count);
proj_geom = astra_create_proj_geom('parallel', 1, det_count, angles);
vol_geom = astra_create_vol_geom(det_count, det_count);

% create sinogram
[proj_id]               = astra_create_projector('strip', proj_geom, vol_geom);
[sinogram_id, sinogram] = astra_create_sino(I, proj_id);
astra_mex_data2d('delete', sinogram_id);

% DART
D						= DARTalgorithm(sinogram, proj_geom);
D.t0					= 100;
D.t						= 10;

D.tomography.gpu	    = 'no';
D.tomography.method		= 'SIRT';
D.tomography.proj_type  = 'strip';
D.tomography.gpu_core	= gpu_core;
D.tomography.use_minc	= 'yes';

D.segmentation.rho		= rho;
D.segmentation.tau		= tau;

D.smoothing.gpu	        = 'no'; 
D.smoothing.b			= 0.1;
D.smoothing.gpu_core	= gpu_core;

D.masking.gpu	        = 'no'; 
D.masking.random		= 0.1;
D.masking.gpu_core		= gpu_core;

D.output.directory		= outdir;
D.output.pre			= [prefix '_'];
D.output.save_images	= 'no';
D.output.save_results	= {'stats', 'settings', 'S', 'V'};
D.output.save_interval	= dart_iterations;
D.output.verbose		= 'yes';

D.statistics.proj_diff	= 'no';

D.initialize();

D.iterate(dart_iterations);

% save the reconstruction and the segmentation to file
imwritesc(D.S, [outdir '/' prefix '_S.png']);
imwritesc(D.V, [outdir '/' prefix '_V.png']);
