% -----------------------------------------------------------------------
% This file is part of the ASTRA Toolbox
% 
% Copyright: 2010-2015, iMinds-Vision Lab, University of Antwerp
%            2014-2015, CWI, Amsterdam
% License: Open Source under GPLv3
% Contact: astra@uantwerpen.be
% Website: http://sf.net/projects/astra-toolbox
% -----------------------------------------------------------------------

vol_geom = astra_create_vol_geom(256, 256);
proj_geom = astra_create_proj_geom('parallel', 1.0, 384, linspace2(0,pi,180));
P = phantom(256);

% Create a sinogram from a phantom, using GPU #1. (The default is #0)
[sinogram_id, sinogram] = astra_create_sino_gpu(P, proj_geom, vol_geom, 1);


% Set up the parameters for a reconstruction algorithm using the GPU
rec_id = astra_mex_data2d('create', '-vol', vol_geom);
cfg = astra_struct('SIRT_CUDA');
cfg.ReconstructionDataId = rec_id;
cfg.ProjectionDataId = sinogram_id;

% Use GPU #1 for the reconstruction. (The default is #0.)
cfg.option.GPUindex = 1;

% Run 150 iterations of the algorithm
alg_id = astra_mex_algorithm('create', cfg);
astra_mex_algorithm('iterate', alg_id, 150);
rec = astra_mex_data2d('get', rec_id);


% Clean up.
astra_mex_algorithm('delete', alg_id);
astra_mex_data2d('delete', rec_id);
astra_mex_data2d('delete', sinogram_id);
