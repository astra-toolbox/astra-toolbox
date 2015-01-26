% -----------------------------------------------------------------------
% This file is part of the ASTRA Toolbox
% 
% Copyright: 2010-2015, iMinds-Vision Lab, University of Antwerp
%            2014-2015, CWI, Amsterdam
% License: Open Source under GPLv3
% Contact: astra@uantwerpen.be
% Website: http://sf.net/projects/astra-toolbox
% -----------------------------------------------------------------------

% In this example we will create a reconstruction constrained to
% greyvalues between 0 and 1

vol_geom = astra_create_vol_geom(256, 256);
proj_geom = astra_create_proj_geom('parallel', 1.0, 384, linspace2(0,pi,50));

% As before, create a sinogram from a phantom
P = phantom(256);
[sinogram_id, sinogram] = astra_create_sino_gpu(P, proj_geom, vol_geom);
figure(1); imshow(P, []);
figure(2); imshow(sinogram, []);

% Create a data object for the reconstruction
rec_id = astra_mex_data2d('create', '-vol', vol_geom);

% Set up the parameters for a reconstruction algorithm using the GPU
cfg = astra_struct('SIRT_CUDA');
cfg.ReconstructionDataId = rec_id;
cfg.ProjectionDataId = sinogram_id;
cfg.option.MinConstraint = 0;
cfg.option.MaxConstraint = 1;

% Create the algorithm object from the configuration structure
alg_id = astra_mex_algorithm('create', cfg);

% Run 150 iterations of the algorithm
astra_mex_algorithm('iterate', alg_id, 150);

% Get the result
rec = astra_mex_data2d('get', rec_id);
figure(3); imshow(rec, []);

% Clean up. Note that GPU memory is tied up in the algorithm object,
% and main RAM in the data objects.
astra_mex_algorithm('delete', alg_id);
astra_mex_data2d('delete', rec_id);
astra_mex_data2d('delete', sinogram_id);
