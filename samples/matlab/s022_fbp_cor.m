% -----------------------------------------------------------------------
% This file is part of the ASTRA Toolbox
% 
% Copyright: 2010-2018, imec Vision Lab, University of Antwerp
%            2014-2018, CWI, Amsterdam
% License: Open Source under GPLv3
% Contact: astra@astra-toolbox.com
% Website: http://www.astra-toolbox.com/
% -----------------------------------------------------------------------

cor_shift = 3.6;

vol_geom = astra_create_vol_geom(256, 256);
proj_geom = astra_create_proj_geom('parallel', 1.0, 256, linspace2(0,pi,180));

% Projection geometry with shifted center of rotation
proj_geom_cor = astra_geom_postalignment(proj_geom, cor_shift);

% As before, create a sinogram from a phantom, using the shifted center of rotation
P = phantom(256);
[sinogram_id, sinogram] = astra_create_sino_gpu(P, proj_geom_cor, vol_geom);
figure(1); imshow(P, []);
figure(2); imshow(sinogram, []);

astra_mex_data2d('delete', sinogram_id);

% We now re-create the sinogram data object as we would do when loading
% an external sinogram, using a standard geometry, and try to do a reconstruction,
% to show the misalignment artifacts caused by the shifted center of rotation
sinogram_id = astra_mex_data2d('create', '-sino', proj_geom, sinogram);

% Create a data object for the reconstruction
rec_id = astra_mex_data2d('create', '-vol', vol_geom);

% Set up the parameters for a reconstruction algorithm using the GPU
cfg = astra_struct('FBP_CUDA');
cfg.ReconstructionDataId = rec_id;
cfg.ProjectionDataId = sinogram_id;
alg_id = astra_mex_algorithm('create', cfg);
astra_mex_algorithm('run', alg_id);

% Get the result
rec = astra_mex_data2d('get', rec_id);
figure(3); imshow(rec, []);

astra_mex_algorithm('delete', alg_id);
astra_mex_data2d('delete', rec_id);

% Now change back to the proper, shifted geometry, and do another reconstruction
astra_mex_data2d('change_geometry', sinogram_id, proj_geom_cor);
rec_id = astra_mex_data2d('create', '-vol', vol_geom);

cfg = astra_struct('FBP_CUDA');
cfg.ReconstructionDataId = rec_id;
cfg.ProjectionDataId = sinogram_id;
alg_id = astra_mex_algorithm('create', cfg);
astra_mex_algorithm('run', alg_id);

% Get the result
rec = astra_mex_data2d('get', rec_id);
figure(4); imshow(rec, []);


astra_mex_algorithm('delete', alg_id);
astra_mex_data2d('delete', rec_id);
astra_mex_data2d('delete', sinogram_id);
