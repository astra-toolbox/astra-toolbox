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
proj_geom = astra_create_proj_geom('parallel', 3.0, 128, linspace2(0,pi,180));
P = phantom(256);

% Because the astra_create_sino_gpu wrapper does not have support for
% all possible algorithm options, we manually create a sinogram
phantom_id = astra_mex_data2d('create', '-vol', vol_geom, P);
sinogram_id = astra_mex_data2d('create', '-sino', proj_geom);
cfg = astra_struct('FP_CUDA');
cfg.VolumeDataId = phantom_id;
cfg.ProjectionDataId = sinogram_id;

% Set up 3 rays per detector element
cfg.option.DetectorSuperSampling = 3;

alg_id = astra_mex_algorithm('create', cfg);
astra_mex_algorithm('run', alg_id);
astra_mex_algorithm('delete', alg_id);
astra_mex_data2d('delete', phantom_id);

sinogram3 = astra_mex_data2d('get', sinogram_id);

figure(1); imshow(P, []);
figure(2); imshow(sinogram3, []);


% Create a reconstruction, also using supersampling
rec_id = astra_mex_data2d('create', '-vol', vol_geom);
cfg = astra_struct('SIRT_CUDA');
cfg.ReconstructionDataId = rec_id;
cfg.ProjectionDataId = sinogram_id;
% Set up 3 rays per detector element
cfg.option.DetectorSuperSampling = 3;

% There is also an option for supersampling during the backprojection step.
% This should be used if your detector pixels are smaller than the voxels.

% Set up 2 rays per image pixel dimension, for 4 rays total per image pixel.
% cfg.option.PixelSuperSampling = 2;


alg_id = astra_mex_algorithm('create', cfg);
astra_mex_algorithm('iterate', alg_id, 150);
astra_mex_algorithm('delete', alg_id);

rec = astra_mex_data2d('get', rec_id);
figure(3); imshow(rec, []);

