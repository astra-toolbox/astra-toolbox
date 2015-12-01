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

% We create a projector set up to use 3 rays per detector element
cfg_proj = astra_struct('cuda');
cfg_proj.option.DetectorSuperSampling = 3;
cfg_proj.ProjectionGeometry = proj_geom;
cfg_proj.VolumeGeometry = vol_geom;
proj_id = astra_mex_projector('create', cfg_proj);


[sinogram3 sinogram_id] = astra_create_sino(P, proj_id);

figure(1); imshow(P, []);
figure(2); imshow(sinogram3, []);


% Create a reconstruction, also using supersampling
rec_id = astra_mex_data2d('create', '-vol', vol_geom);
cfg = astra_struct('SIRT_CUDA');
cfg.ReconstructionDataId = rec_id;
cfg.ProjectionDataId = sinogram_id;
cfg.ProjectorId = proj_id;


% There is also an option for supersampling during the backprojection step.
% This should be used if your detector pixels are smaller than the voxels.

% Set up 2 rays per image pixel dimension, for 4 rays total per image pixel.
% cfg_proj.option.PixelSuperSampling = 2;


alg_id = astra_mex_algorithm('create', cfg);
astra_mex_algorithm('iterate', alg_id, 150);
astra_mex_algorithm('delete', alg_id);

rec = astra_mex_data2d('get', rec_id);
figure(3); imshow(rec, []);

