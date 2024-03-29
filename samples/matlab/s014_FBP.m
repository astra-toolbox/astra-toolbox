% -----------------------------------------------------------------------
% This file is part of the ASTRA Toolbox
% 
% Copyright: 2010-2022, imec Vision Lab, University of Antwerp
%            2014-2022, CWI, Amsterdam
% License: Open Source under GPLv3
% Contact: astra@astra-toolbox.com
% Website: http://www.astra-toolbox.com/
% -----------------------------------------------------------------------

vol_geom = astra_create_vol_geom(256, 256);
proj_geom = astra_create_proj_geom('fanflat', 1.0, 384, linspace2(0,2*pi,1800), 500, 0);

% As before, create a sinogram from a phantom
P = phantom(256);
[sinogram_id, sinogram] = astra_create_sino_gpu(P, proj_geom, vol_geom);
figure(1); imshow(P, []);
figure(2); imshow(sinogram, []);

% Create a data object for the reconstruction
rec_id = astra_mex_data2d('create', '-vol', vol_geom);

% create configuration 
cfg = astra_struct('FBP_CUDA');
cfg.ReconstructionDataId = rec_id;
cfg.ProjectionDataId = sinogram_id;
cfg.option.FilterType = 'Ram-Lak';

% possible values for FilterType:
% none, ram-lak, shepp-logan, cosine, hamming, hann, tukey, lanczos,
% triangular, gaussian, barlett-hann, blackman, nuttall, blackman-harris,
% blackman-nuttall, flat-top, kaiser, parzen


% Create and run the algorithm object from the configuration structure
alg_id = astra_mex_algorithm('create', cfg);
astra_mex_algorithm('run', alg_id);

% Get the result
rec = astra_mex_data2d('get', rec_id);
figure(3); imshow(rec, []);

% Clean up. Note that GPU memory is tied up in the algorithm object,
% and main RAM in the data objects.
astra_mex_algorithm('delete', alg_id);
astra_mex_data2d('delete', rec_id);
astra_mex_data2d('delete', sinogram_id);
