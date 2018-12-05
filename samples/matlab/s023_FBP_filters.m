% -----------------------------------------------------------------------
% This file is part of the ASTRA Toolbox
% 
% Copyright: 2010-2018, imec Vision Lab, University of Antwerp
%            2014-2018, CWI, Amsterdam
% License: Open Source under GPLv3
% Contact: astra@astra-toolbox.com
% Website: http://www.astra-toolbox.com/
% -----------------------------------------------------------------------


% This sample script illustrates three ways of passing filters to FBP.
% They work with both the FBP (CPU) and the FBP_CUDA (GPU) algorithms.

N = 256;

vol_geom = astra_create_vol_geom(N, N);
proj_geom = astra_create_proj_geom('parallel', 1.0, N, linspace2(0,pi,180));

proj_id = astra_create_projector('strip', proj_geom, vol_geom);

P = phantom(256);

[sinogram_id, sinogram] = astra_create_sino(P, proj_id);

rec_id = astra_mex_data2d('create', '-vol', vol_geom);

cfg = astra_struct('FBP');
cfg.ReconstructionDataId = rec_id;
cfg.ProjectionDataId = sinogram_id;
cfg.ProjectorId = proj_id;


% 1. Use a standard Ram-Lak filter
cfg.option.FilterType = 'ram-lak';

alg_id = astra_mex_algorithm('create', cfg);
astra_mex_algorithm('run', alg_id);
rec_RL = astra_mex_data2d('get', rec_id);
astra_mex_algorithm('delete', alg_id);


% 2. Define a filter in Fourier space
% This is assumed to be symmetric, and ASTRA therefore expects only half

% The full filter size should be the smallest power of two that is at least
% twice the number of detector pixels.
fullFilterSize = 2*N;
kernel = [linspace2(0, 1, floor(fullFilterSize / 2)) linspace2(1, 0, ceil(fullFilterSize / 2))];
halfFilterSize = floor(fullFilterSize / 2) + 1;
filter = kernel(1:halfFilterSize);

filter_geom = astra_create_proj_geom('parallel', 1.0, halfFilterSize, [0]);
filter_id = astra_mex_data2d('create', '-sino', filter_geom, filter);

cfg.option.FilterType = 'projection';
cfg.option.FilterSinogramId = filter_id;

alg_id = astra_mex_algorithm('create', cfg);
astra_mex_algorithm('run', alg_id);
rec_filter = astra_mex_data2d('get', rec_id);
astra_mex_algorithm('delete', alg_id);

% 3. Define a (spatial) convolution kernel directly
% For a kernel of odd size 2*k+1, the central component is at kernel(k+1)
% For a kernel of even size 2*k, the central component is at kernel(k+1)

kernel = zeros(1, N);
for i = 0:floor(N/4)-1
  f = pi * (2*i + 1);
  val = -2.0 / (f * f);
  kernel(floor(N/2) + 1 + (2*i+1)) = val;
  kernel(floor(N/2) + 1 - (2*i+1)) = val;
end
kernel(floor(N/2)+1) = 0.5;

kernel_geom = astra_create_proj_geom('parallel', 1.0, N, [0]);
kernel_id = astra_mex_data2d('create', '-sino', kernel_geom, kernel);

cfg.option.FilterType = 'rprojection';
cfg.option.FilterSinogramId = kernel_id;

alg_id = astra_mex_algorithm('create', cfg);
astra_mex_algorithm('run', alg_id);
rec_kernel = astra_mex_data2d('get', rec_id);
astra_mex_algorithm('delete', alg_id);

figure(1); imshow(P, []);
figure(2); imshow(rec_RL, []);
figure(3); imshow(rec_filter, []);
figure(4); imshow(rec_kernel, []);


astra_mex_data2d('delete', rec_id);
astra_mex_data2d('delete', sinogram_id);
astra_mex_projector('delete', proj_id);
