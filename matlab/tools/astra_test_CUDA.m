%--------------------------------------------------------------------------
% Perform a basic test of ASTRA CPU and CUDA functionality.
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
% This file is part of the ASTRA Toolbox
% 
% Copyright: 2010-2018, imec Vision Lab, University of Antwerp
%            2014-2018, CWI, Amsterdam
% License: Open Source under GPLv3
% Contact: astra@astra-toolbox.com
% Website: http://www.astra-toolbox.com/
%--------------------------------------------------------------------------

%%
fprintf('Getting GPU info...')
astra_get_gpu_info()

%%
astra_test_noCUDA()

%%
fprintf('Testing basic CUDA 2D functionality...')

vg = astra_create_vol_geom(2, 32);
pg = astra_create_proj_geom('parallel', 1, 32, [0]);
proj_id = astra_create_projector('cuda', pg, vg);

vol = rand(2, 32);
[sino_id, sino] = astra_create_sino(vol, proj_id);
astra_mex_data2d('delete', sino_id);
astra_mex_projector('delete', proj_id);

err = max(abs(sino - sum(vol)));

if err < 1e-6
  disp('Ok')
else
  disp('Error')
end

%%
fprintf('Testing basic CUDA 3D functionality...')

vg = astra_create_vol_geom(2, 32, 32);
pg = astra_create_proj_geom('parallel3d', 1, 1, 32, 32, [0]);

vol = rand(32, 2, 32);
[sino_id, sino] = astra_create_sino3d_cuda(vol, pg, vg);
astra_mex_data3d('delete', sino_id);

err = max(max(abs(sino - sum(vol,2))));

if err < 1e-6
  disp('Ok')
else
  disp('Error')
end

