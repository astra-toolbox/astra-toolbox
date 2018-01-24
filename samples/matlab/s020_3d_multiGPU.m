% -----------------------------------------------------------------------
% This file is part of the ASTRA Toolbox
% 
% Copyright: 2010-2018, imec Vision Lab, University of Antwerp
%            2014-2018, CWI, Amsterdam
% License: Open Source under GPLv3
% Contact: astra@astra-toolbox.com
% Website: http://www.astra-toolbox.com/
% -----------------------------------------------------------------------


% Set up multi-GPU usage.
% This only works for 3D GPU forward projection and back projection.
astra_mex('set_gpu_index', [0 1]);

% Optionally, you can also restrict the amount of GPU memory ASTRA will use.
% The line commented below sets this to 1GB.
%astra_mex('set_gpu_index', [0 1], 'memory', 1024*1024*1024);

vol_geom = astra_create_vol_geom(1024, 1024, 1024);

angles = linspace2(0, pi, 1024);
proj_geom = astra_create_proj_geom('parallel3d', 1.0, 1.0, 1024, 1024, angles);

% Create a simple hollow cube phantom
cube = zeros(1024,1024,1024);
cube(129:896,129:896,129:896) = 1;
cube(257:768,257:768,257:768) = 0;

% Create projection data from this
[proj_id, proj_data] = astra_create_sino3d_cuda(cube, proj_geom, vol_geom);

% Backproject projection data
[bproj_id, bproj_data] = astra_create_backprojection3d_cuda(proj_data, proj_geom, vol_geom);

astra_mex_data3d('delete', proj_id);
astra_mex_data3d('delete', bproj_id);

