% -----------------------------------------------------------------------
% This file is part of the ASTRA Toolbox
% 
% Copyright: 2010-2015, iMinds-Vision Lab, University of Antwerp
%            2014-2015, CWI, Amsterdam
% License: Open Source under GPLv3
% Contact: astra@uantwerpen.be
% Website: http://sf.net/projects/astra-toolbox
% -----------------------------------------------------------------------

% Create a 3D volume geometry.
% Parameter order: rows, colums, slices (y, x, z)
vol_geom = astra_create_vol_geom(64, 48, 32);


% Create volumes

% initialized to zero
v0 = astra_mex_data3d('create', '-vol', vol_geom);

% initialized to 3.0
v1 = astra_mex_data3d('create', '-vol', vol_geom, 3.0);

% initialized to a matrix. A may be a single or double array.
% Coordinate order: column, row, slice (x, y, z)
A = zeros(48, 64, 32);
v2 = astra_mex_data3d('create', '-vol', vol_geom, A);


% Projection data

% 2 projection directions, along x and y axis resp.
V = [ 1 0 0  0 0 0   0 1 0  0 0 1 ; ...
      0 1 0  0 0 0  -1 0 0  0 0 1 ];
% 32 rows (v), 64 columns (u)
proj_geom = astra_create_proj_geom('parallel3d_vec', 32, 64, V);

s0 = astra_mex_data3d('create', '-proj3d', proj_geom);

% Initialization to a scalar or zero works exactly as with a volume.

% Initialized to a matrix:
% Coordinate order: column (u), angle, row (v)
A = zeros(64, 2, 32);
s1 = astra_mex_data3d('create', '-proj3d', proj_geom, A);


% Retrieve data:
R = astra_mex_data3d('get', v1);

% Retrieve data as a single array. Since astra internally stores
% data as single precision, this is more efficient:
R = astra_mex_data3d('get_single', v1);



% Delete all created data objects
astra_mex_data3d('delete', v0);
astra_mex_data3d('delete', v1);
astra_mex_data3d('delete', v2);
astra_mex_data3d('delete', s0);
astra_mex_data3d('delete', s1);
