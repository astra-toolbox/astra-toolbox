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


% Create volumes

% initialized to zero
v0 = astra_mex_data2d('create', '-vol', vol_geom);

% initialized to 3.0
v1 = astra_mex_data2d('create', '-vol', vol_geom, 3.0);

% initialized to a matrix. A may be a single, double or logical (0/1) array.
A = phantom(256);
v2 = astra_mex_data2d('create', '-vol', vol_geom, A);


% Projection data
s0 = astra_mex_data2d('create', '-sino', proj_geom);
% Initialization to a scalar or a matrix also works, exactly as with a volume.


% Update data

% set to zero
astra_mex_data2d('store', v0, 0);

% set to a matrix
astra_mex_data2d('store', v2, A);



% Retrieve data

R = astra_mex_data2d('get', v2);
imshow(R, []);


% Retrieve data as a single array. Since astra internally stores
% data as single precision, this is more efficient:

R = astra_mex_data2d('get_single', v2);


% Free memory
astra_mex_data2d('delete', v0);
astra_mex_data2d('delete', v1);
astra_mex_data2d('delete', v2);
astra_mex_data2d('delete', s0);
