% -----------------------------------------------------------------------
% This file is part of the ASTRA Toolbox
% 
% Copyright: 2010-2015, iMinds-Vision Lab, University of Antwerp
%            2014-2015, CWI, Amsterdam
% License: Open Source under GPLv3
% Contact: astra@uantwerpen.be
% Website: http://sf.net/projects/astra-toolbox
% -----------------------------------------------------------------------

% Create two volume geometries
vol_geom1 = astra_create_vol_geom(256, 256);
vol_geom2 = astra_create_vol_geom(512, 256);

% Create volumes
v0 = astra_mex_data2d('create', '-vol', vol_geom1);
v1 = astra_mex_data2d('create', '-vol', vol_geom2);
v2 = astra_mex_data2d('create', '-vol', vol_geom2);

% Show the currently allocated volumes
astra_mex_data2d('info');


astra_mex_data2d('delete', v2);
astra_mex_data2d('info');

astra_mex_data2d('clear');
astra_mex_data2d('info');



% The same clear and info command also work for other object types:
astra_mex_algorithm('info');
astra_mex_data3d('info');
astra_mex_projector('info');
astra_mex_matrix('info');
