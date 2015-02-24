
addpath(genpath('bin/'));
addpath(genpath('matlab/'));

%load('phantom3d');
d = 256;
I = ones(d,d,d);
S = I > 0.5;

%% create geometries
vol_geom2d = astra_create_vol_geom(d,d);
vol_geom3d = astra_create_vol_geom(d,d,d);


%% create data objects
vol2d_id = astra_mex_data2d('create', '-vol', vol_geom2d, 0);
vol3d_id = astra_mex_data3d('create', '-vol', vol_geom3d, 0);

%% get geometries
vol_geom2d_new = astra_mex_data2d('get_geometry', vol2d_id);
vol_geom3d_new = astra_mex_data3d('get_geometry', vol3d_id);


astra_clear;