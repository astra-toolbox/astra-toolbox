
addpath(genpath('bin/'));
addpath(genpath('matlab/'));

%load('phantom3d');
d = 256;
I = ones(d,d,d);
S = I > 0.5;

%% create geometries
vol_geom2d = astra_create_vol_geom(d,d);
vol_geom3d = astra_create_vol_geom(d,d,d);

proj_geom_parallel = astra_create_proj_geom('parallel', 1, 64, 1:180);
proj_geom_fanflat = astra_create_proj_geom('fanflat', 1, 64, 1:180, 0, 2000);
proj_geom_fanflat_vec = astra_geom_2vec(proj_geom_fanflat);

%% create data objects
vol2d_id = astra_mex_data2d('create', '-vol', vol_geom2d, 0);
vol3d_id = astra_mex_data3d('create', '-vol', vol_geom3d, 0);

proj_parallel_id = astra_mex_data2d('create', '-sino', proj_geom_parallel, 0);
proj_fanflat_id = astra_mex_data2d('create', '-sino', proj_geom_fanflat, 0);
proj_fanflatvec_id = astra_mex_data2d('create', '-sino', proj_geom_fanflat_vec, 0);


%% get geometries
vol_geom2d_new = astra_mex_data2d('get_geometry', vol2d_id);
vol_geom3d_new = astra_mex_data3d('get_geometry', vol3d_id);

proj_geom_parallel_new = astra_mex_data2d('get_geometry', proj_parallel_id);
proj_geom_fanflat_new = astra_mex_data2d('get_geometry', proj_fanflat_id);
proj_geom_fanflat_vec_new = astra_mex_data2d('get_geometry', proj_fanflatvec_id);

proj_geom_fanflat_vec
proj_geom_fanflat_vec_new

proj_geom_fanflat_vec.Vectors(110,:)
proj_geom_fanflat_vec_new.Vectors(110,:)

%%
astra_clear;