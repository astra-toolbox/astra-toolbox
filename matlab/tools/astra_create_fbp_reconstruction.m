function [FBP_id, FBP] = astra_create_fbp_reconstruction(sinogram, proj_id)

proj_geom = astra_mex_projector('projection_geometry', proj_id);
vol_geom = astra_mex_projector('volume_geometry', proj_id);

if numel(sinogram) == 1
	sinogram_id = sinogram;
else
	sinogram_id = astra_mex_data2d('create', '-sino', proj_geom, sinogram);
end

FBP_id = astra_mex_data2d('create','-vol',vol_geom, 0);

cfg = astra_struct('FBP_CUDA');
cfg.ProjectionDataId = sinogram_id;
cfg.ReconstructionDataId = FBP_id;
cfg.FilterType = 'Ram-Lak';
cfg.ProjectorId = proj_id;
cfg.Options.GPUindex = 0;
alg_id = astra_mex_algorithm('create', cfg);
astra_mex_algorithm('run', alg_id);
astra_mex_algorithm('delete', alg_id);

if numel(sinogram) ~= 1
	astra_mex_data2d('delete', sinogram_id);
end

FBP = astra_mex_data2d('get', FBP_id);
