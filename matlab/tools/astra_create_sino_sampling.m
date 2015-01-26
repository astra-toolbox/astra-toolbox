function [sino_id, sino] = astra_create_sino_sampling(data, proj_geom, vol_geom, gpu_index, sampling)

%--------------------------------------------------------------------------
% [sino_id, sino] = astra_create_sino_cuda(data, proj_geom, vol_geom, gpu_index)
% 
% Create a GPU based forward projection.
%
% data: input volume, can be either MATLAB-data or an astra-identifier.
% proj_geom: MATLAB struct containing the projection geometry.
% vol_geom: MATLAB struct containing the volume geometry.
% gpu_index: the index of the GPU to use (optional).
% sino_id: identifier of the sinogram data object as it is now stored in
% the astra-library. 
% sino: MATLAB data version of the sinogram.
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
% This file is part of the ASTRA Toolbox
% 
% Copyright: 2010-2015, iMinds-Vision Lab, University of Antwerp
%            2014-2015, CWI, Amsterdam
% License: Open Source under GPLv3
% Contact: astra@uantwerpen.be
% Website: http://sf.net/projects/astra-toolbox
%--------------------------------------------------------------------------
% $Id$


% store volume
if (numel(data) > 1)
	volume_id = astra_mex_data2d('create','-vol', vol_geom, data);
else
	volume_id = data;
end

% store sino
sino_id = astra_mex_data2d('create','-sino', proj_geom, 0);

% create sinogram
cfg = astra_struct('FP_CUDA');
cfg.ProjectionDataId = sino_id;
cfg.VolumeDataId = volume_id;
cfg.option.DetectorSuperSampling = sampling;
if nargin > 3
  cfg.option.GPUindex = gpu_index;
end
alg_id = astra_mex_algorithm('create', cfg);
astra_mex_algorithm('iterate', alg_id);
astra_mex_algorithm('delete', alg_id);

if (numel(data) > 1)
	astra_mex_data2d('delete', volume_id);
end

if nargout >= 2
	sino = astra_mex_data2d('get',sino_id);
end



