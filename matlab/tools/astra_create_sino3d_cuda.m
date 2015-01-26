function [sino_id, sino] = astra_create_sino3d_cuda(data, proj_geom, vol_geom)

%--------------------------------------------------------------------------
% [sino_id, sino] = astra_create_sino3d_cuda(data, proj_geom, vol_geom)
% 
% Create a GPU based forward projection.
%
% data: input volume, can be either MATLAB-data or an astra-identifier.
% proj_geom: MATLAB struct containing the projection geometry.
% vol_geom: MATLAB struct containing the volume geometry.
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
	if (strcmp(class(data),'single'))
		% read-only link
		volume_id = astra_mex_data3d('link','-vol', vol_geom, data, 1);
	else
		volume_id = astra_mex_data3d('create','-vol', vol_geom, data);
	end
else
	volume_id = data;
end

% store sino
sino_id = astra_mex_data3d('create','-sino', proj_geom, 0);

% create sinogram
cfg = astra_struct('FP3D_CUDA');
cfg.ProjectionDataId = sino_id;
cfg.VolumeDataId = volume_id;
alg_id = astra_mex_algorithm('create', cfg);
astra_mex_algorithm('iterate', alg_id);
astra_mex_algorithm('delete', alg_id);

if (numel(data) > 1)
	astra_mex_data3d('delete', volume_id);
end

if nargout >= 2
	sino = astra_mex_data3d('get',sino_id);
end



