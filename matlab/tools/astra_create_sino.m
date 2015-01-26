function [sino_id, sino] = astra_create_sino(data, proj_id)

%--------------------------------------------------------------------------
% [sino_id, sino] = astra_create_sino(data, proj_id)
% 
% Create a CPU based forward projection.
%
% data: input volume, can be either MATLAB-data or an astra-identifier.
% proj_id: identifier of the projector as it is stored in the astra-library
% sino_id: identifier of the sinogram data object as it is now stored in the astra-library.
% sino: MATLAB data version of the sinogram
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


% get projection geometry
proj_geom = astra_mex_projector('projection_geometry', proj_id);
vol_geom = astra_mex_projector('volume_geometry', proj_id);

% store volume
if (numel(data) > 1)
	volume_id = astra_mex_data2d('create','-vol', vol_geom, data);
else
	volume_id = data;
end

% store sino
sino_id = astra_mex_data2d('create','-sino', proj_geom, 0);

if astra_mex_projector('is_cuda', proj_id)
	cfg = astra_struct('FP_CUDA');
else
	cfg = astra_struct('FP');
end

cfg.ProjectorId = proj_id;
cfg.ProjectionDataId = sino_id;
cfg.VolumeDataId = volume_id;

% create sinogram
alg_id = astra_mex_algorithm('create', cfg);
astra_mex_algorithm('iterate', alg_id);
astra_mex_algorithm('delete', alg_id);

if (numel(data) > 1)
	astra_mex_data2d('delete', volume_id);
end

if nargout >= 2
	sino = astra_mex_data2d('get',sino_id);
end



