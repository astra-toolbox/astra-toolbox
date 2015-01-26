function [vol_id, vol] = astra_create_backprojection(data, proj_id)

%--------------------------------------------------------------------------
% [vol_id, vol] = astra_create_backprojection(data, proj_id)
% 
% Create a CPU based back projection.
%
% data: input sinogram, can be either MATLAB-data or an astra-identifier.
% proj_id: identifier of the projector as it is stored in the astra-library
% vol_id: identifier of the volume data object as it is now stored in the astra-library.
% vol: MATLAB data version of the volume
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

% store sinogram
if (numel(data) > 1)
	sino_id = astra_mex_data2d('create','-sino', proj_geom, data);
else
	sino_id = data;
end

% store volume
vol_id = astra_mex_data2d('create','-vol', vol_geom, 0);

if astra_mex_projector('is_cuda', proj_id)
	cfg = astra_struct('BP_CUDA');
else
	cfg = astra_struct('BP');
end

cfg.ProjectorId = proj_id;
cfg.ProjectionDataId = sino_id;
cfg.ReconstructionDataId = vol_id;

% create backprojection
alg_id = astra_mex_algorithm('create', cfg);
astra_mex_algorithm('iterate', alg_id);
astra_mex_algorithm('delete', alg_id);

if (numel(data) > 1)
	astra_mex_data2d('delete', sino_id);
end

if nargout >= 2
	vol = astra_mex_data2d('get',vol_id);
end



