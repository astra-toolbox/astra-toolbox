function [vol_id, vol] = astra_create_backprojection3d_cuda(data, proj_geom, vol_geom)

%--------------------------------------------------------------------------
% [vol_id, vol] = astra_create_backprojection3d_cuda(data, proj_geom, vol_geom)
% 
% Create a GPU based backprojection.
%
% data: input projection data, can be either MATLAB-data or an astra-identifier.
% proj_geom: MATLAB struct containing the projection geometry.
% vol_geom: MATLAB struct containing the volume geometry.
% vol_id: identifier of the volume data object as it is now stored in
% the astra-library. 
% vol: MATLAB data version of the volume.
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


% store projection data
if (numel(data) > 1)
	sino_id = astra_mex_data3d('create','-proj3d', proj_geom, data);
else
	sino_id = data;
end

% store volume
vol_id = astra_mex_data3d('create','-vol', vol_geom, 0);

% create sinogram
cfg = astra_struct('BP3D_CUDA');
cfg.ProjectionDataId = sino_id;
cfg.ReconstructionDataId = vol_id;
alg_id = astra_mex_algorithm('create', cfg);
astra_mex_algorithm('iterate', alg_id);
astra_mex_algorithm('delete', alg_id);

if (numel(data) > 1)
	astra_mex_data3d('delete', sino_id);
end

if nargout >= 2
	vol = astra_mex_data3d('get',vol_id);
end



