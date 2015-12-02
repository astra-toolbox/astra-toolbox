function proj_id = astra_create_projector(type, proj_geom, vol_geom, options)

%--------------------------------------------------------------------------
% proj_id = astra_create_projector(type, proj_geom, vol_geom, options)
% 
% Create a new projector object based on projection and volume geometry.  
% Used when the default values of each projector are sufficient.  
%
% type: type of the projector.  'blob', 'line', 'linear' 'strip', ... See API for more information.
% proj_geom: MATLAB struct containing the projection geometry.
% vol_geom: MATLAB struct containing the volume geometry.
% options: Optional MATLAB struct containing projector options (like: 'GPUindex', 'DetectorSuperSampling', and 'VoxelSuperSampling')
% proj_id: identifier of the projector as it is now stored in the astra-library.
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
% This file is part of the ASTRA Toolbox
% 
% Copyright: 2010-2015, iMinds-Vision Lab, University of Antwerp
%            2014-2015, CWI, Amsterdam
% License: Open Source under GPLv3
% Contact: astra@uantwerpen.be
% Website: http://sf.net/projects/astra-toolbox
% -------------------------------------------------------------------------
% $Id$


cfg_proj = astra_struct(type);
cfg_proj.ProjectionGeometry = proj_geom;
cfg_proj.VolumeGeometry = vol_geom;

if strcmp(type,'blob')
	% Blob options
	blob_size = 2;
	blob_sample_rate = 0.01;
	blob_values = kaiserBessel(2, 10.4, blob_size, 0:blob_sample_rate:blob_size);
	cfg_proj.Kernel.KernelSize = blob_size;
	cfg_proj.Kernel.SampleRate = blob_sample_rate;
	cfg_proj.Kernel.SampleCount = length(blob_values);
	cfg_proj.Kernel.KernelValues = blob_values;
end

if exist('options', 'var')
    cfg_proj.options = options;
end

if strcmp(type,'linear3d') || strcmp(type,'linearcone') || strcmp(type,'cuda3d')
	proj_id = astra_mex_projector3d('create', cfg_proj);
else
	proj_id = astra_mex_projector('create', cfg_proj);
end





