%--------------------------------------------------------------------------
% This file is part of the ASTRA Toolbox
%
% Copyright: 2010-2014, iMinds-Vision Lab, University of Antwerp
%                 2014, CWI, Amsterdam
% License: Open Source under GPLv3
% Contact: astra@uantwerpen.be
% Website: http://sf.net/projects/astra-toolbox
%--------------------------------------------------------------------------

function base = dart_create_base_phantom(Im, angle_count, angle_range, gpu_core)

base.phantom = imreadgs(Im);
base.proj_geom = astra_create_proj_geom('parallel', 1, size(base.phantom,1), linspace2(angle_range(1),angle_range(2),angle_count));
base.vol_geom = astra_create_vol_geom(size(base.phantom,1),size(base.phantom,2));
[sino_id, base.sinogram] = astra_create_sino_cuda(base.phantom, base.proj_geom, base.vol_geom, gpu_core);
astra_mex_data2d('delete',sino_id);
