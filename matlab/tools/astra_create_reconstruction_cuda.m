function [recon_id, recon] = astra_create_reconstruction_cuda(rec_type, proj_geom, vol_geom, sinogram, iterations, use_mask, mask, use_minc, minc, use_maxc, maxc)

%--------------------------------------------------------------------------
% [recon_id, recon] = astra_create_reconstruction_cuda(rec_type, proj_geom, vol_geom, sinogram, iterations, use_mask, mask, use_minc, minc, use_maxc, maxc)
%
% Create a GPU based iterative reconstruction.
%
% rec_type: reconstruction type, defaults to 'SIRT_CUDA' if empty
% proj_geom: projection geometry struct
% vol_geom: volume geometry struct
% sinogram: sinogram data OR sinogram identifier
% iterations: number of iterations to perform
% use_mask: use a reconstrucionmask? 'yes' or 'no'
% mask: mask data OR mask identifier.
% use_minc: use a minimum constraint? 'yes' or 'no'
% minc: minimum constraint value
% use_maxc: use a maximum constraint? 'yes' or 'no'
% maxc: maximum constraint value
% recon_id: identifier of the reconstruction data object as it is now stored in the astra-library
% recon: MATLAB data version of the reconstruction
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


if numel(sinogram) == 1
	sinogram_id = sinogram;
else
	sinogram_id = astra_mex_data2d('create', '-sino', proj_geom, sinogram);
end

% create reconstruction object
recon_id = astra_mex_data2d('create', '-vol', vol_geom, 0);

if strcmp(rec_type,'')
  rec_type = 'SIRT_CUDA';
end

% configure
cfg = astra_struct('SIRT_CUDA');
cfg.ProjectionGeometry = proj_geom;
cfg.ReconstructionGeometry = vol_geom;
cfg.ProjectionDataId = sinogram_id;
cfg.ReconstructionDataId = recon_id;
if strcmp(use_mask,'yes')
	if numel(mask) == 1
		mask_id = mask;
	else
		mask_id = astra_mex_data2d('create', '-vol', vol_geom, mask);
	end
	cfg.options.ReconstructionMaskId = mask_id;	
end
cfg.options.UseMinConstraint = use_minc;
cfg.options.MinConstraintValue = minc;
cfg.options.UseMaxConstraint = use_maxc;
cfg.options.MaxConstraintValue = maxc;
alg_id = astra_mex_algorithm('create', cfg);

% iterate
astra_mex_algorithm('iterate', alg_id, iterations);

% return object
recon = astra_mex_data2d('get', recon_id);

% garbage collection
astra_mex_algorithm('delete', alg_id);
if numel(sinogram) ~= 1
	astra_mex_data2d('delete', sinogram_id);
end

if strcmp(use_mask,'yes')
	if numel(mask) ~= 1
		astra_mex_data2d('delete', mask_id);
	end
end

