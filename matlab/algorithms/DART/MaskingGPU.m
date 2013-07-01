% This file is part of the
% All Scale Tomographic Reconstruction Antwerp Toolbox ("ASTRA-Toolbox")
%
% Copyright: iMinds-Vision Lab, University of Antwerp
% License: Open Source under GPLv3
% Contact: mailto:astra@ua.ac.be
% Website: http://astra.ua.ac.be
%
% Author of this DART Algorithm: Wim van Aarle


classdef MaskingGPU < matlab.mixin.Copyable

	% Policy class for masking for DART with GPU accelerated code (deprecated).
	
	%----------------------------------------------------------------------
	properties (Access=public)
		
		radius			= 1;			% SETTING: Radius of masking kernel.
		conn			= 8;			% SETTING: Connectivity window. For 2D: 4 or 8.  For 3D: 6 or 26.
		edge_threshold	= 1;			% SETTING: Number of pixels in the window that should be different.
		gpu_core		= 0;			% SETTING:
		random			= 0.1;			% SETTING: Percentage of random points.  Between 0 and 1.
		
	end
	
	%----------------------------------------------------------------------
	methods (Access=public)
		
		%------------------------------------------------------------------
		function settings = getsettings(this)
			% Returns a structure containing all settings of this object.
			% >> settings = DART.masking.getsettings();				
			settings.radius				= this.radius;
			settings.conn				= this.conn;
			settings.edge_threshold		= this.edge_threshold;
			settings.random				= this.random;
		end
		
		%------------------------------------------------------------------
		function Mask = apply(this, ~, V_in)
			% Applies masking.
			% >> Mask = DART.segmentation.apply(DART, V_in);	
			
			% 2D, one slice
			if size(V_in,3) == 1
				Mask = this.apply_2D(V_in);
						
			% 3D, slice by slice
			elseif this.conn == 4 || this.conn == 8
				Mask = zeros(size(V_in));
				for slice = 1:size(V_in,3)
					Mask(:,:,slice) = this.apply_2D(V_in(:,:,slice)); 
				end
			
			% 3D, full
			else
				error('Full 3D masking on GPU not implemented.')
			end
			
		end
		
	end
		
	%----------------------------------------------------------------------
	methods (Access=protected)
		
		%------------------------------------------------------------------
		function Mask = apply_2D(this, S)
		
			vol_geom = astra_create_vol_geom(size(S));
			data_id = astra_mex_data2d('create', '-vol', vol_geom, S);
			mask_id = astra_mex_data2d('create', '-vol', vol_geom, 0);

			cfg = astra_struct('DARTMASK_CUDA');
			cfg.SegmentationDataId = data_id;
			cfg.MaskDataId = mask_id;
			cfg.option.GPUindex = this.gpu_core;
			%cfg.option.Connectivity = this.conn;
			
			alg_id = astra_mex_algorithm('create',cfg);	
			astra_mex_algorithm('iterate',alg_id,1);
			Mask = astra_mex_data2d('get', mask_id);
		
			astra_mex_algorithm('delete', alg_id);
			astra_mex_data2d('delete', data_id, mask_id);
			
		end	
	end


	
end

