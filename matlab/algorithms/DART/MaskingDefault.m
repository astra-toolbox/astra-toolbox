%--------------------------------------------------------------------------
% This file is part of the ASTRA Toolbox
%
% Copyright: 2010-2015, iMinds-Vision Lab, University of Antwerp
%            2014-2015, CWI, Amsterdam
% License: Open Source under GPLv3
% Contact: astra@uantwerpen.be
% Website: http://sf.net/projects/astra-toolbox
%--------------------------------------------------------------------------

classdef MaskingDefault < matlab.mixin.Copyable

	% Default policy class for masking for DART.
	
	%----------------------------------------------------------------------
	properties (Access=public)
		
		radius			= 1;			% SETTING: Radius of masking kernel.
		conn			= 8;			% SETTING: Connectivity window. For 2D: 4 or 8.  For 3D: 6 or 26.
		edge_threshold	= 1;			% SETTING: Number of pixels in the window that should be different.
		random			= 0.1;			% SETTING: Percentage of random points.  Between 0 and 1.
		gpu				= 'yes';		% SETTING: Use gpu? {'yes', 'no'}
		gpu_core		= 0;			% SETTING: Which gpu core to use, only when gpu='yes'.
				
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
		function Mask = apply(this, ~, S)
			% Applies masking.
			% >> Mask = DART.segmentation.apply(DART, S);	
			
			% 2D, one slice
			if size(S,3) == 1
				if strcmp(this.gpu,'yes')
					Mask = this.apply_2D_gpu(S);
				else
					Mask = this.apply_2D(S);
				end
						
			% 3D, slice by slice
			elseif this.conn == 4 || this.conn == 8
				Mask = zeros(size(S));
				for slice = 1:size(S,3)
					if strcmp(this.gpu,'yes')
						Mask(:,:,slice) = this.apply_2D_gpu(S(:,:,slice));
					else
						Mask(:,:,slice) = this.apply_2D(S(:,:,slice));						
					end
				end
			
			% 3D, full
			else
				if strcmp(this.gpu,'yes')
					Mask = this.apply_3D_gpu(S);	
				else
					Mask = this.apply_3D(S);	
				end
			end
			
		end
		
	end
		
	%----------------------------------------------------------------------
	methods (Access=protected)
		
		%------------------------------------------------------------------
		function Mask = apply_2D_gpu(this, S)
		
			vol_geom = astra_create_vol_geom(size(S));
			data_id = astra_mex_data2d('create', '-vol', vol_geom, S);
			mask_id = astra_mex_data2d('create', '-vol', vol_geom, 0);

			cfg = astra_struct('DARTMASK_CUDA');
			cfg.SegmentationDataId = data_id;
			cfg.MaskDataId = mask_id;
			cfg.option.GPUindex = this.gpu_core;
			cfg.option.Connectivity = this.conn;
			cfg.option.Threshold = this.edge_threshold;
			cfg.option.Radius = this.radius;
			
			alg_id = astra_mex_algorithm('create',cfg);	
			astra_mex_algorithm('iterate',alg_id,1);
			Mask = astra_mex_data2d('get', mask_id);
		
			astra_mex_algorithm('delete', alg_id);
			astra_mex_data2d('delete', data_id, mask_id);
			
			% random
			RandomField = double(rand(size(S)) < this.random);

			% combine
			Mask = or(Mask, RandomField);			
			
		end			
		
		%------------------------------------------------------------------
		function Mask = apply_2D(this, S)
		
			r = this.radius;
			w = 2 * r + 1;
			
			kernel = Kernels.BinaryPixelKernel(r, this.conn);
			
			% edges
			Xlarge = zeros(size(S,1)+w-1, size(S,2)+w-1); 
			Xlarge(1+r:end-r, 1+r:end-r) = S;

			Edges = zeros(size(S));
			for s = -r:r
				for t = -r:r
					if kernel(s+r+1, t+r+1) == 0
						continue
					end
					Temp = abs(Xlarge(1+r:end-r, 1+r:end-r) - Xlarge(1+r+s:end-r+s, 1+r+t:end-r+t));
					Edges(Temp > eps) = Edges(Temp > eps) + 1;
				end
			end

			Edges = Edges > this.edge_threshold;
			
			% random
			RandomField = double(rand(size(S)) < this.random);

			% combine
			Mask = or(Edges, RandomField);
			
		end		
		
		%------------------------------------------------------------------
		function Mask = apply_3D(this, S)
		
			r = this.radius;
			w = 2 * r + 1;
			
			kernel = Kernels.BinaryPixelKernel(r, this.conn);			
			
			% edges
			Xlarge = zeros(size(S,1)+w-1, size(S,2)+w-1, size(S,3)+w-1); 
			Xlarge(1+r:end-r, 1+r:end-r, 1+r:end-r) = S;

			Edges = zeros(size(S));
			for s = -r:r
				for t = -r:r
					for u = -r:r
						if kernel(s+r+1, t+r+1, u+r+1) == 0
							continue
						end
						Temp = abs(Xlarge(1+r:end-r, 1+r:end-r, 1+r:end-r) - Xlarge(1+r+s:end-r+s, 1+r+t:end-r+t, 1+r+u:end-r+u));
						Edges(Temp > eps) = 1;
					end
				end
			end
			
			clear Xlarge;

			% random
			RandomField = double(rand(size(S)) < this.random);

			% combine
			Mask = or(Edges, RandomField);
			
		end		
		
		%------------------------------------------------------------------
		function Mask = apply_3D_gpu(this, S)
	
			vol_geom = astra_create_vol_geom(size(S,2), size(S,1), size(S,3));
			data_id = astra_mex_data3d('create', '-vol', vol_geom, S);

			cfg = astra_struct('DARTMASK3D_CUDA');
			cfg.SegmentationDataId = data_id;
			cfg.MaskDataId = data_id;
			cfg.option.GPUindex = this.gpu_core;
			cfg.option.Connectivity = this.conn;
			cfg.option.Threshold = this.edge_threshold;
			cfg.option.Radius = this.radius;
			
			alg_id = astra_mex_algorithm('create',cfg);	
			astra_mex_algorithm('iterate',alg_id,1);
			Mask = astra_mex_data3d('get', data_id);
		
			astra_mex_algorithm('delete', alg_id);
			astra_mex_data3d('delete', data_id);
			
			% random
			RandomField = double(rand(size(S)) < this.random);

			% combine
			Mask = or(Mask, RandomField);			
			
		end					
		
		%------------------------------------------------------------------
	end
	
end

