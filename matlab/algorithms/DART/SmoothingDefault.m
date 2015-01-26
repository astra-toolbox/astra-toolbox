%--------------------------------------------------------------------------
% This file is part of the ASTRA Toolbox
%
% Copyright: 2010-2015, iMinds-Vision Lab, University of Antwerp
%            2014-2015, CWI, Amsterdam
% License: Open Source under GPLv3
% Contact: astra@uantwerpen.be
% Website: http://sf.net/projects/astra-toolbox
%--------------------------------------------------------------------------

classdef SmoothingDefault < matlab.mixin.Copyable
	 
	% Default policy class for smoothing for DART.
	
	%----------------------------------------------------------------------
	properties (Access=public)
		radius		= 1;			% SETTING: Radius of smoothing kernel.
		b			= 0.1;			% SETTING: Intensity of smoothing.  Between 0 and 1.	
		full3d		= 'yes';		% SETTING: smooth in 3D? {'yes','no'}
		gpu			= 'yes';		% SETTING: Use gpu? {'yes', 'no'}
		gpu_core	= 0;			% SETTING: Which gpu core to use, only when gpu='yes'.		
	end
	
	
	%----------------------------------------------------------------------
	methods (Access=public)

		%------------------------------------------------------------------
		function settings = getsettings(this)
			% Returns a structure containing all settings of this object.
			% >> settings = DART.smoothing.getsettings();					
			settings.radius				= this.radius;
			settings.b					= this.b;
			settings.full3d				= this.full3d;
		end		
		
		%------------------------------------------------------------------
		function V_out = apply(this, ~, V_in)
			% Applies smoothing.
			% >> V_out = DART.smoothing.apply(DART, V_in);	
			
			% 2D, one slice
			if size(V_in,3) == 1
				if strcmp(this.gpu,'yes')
					V_out = this.apply_2D_gpu(V_in);
				else
					V_out = this.apply_2D(V_in);
				end
						
			% 3D, slice by slice
			elseif ~strcmp(this.full3d,'yes')
				V_out = zeros(size(V_in));
				for slice = 1:size(V_in,3)
					if strcmp(this.gpu,'yes')
						V_out(:,:,slice) = this.apply_2D_gpu(V_in(:,:,slice));
					else
						V_out(:,:,slice) = this.apply_2D(V_in(:,:,slice));						
					end
				end
			
			% 3D, full
			else
				if strcmp(this.gpu,'yes')
					V_out = this.apply_3D_gpu(V_in);
				else
					V_out = this.apply_3D(V_in);
				end
			end
			
		end	
			
	end
		
	%----------------------------------------------------------------------
	methods (Access=protected)
		
		%------------------------------------------------------------------
		function V_out = apply_2D(this, V_in)
			
			r = this.radius;
			w = 2 * r + 1;
			
			% Set Kernel
			K = ones(w) * this.b / (w.^2-1); % edges
			K(r+1,r+1) = 1 - this.b; % center
			
			% output window
			V_out = zeros(size(V_in,1) + w-1, size(V_in,2) + w - 1);

			% blur convolution
			for s = -r:r 
				for t = -r:r 
					V_out(1+r+s:end-r+s, 1+r+t:end-r+t) = V_out(1+r+s:end-r+s, 1+r+t:end-r+t) + K(r+1+s, r+1+t) * V_in;
				end
			end
			
			% shrink output window
			V_out = V_out(1+r:end-r, 1+r:end-r);
			
		end

		%------------------------------------------------------------------
		function V_out = apply_2D_gpu(this, V_in)
			
			vol_geom = astra_create_vol_geom(size(V_in));
			in_id = astra_mex_data2d('create', '-vol', vol_geom, V_in);
			out_id = astra_mex_data2d('create', '-vol', vol_geom, 0);

			cfg = astra_struct('DARTSMOOTHING_CUDA');
			cfg.InDataId = in_id;
			cfg.OutDataId = out_id;
			cfg.option.Intensity = this.b;
			cfg.option.Radius = this.radius;			
			cfg.option.GPUindex = this.gpu_core;
			
			alg_id = astra_mex_algorithm('create',cfg);	
			astra_mex_algorithm('iterate',alg_id,1);
			V_out = astra_mex_data2d('get', out_id);
		
			astra_mex_algorithm('delete', alg_id);
			astra_mex_data2d('delete', in_id, out_id);
			
		end		
		
		%------------------------------------------------------------------
		function I_out = apply_3D(this, I_in)
			
			r = this.radius;
			w = 2 * r + 1;
			
			% Set Kernel
			K = ones(w,w,w) * this.b / (w.^3-1); % edges
			K(r+1,r+1,r+1) = 1 - this.b; % center
			
			% output window
			I_out = zeros(size(I_in,1)+w-1, size(I_in,2)+w-1, size(I_in,3)+w-1);

			% blur convolution
			for s = -r:r 
				for t = -r:r 
					for u = -r:r 
						I_out(1+r+s:end-r+s, 1+r+t:end-r+t, 1+r+u:end-r+u) = I_out(1+r+s:end-r+s, 1+r+t:end-r+t, 1+r+u:end-r+u) + K(r+1+s, r+1+t, r+1+u) * I_in;
					end
				end
			end
			
			% shrink output window
			I_out = I_out(1+r:end-r, 1+r:end-r, 1+r:end-r);
			
		end		
		
		%------------------------------------------------------------------
		function V_out = apply_3D_gpu(this, V_in)

			vol_geom = astra_create_vol_geom(size(V_in,2),size(V_in,1),size(V_in,3));
			data_id = astra_mex_data3d('create', '-vol', vol_geom, V_in);

			cfg = astra_struct('DARTSMOOTHING3D_CUDA');
			cfg.InDataId = data_id;
			cfg.OutDataId = data_id;
			cfg.option.Intensity = this.b;
			cfg.option.Radius = this.radius;
			cfg.option.GPUindex = this.gpu_core;
			
			alg_id = astra_mex_algorithm('create',cfg);	
			astra_mex_algorithm('iterate', alg_id, 1);
			V_out = astra_mex_data3d('get', data_id);
		
			astra_mex_algorithm('delete', alg_id);
			astra_mex_data3d('delete', data_id);
			
		end		
		%------------------------------------------------------------------
		
	end
	
end

