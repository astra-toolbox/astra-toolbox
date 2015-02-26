classdef IterativeTomography3D < matlab.mixin.Copyable

	% Algorithm class for 3D Iterative Tomography.

	%----------------------------------------------------------------------
	properties (SetAccess=public, GetAccess=public)				
		superresolution		= 1;				% SETTING: Volume upsampling factor.
		proj_type			= 'linear';			% SETTING: Projector type, only when gpu='no'.
		method				= 'SIRT3D_CUDA';	% SETTING: Iterative method (see ASTRA toolbox documentation).
		gpu					= 'yes';			% SETTING: Use gpu? {'yes', 'no'}
		gpu_core			= 0;				% SETTING: Which gpu core to use? Only when gpu='yes'.
		inner_circle		= 'yes';			% SETTING: Do roi only? {'yes', 'no'}
		image_size			= [];				% SETTING: Overwrite default reconstruction size.  Only if no vol_geom is specified.
		use_minc			= 'no';				% SETTING: Use minimum constraint. {'no', 'yes'}
		maxc 				= +Inf;				% SETTING: Maximum constraint. +Inf means off.
	end
	%----------------------------------------------------------------------
	properties (SetAccess=public, GetAccess=public)	
		sinogram			= [];				% DATA: Projection data.
		proj_geom			= [];				% DATA: Projection geometry.
		V					= [];				% DATA: Volume data.  Also used to set initial estimate (optional).
		vol_geom			= [];				% DATA: Volume geometry.
	end
	%----------------------------------------------------------------------
	properties (SetAccess=private, GetAccess=public)
		initialized     	= 0;				% Is this object initialized?
	end
	%----------------------------------------------------------------------
	properties (SetAccess=protected, GetAccess=protected)
		proj_geom_sr		= [];				% PROTECTED: geometry of sinogram (with super-resolution)
        proj_id             = [];				% PROTECTED: astra id of projector (when gpu='no')
        proj_id_sr          = [];				% PROTECTED: astra id of super-resolution projector (when gpu='no')	
		cfg_base			= struct();			% PROTECTED: base configuration structure for the reconstruction algorithm.
    end
	%----------------------------------------------------------------------

	methods (Access=public)
		
		%------------------------------------------------------------------
		function this = IterativeTomography3D(varargin)
			% Constructor
			% >> tomography = IterativeTomography3D(proj_geom);
			% >> tomography = IterativeTomography3D(proj_geom, vol_geom);
			
			% Input: IterativeTomography(proj_geom)
			if nargin == 1
				this.proj_geom = varargin{1};
			
			% Input: IterativeTomography(proj_geom, vol_geom)
			elseif nargin == 2
				this.proj_geom = varargin{1};
				this.vol_geom =	varargin{2};
			end
		end

		%------------------------------------------------------------------
		function delete(this)
			% Destructor
			% >> clear tomography;
			if strcmp(this.gpu,'no') && numel(this.proj_id) > 0
				astra_mex_projector('delete', this.proj_id, this.proj_id_sr);
			end
		end
			
		%------------------------------------------------------------------
		function settings = getsettings(this)
			% Returns a structure containing all settings of this object.
			% >> settings = tomography.getsettings();
			settings.superresolution	= this.superresolution;
			settings.proj_type			= this.proj_type;
			settings.method				= this.method;
			settings.gpu				= this.gpu;
			settings.gpu_core			= this.gpu_core;
			settings.inner_circle		= this.inner_circle;
			settings.image_size			= this.image_size;	
			settings.use_minc			= this.use_minc;
            settings.maxc               = this.maxc;
		end		
		
		%------------------------------------------------------------------
		function ok = initialize(this)
			% Initialize this object.  Returns 1 if succesful.
			% >> tomography.initialize();
			
% 			% create projection geometry with super-resolution
% 			this.proj_geom_sr = astra_geom_superresolution(this.proj_geom, this.superresolution);		
			
			% if no volume geometry is specified by the user: create volume geometry
			if numel(this.vol_geom) == 0
				if numel(this.image_size) < 2
					this.image_size(1) = this.proj_geom.DetectorRowCount;
					this.image_size(2) = this.proj_geom.DetectorColCount;
				end
				this.vol_geom = astra_create_vol_geom(this.proj_geom.DetectorColCount, this.proj_geom.DetectorColCount, this.proj_geom.DetectorRowCount);
			else
				this.image_size(1) = this.vol_geom.GridRowCount;
				this.image_size(2) = this.vol_geom.GridColCount;
			end
			
			% create projector
			if strcmp(this.gpu, 'no')
				this.proj_id = astra_create_projector(this.proj_type, this.proj_geom, this.vol_geom);
				this.proj_id_sr = astra_create_projector(this.proj_type, this.proj_geom_sr, this.vol_geom);			
			end
			
			% create reconstruction configuration
			this.cfg_base = astra_struct(upper(this.method));
			if strcmp(this.gpu,'no')
				this.cfg_base.ProjectorId = this.proj_id;
				this.cfg_base.ProjectionGeometry = this.proj_geom;
				this.cfg_base.ReconstructionGeometry = this.vol_geom;
				this.cfg_base.option.ProjectionOrder = 'random';
			end
			this.cfg_base.option.DetectorSuperSampling = this.superresolution;
			if strcmp(this.gpu,'yes')
				this.cfg_base.option.GPUindex = this.gpu_core;
			end
			this.cfg_base.option.UseMinConstraint = this.use_minc;
            if ~isinf(this.maxc)
              this.cfg_base.option.UseMaxConstraint = 'yes';
              this.cfg_base.option.MaxConstraintValue = this.maxc;
            end
		
			this.initialized = 1;
			ok = this.initialized;
		end
		
		%------------------------------------------------------------------
		function projections = project(this, volume)
			% Compute forward projection.  
			% >> projections = tomography.project(volume);
			
			if ~this.initialized 
				this.initialize();
			end

			% project			
			projections = this.project_c(volume);
		end

		%------------------------------------------------------------------
		function reconstruction = reconstruct(this, varargin)
			% Compute reconstruction.
			% Uses tomography.sinogram
			% Initial solution (if available) should be stored in tomography.V  
			% >> reconstruction = tomography.reconstruct(projections, iterations);
			% >> reconstruction = tomography.reconstruct(projections, volume0, iterations);
			
			if ~this.initialized 
				this.initialize();
			end			
			
			if numel(varargin) == 2
				reconstruction = this.reconstruct_c(varargin{1}, [], [], varargin{2});
			elseif numel(varargin) == 3
				reconstruction = this.reconstruct_c(varargin{1}, varargin{2}, [], varargin{3});
			else
				error('invalid parameter list')
			end
				
			if strcmp(this.inner_circle,'yes')
				reconstruction = this.selectROI(reconstruction);
			end
		end
		
		%------------------------------------------------------------------
		function reconstruction = reconstruct_mask(this, varargin)
			% Compute reconstruction with mask.
			% Uses tomography.sinogram  
			% Initial solution (if available) should be stored in tomography.V  
			% >> reconstruction = tomography.reconstructMask(projections, mask, iterations);
			% >> reconstruction = tomography.reconstructMask(projections, volume0, mask, iterations);
			
			if ~this.initialized 
				this.initialize();
			end
			
			if numel(varargin) == 3
				reconstruction = this.reconstruct_c(varargin{1}, [], varargin{2}, varargin{3});
			elseif numel(varargin) == 4
				reconstruction = this.reconstruct_c(varargin{1}, varargin{2}, varargin{3}, varargin{4});
			else
				error('invalid parameter list')
			end
		
			if strcmp(this.inner_circle,'yes')
				reconstruction = this.selectROI(reconstruction);
			end

		end
		%------------------------------------------------------------------
		
	end
	
	%----------------------------------------------------------------------
	methods (Access = protected)
	
		%------------------------------------------------------------------
		% Protected function: create FP
		function sinogram = project_c(this, volume)
			
			if this.initialized == 0
				error('IterativeTomography not initialized');
			end
            
			% data is stored in astra memory
			if numel(volume) == 1
				
				if strcmp(this.gpu, 'yes')
					sinogram_tmp = astra_create_sino_cuda(volume, this.proj_geom_sr, this.vol_geom, this.gpu_core);
				else
					sinogram_tmp = astra_create_sino(volume, this.proj_id);
				end
				
				% sinogram downsampling
				if this.superresolution > 1
					sinogram_data = astra_mex_data2d('get', sinogram_tmp);
					astra_mex_data2d('delete', sinogram_tmp);
					sinogram_data = downsample_sinogram(sinogram_data, this.superresolution);	
					sinogram = astra_mex_data2d('create', 'sino', this.proj_geom, sinogram_data);
				else
					sinogram = sinogram_tmp;
				end
				
			% data is stored in matlab memory	
			else
				[tmp_id, sinogram] = astra_create_sino3d_cuda(volume, this.proj_geom, this.vol_geom);
				astra_mex_data3d('delete', tmp_id);
			end
		end		

		%------------------------------------------------------------------
		% Protected function: reconstruct
		function V = reconstruct_c(this, sinogram, V0, mask, iterations)
			
			if this.initialized == 0
				error('IterativeTomography not initialized');
			end			
			
			% data is stored in astra memory
			if numel(sinogram) == 1
				V = this.reconstruct_c_astra(sinogram, V0, mask, iterations);
			
			% data is stored in matlab memory
			else
				V = this.reconstruct_c_matlab(sinogram, V0, mask, iterations);
			end
		end
		
		%------------------------------------------------------------------
		% Protected function: reconstruct (data in matlab)	
		function V = reconstruct_c_matlab(this, sinogram, V0, mask, iterations)
			
			if this.initialized == 0
				error('IterativeTomography not initialized');
			end						
			
			% parse method
			method2 = upper(this.method);
			if strcmp(method2, 'SART') || strcmp(method2, 'SART_CUDA')
				iterations = iterations * size(sinogram,1);
			elseif strcmp(method2, 'ART')
				iterations = iterations * numel(sinogram);
			end
			
			% create data objects
% 			V = zeros(this.vol_geom.GridRowCount, this.vol_geom.GridColCount, size(sinogram,3));
			reconstruction_id = astra_mex_data3d('create', '-vol', this.vol_geom);
			sinogram_id = astra_mex_data3d('create', '-proj3d', this.proj_geom);
			if numel(mask) > 0
				mask_id = astra_mex_data3d('create', '-vol', this.vol_geom);
			end
			
			% algorithm configuration
			cfg = this.cfg_base;
			cfg.ProjectionDataId = sinogram_id;
			cfg.ReconstructionDataId = reconstruction_id;
			if numel(mask) > 0
				cfg.option.ReconstructionMaskId = mask_id;
			end
			alg_id = astra_mex_algorithm('create', cfg);
			
% 			% loop slices
% 			for slice = 1:size(sinogram,3)
				
				% fetch slice of initial reconstruction
				if numel(V0) > 0
					astra_mex_data3d('store', reconstruction_id, V0);
				else
					astra_mex_data3d('store', reconstruction_id, 0);
				end
				
				% fetch slice of sinogram
				astra_mex_data3d('store', sinogram_id, sinogram);
				
				% fecth slice of mask
				if numel(mask) > 0
					astra_mex_data3d('store', mask_id, mask);    
				end
				
				% iterate
				astra_mex_algorithm('iterate', alg_id, iterations);
				
				% fetch data
				V = astra_mex_data3d('get', reconstruction_id);
				
%			end
			
			% correct attenuation factors for super-resolution
			if this.superresolution > 1 && strcmp(this.gpu,'yes')
				if strcmp(this.proj_geom.type,'fanflat_vec') || strcmp(this.proj_geom.type,'fanflat')
					if numel(mask) > 0
						V(mask > 0) = V(mask > 0) ./ this.superresolution;
					else
						V = V ./ this.superresolution;
					end
				end		
			end
			
			% garbage collection
			astra_mex_algorithm('delete', alg_id);
			astra_mex_data3d('delete', sinogram_id, reconstruction_id);
			if numel(mask) > 0
				astra_mex_data3d('delete', mask_id);
			end	
			
		end
		
		%------------------------------------------------------------------
		% Protected function: reconstruct (data in astra)	
		function V = reconstruct_c_astra(this, sinogram, V0, mask, iterations)
			
			if this.initialized == 0
				error('IterativeTomography not initialized');
			end			
		
			if numel(V0) > 1 || numel(mask) > 1 || numel(sinogram) > 1
				error('Not all required data is stored in the astra memory');
			end			
			
			if numel(V0) == 0
				V0 = astra_mex_data2d('create', '-vol', this.vol_geom, 0);
			end
			
			% parse method
			method2 = upper(this.method);
			if strcmp(method2, 'SART') || strcmp(method2, 'SART_CUDA')
				iterations = iterations * astra_geom_size(this.proj_geom, 1);
			elseif strcmp(method2, 'ART')
				s = astra_geom_size(this.proj_geom);
				iterations = iterations * s(1) * s(2);
			end	
			
			% algorithm configuration
			cfg = this.cfg_base;
			cfg.ProjectionDataId = sinogram;
			cfg.ReconstructionDataId = V0;
			if numel(mask) > 0
				cfg.option.ReconstructionMaskId = mask;
			end
			alg_id = astra_mex_algorithm('create', cfg);				
			
			% iterate
			astra_mex_algorithm('iterate', alg_id, iterations);	
			
			% fetch data
			V = V0;
			
			% correct attenuation factors for super-resolution
			if this.superresolution > 1
				if strcmp(this.proj_geom.type,'fanflat_vec') || strcmp(this.proj_geom.type,'fanflat')
					if numel(mask) > 0
						astra_data_op_masked('$1./s1', [V V], [this.superresolution this.superresolution], mask, this.gpu_core);
					else
						astra_data_op('$1./s1', [V V], [this.superresolution this.superresolution], this.gpu_core);
					end
				end	
			end
			
			% garbage collection
			astra_mex_algorithm('delete', alg_id);
			
		end
		
		%------------------------------------------------------------------
		function V_out = selectROI(~, V_in)
			
			if numel(V_in) == 1
				cfg = astra_struct('RoiSelect_CUDA');
				cfg.DataId = V_in;
				alg_id = astra_mex_algorithm('create',cfg);
				astra_mex_algorithm('run', alg_id);
				astra_mex_algorithm('delete', alg_id);
				V_out = V_in;
			else
				V_out = ROIselectfull(V_in, min([size(V_in,1), size(V_in,2)]));
			end
			
		end
		%------------------------------------------------------------------
		
	end
		
end

