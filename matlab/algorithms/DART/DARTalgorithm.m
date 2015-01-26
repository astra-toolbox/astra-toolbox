%--------------------------------------------------------------------------
% This file is part of the ASTRA Toolbox
%
% Copyright: 2010-2015, iMinds-Vision Lab, University of Antwerp
%            2014-2015, CWI, Amsterdam
% License: Open Source under GPLv3
% Contact: astra@uantwerpen.be
% Website: http://sf.net/projects/astra-toolbox
%--------------------------------------------------------------------------

classdef DARTalgorithm < matlab.mixin.Copyable

	% Algorithm class for Discrete Algebraic Reconstruction Technique (DART).
	
	% todo: reset()
	% todo: fixed random seed
	% todo: initialize from settings (?)

	%----------------------------------------------------------------------
	properties (GetAccess=public, SetAccess=public)
	
		tomography		= IterativeTomography();	% POLICY: Tomography object. 
		segmentation	= SegmentationDefault();	% POLICY: Segmentation object. 
		smoothing		= SmoothingDefault();		% POLICY: Smoothing object.
		masking			= MaskingDefault();			% POLICY: Masking object.
		output			= OutputDefault();			% POLICY: Output object.
		statistics		= StatisticsDefault();		% POLICY: Statistics object.

		base			= struct();					% DATA(set): base structure, should contain: 'sinogram', 'proj_geom', 'phantom' (optional).
		
		memory			= 'no';						% SETTING: reduce memory usage? (disables some features)
	
		implementation  = 'linear';					% SETTING: which type of projector is used ('linear', 'nonlinear')
		t				= 5;						% SETTING: # ARMiterations, each DART iteration.
		t0				= 100;						% SETTING: # ARM iterations at DART initialization.
	end
	%----------------------------------------------------------------------
	properties (GetAccess=public, SetAccess=private)
		V0					= [];			% DATA(get): Initial reconstruction.
		V					= [];			% DATA(get): Reconstruction.
		S					= [];			% DATA(get): Segmentation.
		R					= [];			% DATA(get): Residual projection data.
		Mask				= [];			% DATA(get): Reconstruction Mask.
		stats				= struct();		% Structure containing various statistics.
		iterationcount		= 0;			% Number of performed iterations.	
		start_tic			= 0;		  
		initialized			= 0;			% Is initialized?
	end
	%----------------------------------------------------------------------
	properties (Access=private)		
		adaptparam_name		= {};
		adaptparam_values	= {};
		adaptparam_iters	= {};
	end
	
	%----------------------------------------------------------------------
	methods
		
		%------------------------------------------------------------------
		function this = DARTalgorithm(varargin)
			% Constructor
			% >> D = DARTalgorithm(base);  [base is a matlab struct that
			%                               should contain 'sinogram' and
			%                               'proj_geom']
			% >> D = DARTalgorithm('base_path'); [path to base struct file]
			% >> D = DARTalgorithm(sinogram, proj_geom)
			% 
			narginchk(1, 2)
			if nargin == 1 && ischar(varargin{1})
				this.base = load(varargin{1});
			elseif nargin == 1 && isstruct(varargin{1})
				this.base = varargin{1};
			elseif nargin == 2
				this.base = struct();
				this.base.sinogram = varargin{1};
				this.base.proj_geom = varargin{2};
			else
				error('invalid arguments')
			end
		end	
	
		
		%------------------------------------------------------------------
		function D = deepcopy(this)
			% Create a deep copy of this object.
			% >> D2 = D.deepcopy();
			D = copy(this);
			props = properties(this);
			for i = 1:length(props)
				if isa(this.(props{i}), 'handle')
					D.(props{i}) = copy(this.(props{i}));
				end
			end			
		end
		
		%------------------------------------------------------------------
		function this = initialize(this)
			% Initializes this object.
			% >> D.initialize();
			
			% Initialize tomography part
			if ~this.tomography.initialized
				this.tomography.sinogram = this.base.sinogram;
				this.tomography.proj_geom = this.base.proj_geom;	
				this.tomography.initialize();
			end

			% Create an Initial Reconstruction
			if isfield(this.base, 'V0')
				this.V0 = this.base.V0;
			else
				this.output.pre_initial_iteration(this);
				this.V0 = this.tomography.reconstruct2(this.base.sinogram, [], this.t0);
				this.output.post_initial_iteration(this);
			end
			this.V = this.V0;
			if strcmp(this.memory,'yes')
				this.base.V0 = [];
				this.V0 = [];
			end
			this.initialized = 1;
		end		
		
		%------------------------------------------------------------------
		% iterate
		function this = iterate(this, iters)
			% Perform several iterations of the DART algorithm.
			% >> D.iterate(iterations);		
			if strcmp(this.implementation,'linear')
				this.iterate_linear(iters);
			elseif strcmp(this.implementation,'nonlinear')
				this.iterate_nonlinear(iters);
			end
		end
		
		
		%------------------------------------------------------------------
		% iterate - linear projector implementation
		function this = iterate_linear(this, iters)
	
			this.start_tic = tic;
			
			for iteration = 1:iters
				this.iterationcount = this.iterationcount + 1;				

				% initial output
				this.output.pre_iteration(this);

				% update adaptive parameters
				this.update_adaptiveparameter(this.iterationcount);

				% segmentation
				this.segmentation.estimate_grey_levels(this, this.V);
				this.S = this.segmentation.apply(this, this.V);

				% select update and fixed pixels
				this.Mask = this.masking.apply(this, this.S);
				this.V = (this.V .* this.Mask) + (this.S .* (1 - this.Mask));
				F = this.V;
				F(this.Mask == 1) = 0;

				% compute residual projection difference
				this.R = this.base.sinogram - this.tomography.project(F);
				
				% ART update part
				this.V = this.tomography.reconstruct2_mask(this.R, this.V, this.Mask, this.t);

				% blur
				this.V = this.smoothing.apply(this, this.V);

				%calculate statistics
				this.stats = this.statistics.apply(this);
			
				% output
				this.output.post_iteration(this);
			end
		
		end				

		%------------------------------------------------------------------
		% iterate - nonlinear projector implementation
		function this = iterate_nonlinear(this, iters)
			
			this.start_tic = tic;
			
			for iteration = 1:iters
				this.iterationcount = this.iterationcount + 1;				
		
				% Output
				this.output.pre_iteration(this);

				% update adaptive parameters
				this.update_adaptiveparameter(this.iterationcount)

				% Segmentation
				this.segmentation.estimate_grey_levels(this, this.V);
				this.S = this.segmentation.apply(this, this.V);

				% Select Update and Fixed Pixels
				this.Mask = this.masking.apply(this, this.S);
				this.V = (this.V .* this.Mask) + (this.S .* (1 - this.Mask));

				% ART update part
				this.V = this.tomography.reconstruct2_mask(this.base.sinogram, this.V, this.Mask, this.t);
				
				% blur
				this.V = this.smoothing.apply(this, this.V);
				
				% calculate statistics
				this.stats = this.statistics.apply(this);

				% output
				this.output.post_iteration(this);
			
			end
		
		end						
		
		
		%------------------------------------------------------------------
		% get data		
		function data = getdata(this, string)
			if numel(this.(string)) == 1
				data = astra_mex_data2d('get',this.(string));
			else
				data = this.(string);
			end
		end
		
		%------------------------------------------------------------------
		% add adaptive parameter
		function this = adaptiveparameter(this, name, values, iterations)
			this.adaptparam_name{end+1} = name;
			this.adaptparam_values{end+1} = values;
			this.adaptparam_iters{end+1} = iterations;
		end
		
		%------------------------------------------------------------------
		% update adaptive parameter
		function this = update_adaptiveparameter(this, iteration)
			for i = 1:numel(this.adaptparam_name)
				for j = 1:numel(this.adaptparam_iters{i})
					if iteration == this.adaptparam_iters{i}(j)
						new_value = this.adaptparam_values{i}(j);
						eval(['this.' this.adaptparam_name{i} ' = ' num2str(new_value) ';']);
					end						
				end
			end		
		end
			
		%------------------------------------------------------------------
		function settings = getsettings(this)
			% Returns a structure containing all settings of this object.
			% >> settings = tomography.getsettings();		
			
			settings.tomography = this.tomography.getsettings();
			settings.smoothing = this.smoothing.getsettings();
			settings.masking = this.masking.getsettings();
			settings.segmentation = this.segmentation.getsettings();
		end		
		%------------------------------------------------------------------
		
	end % methods
		
end % class


