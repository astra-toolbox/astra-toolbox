% This file is part of the
% All Scale Tomographic Reconstruction Antwerp Toolbox ("ASTRA-Toolbox")
%
% Copyright: iMinds-Vision Lab, University of Antwerp
% License: Open Source under GPLv3
% Contact: mailto:astra@ua.ac.be
% Website: http://astra.ua.ac.be
%
% Author of this DART Algorithm: Wim van Aarle


classdef DARTalgorithm < matlab.mixin.Copyable

	% Algorithm class for Discrete Algebraic Reconstruction Technique (DART).
	
	% todo: reset()
	% todo: fixed random seed
	% todo: initialize from settings (?)

	%----------------------------------------------------------------------
	properties (GetAccess=public, SetAccess=public)
	
		tomography		= TomographyDefault();		% POLICY: Tomography object. 
		segmentation	= SegmentationDefault();	% POLICY: Segmentation object. 
		smoothing		= SmoothingDefault();		% POLICY: Smoothing object.
		masking			= MaskingDefault();			% POLICY: Masking object.
		output			= OutputDefault();			% POLICY: Output object.
		statistics		= StatisticsDefault();		% POLICY: Statistics object.

		base			= struct();					% DATA(set): base structure, should contain: 'sinogram', 'proj_geom', 'phantom' (optional).
		
		memory			= 'yes';					% SETTING: reduce memory usage? (disables some features)
		
		testdata = struct();
		
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
		function this = DARTalgorithm(base)
			% Constructor
			% >> D = DARTalgorithm(base); base is a matlab struct (or the path towards one) 
			%					          that should contain 'sinogram', 'proj_geom'.

			if ischar(base)
				this.base = load(base);
			else
				this.base = base;
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
			this.tomography.initialize(this);

			% Create an Initial Reconstruction
			if isfield(this.base, 'V0')
				this.V0 = this.base.V0;
			else
				this.output.pre_initial_iteration(this);
				this.V0 = this.tomography.createInitialReconstruction(this, this.base.sinogram);
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
			
			this.start_tic = tic;
			
			for iteration = 1:iters
				
				this.iterationcount = this.iterationcount + 1;				
		
				%----------------------------------------------------------
				% Initial Output
				this.output.pre_iteration(this);

				%----------------------------------------------------------
				% Update Adaptive Parameters
				for i = 1:numel(this.adaptparam_name)
					
					for j = 1:numel(this.adaptparam_iters{i})
						if this.iterationcount == this.adaptparam_iters{i}(j)
							new_value = this.adaptparam_values{i}(j);
							eval(['this.' this.adaptparam_name{i} ' = ' num2str(new_value) ';']);
							disp(['value ' this.adaptparam_name{i} ' changed to ' num2str(new_value) ]);
						end						
					end
				
				end

				%----------------------------------------------------------
				% Segmentation
				this.segmentation.estimate_grey_levels(this, this.V);
				this.S = this.segmentation.apply(this, this.V);

				%----------------------------------------------------------
				% Select Update and Fixed Pixels
				this.Mask = this.masking.apply(this, this.S);

				this.V = (this.V .* this.Mask) + (this.S .* (1 - this.Mask));
				%this.V(this.Mask == 0) = this.S(this.Mask == 0); 

				F = this.V;
				F(this.Mask == 1) = 0;

				%----------------------------------------------------------
				% Create Residual Projection Difference
				%this.testdata.F{iteration} = F;
				this.R = this.base.sinogram - this.tomography.createForwardProjection(this, F);
				%this.testdata.R{iteration} = this.R;
				
				%----------------------------------------------------------
				% ART Loose Part
				%this.testdata.V1{iteration} = this.V;
				%this.testdata.Mask{iteration} = this.Mask;
				
				%X = zeros(size(this.V));
				%Y = this.tomography.createReconstruction(this, this.R, X, this.Mask);
				%this.V(this.Mask) = Y(this.Mask);
				this.V = this.tomography.createReconstruction(this, this.R, this.V, this.Mask);
				
				%this.testdata.V2{iteration} = this.V;
				
				%----------------------------------------------------------
				% Blur
				this.V = this.smoothing.apply(this, this.V);
				%this.testdata.V3{iteration} = this.V;
				
				%----------------------------------------------------------
				% Calculate Statistics
				this.stats = this.statistics.apply(this);

				%----------------------------------------------------------
				% Output
				this.output.post_iteration(this);
			
			end % end iteration loop
			
			%test = this.testdata;
			%save('testdata.mat','test');
		
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


