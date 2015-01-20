%--------------------------------------------------------------------------
% This file is part of the ASTRA Toolbox
%
% Copyright: 2010-2014, iMinds-Vision Lab, University of Antwerp
%                 2014, CWI, Amsterdam
% License: Open Source under GPLv3
% Contact: astra@uantwerpen.be
% Website: http://sf.net/projects/astra-toolbox
%--------------------------------------------------------------------------

classdef DARToptimizerBoneStudy < handle
	
	%----------------------------------------------------------------------
	properties (SetAccess=public,GetAccess=public)

		% optimization options
		max_evals = 100;
		tolerance = 0.1;
		display = 'off';
		
		% DART options
		DART_iterations = 50;
		
		D_base = [];
		
	end
	
	%----------------------------------------------------------------------
	properties (SetAccess=private,GetAccess=public)
		
		stats = struct();
		
	end	
	
	%----------------------------------------------------------------------
	methods (Access=public)

		%------------------------------------------------------------------
		% Constructor
		function this = DARToptimizerBoneStudy(D_base)

			this.D_base = D_base;
					
			this.stats.params = {};
			this.stats.values = [];
			this.stats.rmse = [];
			this.stats.f_250 = [];			
			this.stats.f_100 = [];			
			this.stats.w_250 = [];			
			this.stats.w_125 = [];						
			
		end	
	
		%------------------------------------------------------------------
		function opt_values = run(this, params, initial_values)
			
			if nargin < 3
				for i = 1:numel(params)
					initial_values(i) = eval(['this.D_base.' params{i} ';']);
				end
			end
			
			% fminsearch
			options = optimset('display', this.display, 'MaxFunEvals', this.max_evals, 'TolX', this.tolerance);
			opt_values = fminsearch(@optim_func, initial_values, options, this.D_base, params, this);

			% save to D_base
			for i = 1:numel(params)
				eval(sprintf('this.D_base.%s = %d;',params{i}, opt_values(i)));
			end
			
		end
		%------------------------------------------------------------------
	end
	
end
	
%--------------------------------------------------------------------------
function rmse = optim_func(values, D_base, params, Optim)

	% copy DART 
	D = D_base.deepcopy();
	
	% set parameters
	for i = 1:numel(params)
		eval(sprintf('D.%s = %d;',params{i}, values(i)));
		D.output.pre = [D.output.pre num2str(values(i)) '_'];
	end
	
	% evaluate
	if D.initialized == 0
		D.initialize();
	end
	rng('default');
	D.iterate(Optim.DART_iterations);

	% compute rmse
	ROI = load('roi.mat');
	[rmse, f_250, f_100, w_250, w_125] = compute_rmse(D.S, ROI);
	%projection = D.tomography.createForwardProjection(D, D.S);
	%proj_diff = sum((projection(:) - D.base.sinogram(:)).^2);
	
	% save
	Optim.stats.params{end+1} = params;
	Optim.stats.values(end+1,:) = values;
	Optim.stats.rmse(end+1) = rmse;
	Optim.stats.f_250(end+1) = f_250;
	Optim.stats.f_100(end+1) = f_100;
	Optim.stats.w_250(end+1) = w_250;	
	Optim.stats.w_125(end+1) = w_125;		
	
	disp([num2str(values) ': ' num2str(rmse)]);
	
end

	


