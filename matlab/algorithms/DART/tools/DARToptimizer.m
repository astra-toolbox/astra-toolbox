%--------------------------------------------------------------------------
% This file is part of the ASTRA Toolbox
%
% Copyright: 2010-2014, iMinds-Vision Lab, University of Antwerp
%                 2014, CWI, Amsterdam
% License: Open Source under GPLv3
% Contact: astra@uantwerpen.be
% Website: http://sf.net/projects/astra-toolbox
%--------------------------------------------------------------------------

classdef DARToptimizer < handle
	
	%----------------------------------------------------------------------
	properties (SetAccess=public,GetAccess=public)
		
		% optimization options
		max_evals	= 100;					% SETTING: Maximum number of function evaluations during optimization.
		tolerance	= 0.1;					% SETTING: Minimum tolerance to achieve.
		display		= 'off';				% SETTING: Optimization output. {'off','on','iter'}
		verbose		= 'yes';				% SETTING: verbose? {'yes','no}
		
		metric		= ProjDiffOptimFunc();	% SETTING: Optimization object. Default: ProjDiffOptimFunc.
		
		% DART options
		DART_iterations = 20;				% SETTING: number of DART iterations in each evaluation.
		
		D_base = [];
		
	end
	
	%----------------------------------------------------------------------
	properties (SetAccess=private,GetAccess=public)
		
		stats = Statistics();
		
	end	
	
	%----------------------------------------------------------------------
	methods (Access=public)

		%------------------------------------------------------------------
		% Constructor
		function this = DARToptimizer(D_base)

			this.D_base = D_base;
			
			% statistics
			this.stats = Statistics();
			this.stats.register('params');
			this.stats.register('values');
			this.stats.register('score');
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
			opt_values = fminsearch(@this.optim_func, initial_values, options, params);

			% save to D_base
			for i = 1:numel(params)
				eval(sprintf('this.D_base.%s = %d;',params{i}, opt_values(i)));
			end
			
		end
		%------------------------------------------------------------------
		
	end
		
	%----------------------------------------------------------------------
	methods (Access=protected)		
		
		%------------------------------------------------------------------
		function score = optim_func(this, values, params)

			% copy DART 
			D = this.D_base.deepcopy();

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
			D.iterate(this.DART_iterations);

			% compute score
			score = this.metric.calculate(D, this);
			
			% statistics
			this.stats.add('params',params);
			this.stats.add('values',values);
			this.stats.add('score',score);

			% output
			if strcmp(this.verbose,'yes')
				disp([num2str(values) ': ' num2str(score)]);
			end
			
		end
		%------------------------------------------------------------------

	end
	
end

