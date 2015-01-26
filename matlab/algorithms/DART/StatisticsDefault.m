%--------------------------------------------------------------------------
% This file is part of the ASTRA Toolbox
%
% Copyright: 2010-2015, iMinds-Vision Lab, University of Antwerp
%            2014-2015, CWI, Amsterdam
% License: Open Source under GPLv3
% Contact: astra@uantwerpen.be
% Website: http://sf.net/projects/astra-toolbox
%--------------------------------------------------------------------------

classdef StatisticsDefault < matlab.mixin.Copyable
	
	% Default policy class for statistics for DART.	
	
	properties (Access=public)
		pixel_error		= 'no';		% SETTING: Store pixel error? {'yes','no'}
		proj_diff		= 'no';		% SETTING: Store projection difference? {'yes','no'}
		timing			= 'no';		% SETTING: Store timings? {'yes','no'}
	end

	
	methods (Access=public)
		
		%------------------------------------------------------------------
		function stats = apply(this, DART)
			% Applies statistics.
			% >> stats = DART.statistics.apply(DART);				
			
			stats = DART.stats;
			
			% timing
			if strcmp(this.timing, 'yes')
				stats.timing(DART.iterationcount) = toc(DART.start_tic);
			end
			
			% pixel error
			if strcmp(this.pixel_error, 'yes') && isfield(DART.base,'phantom')
				[stats.rnmp, stats.nmp] = compute_rnmp(DART.base.phantom, DART.S);
				stats.rnmp_hist(DART.iterationcount) = stats.rnmp;
				stats.nmp_hist(DART.iterationcount) = stats.nmp;
			end
			
			% projection difference
			if strcmp(this.proj_diff, 'yes') 
				new_sino = DART.tomography.createForwardProjection(DART, DART.S);
				stats.proj_diff = sum((new_sino(:) - DART.base.sinogram(:)) .^2 ) ./ (sum(DART.base.sinogram(:)) );
				stats.proj_diff_hist(DART.iterationcount) = stats.proj_diff;
			end
			
		end
		
		%------------------------------------------------------------------
		function s = tostring(~, stats)
			% To string.
			% >> stats = DART.statistics.apply(stats);				
			
			s = '';
			if isfield(stats, 'nmp')
				s = sprintf('%s [%d]', s, stats.nmp); 
			end
			if isfield(stats, 'proj_diff')
				s = sprintf('%s {%0.2d}', s, stats.proj_diff); 
			end			
			
		end
		%------------------------------------------------------------------
		
	end
	
end

