% This file is part of the
% All Scale Tomographic Reconstruction Antwerp Toolbox ("ASTRA-Toolbox")
%
% Copyright: iMinds-Vision Lab, University of Antwerp
% License: Open Source under GPLv3
% Contact: mailto:astra@ua.ac.be
% Website: http://astra.ua.ac.be
%
% Author of this DART Algorithm: Wim van Aarle


classdef SegmentationDefault < matlab.mixin.Copyable
		
	% Default policy class for segmentation for DART.
	
	%----------------------------------------------------------------------
	properties (Access=public)
		rho			= [];		% SETTING: Grey levels.
		tau			= [];		% SETTING: Threshold values.
	end
	
	%----------------------------------------------------------------------
	methods (Access=public)
		
		%------------------------------------------------------------------
		function settings = getsettings(this)
			% Returns a structure containing all settings of this object.
			% >> settings = DART.segmentation.getsettings();			
			settings.rho	= this.rho;
			settings.tau	= this.tau;
		end
	
		%------------------------------------------------------------------
		function this = estimate_grey_levels(this, ~, ~)
			% Estimates grey levels
			% >> DART.segmentation.estimate_grey_levels();				
		end
		
		%------------------------------------------------------------------
		function V_out = apply(this, ~, V_in)
			% Applies segmentation.
			% >> V_out = DART.segmentation.apply(DART, V_in);	
			
			V_out = ones(size(V_in)) * this.rho(1);
			for n = 2:length(this.rho)
				V_out(this.tau(n-1) < V_in) = this.rho(n);
			end
			
		end
		%------------------------------------------------------------------
			
	end
	
end

