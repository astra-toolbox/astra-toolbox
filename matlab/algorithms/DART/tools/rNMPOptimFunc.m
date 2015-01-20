%--------------------------------------------------------------------------
% This file is part of the ASTRA Toolbox
%
% Copyright: 2010-2014, iMinds-Vision Lab, University of Antwerp
%                 2014, CWI, Amsterdam
% License: Open Source under GPLv3
% Contact: astra@uantwerpen.be
% Website: http://sf.net/projects/astra-toolbox
%--------------------------------------------------------------------------

classdef rNMPOptimFunc < handle

	%----------------------------------------------------------------------
	properties (SetAccess=private, GetAccess=public)

	end	
	
	%----------------------------------------------------------------------
	methods (Access=public)	
		
		function rnmp = calculate(~, D, ~)
			if isfield(D.stats,'rnmp');
				rnmp = D.stats.rnmp;
			else
				rnmp = compute_rnmp(D.base.phantom, D.S);
			end
		end
		
	end
	%----------------------------------------------------------------------

	
end


