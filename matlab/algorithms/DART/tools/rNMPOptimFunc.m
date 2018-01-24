%--------------------------------------------------------------------------
% This file is part of the ASTRA Toolbox
%
% Copyright: 2010-2018, imec Vision Lab, University of Antwerp
%            2014-2018, CWI, Amsterdam
% License: Open Source under GPLv3
% Contact: astra@astra-toolbox.com
% Website: http://www.astra-toolbox.com/
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


