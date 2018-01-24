%--------------------------------------------------------------------------
% This file is part of the ASTRA Toolbox
%
% Copyright: 2010-2018, imec Vision Lab, University of Antwerp
%            2014-2018, CWI, Amsterdam
% License: Open Source under GPLv3
% Contact: astra@astra-toolbox.com
% Website: http://www.astra-toolbox.com/
%--------------------------------------------------------------------------

classdef ProjDiffOptimFunc < handle

	%----------------------------------------------------------------------
	properties (SetAccess=private, GetAccess=public)

	end	
	
	%----------------------------------------------------------------------
	methods (Access=public)	
		
		function proj_diff = calculate(~, D, ~)
			projection = D.tomography.project(D.S);
			proj_diff = sum((projection(:) - D.base.sinogram(:)).^2);
		end
		
	end
	%----------------------------------------------------------------------
	
end
