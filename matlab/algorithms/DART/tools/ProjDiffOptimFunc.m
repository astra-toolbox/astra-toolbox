%--------------------------------------------------------------------------
% This file is part of the ASTRA Toolbox
%
% Copyright: 2010-2014, iMinds-Vision Lab, University of Antwerp
%                 2014, CWI, Amsterdam
% License: Open Source under GPLv3
% Contact: astra@uantwerpen.be
% Website: http://sf.net/projects/astra-toolbox
%--------------------------------------------------------------------------

classdef ProjDiffOptimFunc < handle

	%----------------------------------------------------------------------
	properties (SetAccess=private, GetAccess=public)

	end	
	
	%----------------------------------------------------------------------
	methods (Access=public)	
		
		function proj_diff = calculate(~, D, ~)
			projection = D.tomography.createForwardProjection(D, D.S);
			proj_diff = sum((projection(:) - D.base.sinogram(:)).^2);
		end
		
	end
	%----------------------------------------------------------------------
	
end
