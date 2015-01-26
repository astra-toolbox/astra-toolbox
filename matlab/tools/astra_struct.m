function res = astra_struct(type)

%------------------------------------------------------------------------
% res = astra_struct(type)
%
% Create an ASTRA struct
%
% type: type of the struct to be generated.
% res: the generated matlab struct.
%------------------------------------------------------------------------
%------------------------------------------------------------------------
% This file is part of the ASTRA Toolbox
% 
% Copyright: 2010-2015, iMinds-Vision Lab, University of Antwerp
%            2014-2015, CWI, Amsterdam
% License: Open Source under GPLv3
% Contact: astra@uantwerpen.be
% Website: http://sf.net/projects/astra-toolbox
%------------------------------------------------------------------------
% $Id$
res = struct();
res.options = struct();


if nargin >= 1
	% For backward compatibility, transparently accept SIRT_CUDA2
	% for SIRT_CUDA, and FP_CUDA2 for FP_CUDA.
	if strcmp(type, 'SIRT_CUDA2')
		type = 'SIRT_CUDA';
		warning('SIRT_CUDA2 has been deprecated. Use SIRT_CUDA instead.');
	end
	if strcmp(type, 'FP_CUDA2')
		type = 'FP_CUDA';
		warning('FP_CUDA2 has been deprecated. Use FP_CUDA instead.');
	end
	res.type = type;
end
