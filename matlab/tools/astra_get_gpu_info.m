function astra_set_gpu_index(index)

%--------------------------------------------------------------------------
% Set the index of the GPU to use
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
% This file is part of the ASTRA Toolbox
% 
% Copyright: 2010-2022, imec Vision Lab, University of Antwerp
%            2014-2022, CWI, Amsterdam
% License: Open Source under GPLv3
% Contact: astra@astra-toolbox.com
% Website: http://www.astra-toolbox.com/
%--------------------------------------------------------------------------

if nargin < 1
    astra_mex('get_gpu_info');
else
    astra_mex('get_gpu_info', index);
end
