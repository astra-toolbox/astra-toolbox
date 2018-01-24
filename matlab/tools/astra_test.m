%--------------------------------------------------------------------------
% Perform a basic test of ASTRA functionality.
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
% This file is part of the ASTRA Toolbox
% 
% Copyright: 2010-2018, imec Vision Lab, University of Antwerp
%            2014-2018, CWI, Amsterdam
% License: Open Source under GPLv3
% Contact: astra@astra-toolbox.com
% Website: http://www.astra-toolbox.com/
%--------------------------------------------------------------------------

if astra_mex('use_cuda')
  astra_test_CUDA;
else
  fprintf('No GPU support available')
  astra_test_noCUDA;
end
