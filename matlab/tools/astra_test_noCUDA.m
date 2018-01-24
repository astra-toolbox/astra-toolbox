%--------------------------------------------------------------------------
% Perform a basic test of ASTRA CPU functionality.
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

fprintf('Testing basic CPU 2D functionality...')

vg = astra_create_vol_geom(2, 32);
pg = astra_create_proj_geom('parallel', 1, 32, [0]);
proj_id = astra_create_projector('line', pg, vg);

vol = rand(2, 32);
[sino_id, sino] = astra_create_sino(vol, proj_id);
astra_mex_data2d('delete', sino_id);
astra_mex_projector('delete', proj_id);

err = max(abs(sino - sum(vol)));

if err < 1e-6
  disp('Ok')
else
  disp('Error')
end
