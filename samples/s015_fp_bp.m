% -----------------------------------------------------------------------
% This file is part of the ASTRA Toolbox
% 
% Copyright: 2010-2015, iMinds-Vision Lab, University of Antwerp
%            2014-2015, CWI, Amsterdam
% License: Open Source under GPLv3
% Contact: astra@uantwerpen.be
% Website: http://sf.net/projects/astra-toolbox
% -----------------------------------------------------------------------


% This example demonstrates using the FP and BP primitives with Matlab's lsqr
% solver. Calls to FP (astra_create_sino_cuda) and
% BP (astra_create_backprojection_cuda) are wrapped in a function astra_wrap,
% and a handle to this function is passed to lsqr.

% Because in this case the inputs/outputs of FP and BP have to be vectors
% instead of images (matrices), the calls require reshaping to and from vectors.

function s015_fp_bp


% FP/BP wrapper function
function Y = astra_wrap(X,T)
  if strcmp(T, 'notransp')
    % X is passed as a vector. Reshape it into an image.
    [sid, s] = astra_create_sino_cuda(reshape(X,[vol_geom.GridRowCount vol_geom.GridColCount])', proj_geom, vol_geom);
    astra_mex_data2d('delete', sid);
    % now s is the sinogram. Reshape it back into a vector
    Y = reshape(s',[prod(size(s)) 1]);
  else
    % X is passed as a vector. Reshape it into a sinogram.
    v = astra_create_backprojection_cuda(reshape(X, [proj_geom.DetectorCount size(proj_geom.ProjectionAngles,2)])', proj_geom, vol_geom);
    % now v is the resulting volume. Reshape it back into a vector
    Y = reshape(v',[prod(size(v)) 1]);
  end
end


vol_geom = astra_create_vol_geom(256, 256);
proj_geom = astra_create_proj_geom('parallel', 1.0, 384, linspace2(0,pi,180));

% Create a 256x256 phantom image using matlab's built-in phantom() function
P = phantom(256);

% Create a sinogram using the GPU.
[sinogram_id, sinogram] = astra_create_sino_gpu(P, proj_geom, vol_geom);

% Reshape the sinogram into a vector
b = reshape(sinogram',[prod(size(sinogram)) 1]);

% Call Matlab's lsqr with ASTRA FP and BP
Y = lsqr(@astra_wrap,b,1e-4,25);

% Reshape the result into an image
Y = reshape(Y,[vol_geom.GridRowCount vol_geom.GridColCount])';
imshow(Y,[]);


astra_mex_data2d('delete', sinogram_id);

end
