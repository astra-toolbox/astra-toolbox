% -----------------------------------------------------------------------
% This file is part of the ASTRA Toolbox
% 
% Copyright: 2010-2015, iMinds-Vision Lab, University of Antwerp
%            2014-2015, CWI, Amsterdam
% License: Open Source under GPLv3
% Contact: astra@uantwerpen.be
% Website: http://sf.net/projects/astra-toolbox
% -----------------------------------------------------------------------

% Create a basic 256x256 square volume geometry
vol_geom = astra_create_vol_geom(256, 256);

% Create a parallel beam geometry with 180 angles between 0 and pi, and
% 384 detector pixels of width 1.
% For more details on available geometries, see the online help of the
% function astra_create_proj_geom .
proj_geom = astra_create_proj_geom('parallel', 1.0, 384, linspace2(0,pi,180));

% Create a 256x256 phantom image using matlab's built-in phantom() function
P = phantom(256);

% Create a sinogram using the GPU.
% Note that the first time the GPU is accessed, there may be a delay
% of up to 10 seconds for initialization.
[sinogram_id, sinogram] = astra_create_sino_gpu(P, proj_geom, vol_geom);

figure(1); imshow(P, []);
figure(2); imshow(sinogram, []);


% Free memory
astra_mex_data2d('delete', sinogram_id);
