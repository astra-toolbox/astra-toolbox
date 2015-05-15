% -----------------------------------------------------------------------
% This file is part of the ASTRA Toolbox
% 
% Copyright: 2010-2015, iMinds-Vision Lab, University of Antwerp
%            2014-2015, CWI, Amsterdam
% License: Open Source under GPLv3
% Contact: astra@uantwerpen.be
% Website: http://sf.net/projects/astra-toolbox
% -----------------------------------------------------------------------

% This sample illustrates the use of opTomo.
%
% opTomo is a wrapper around the FP and BP operations of the ASTRA Toolbox,
% to allow you to use them as you would a matrix.
%
% This class requires the Spot Linear-Operator Toolbox to be installed.
% You can download this at http://www.cs.ubc.ca/labs/scl/spot/

% load a phantom image
im = phantom(256);
% and flatten it to a vector
x = im(:);

%% Setting up the geometry
% projection geometry
proj_geom = astra_create_proj_geom('parallel', 1, 256, linspace2(0,pi,180));
% object dimensions
vol_geom  = astra_create_vol_geom(256,256);

%% Generate projection data
% Create the Spot operator for ASTRA using the GPU.
W = opTomo('cuda', proj_geom, vol_geom);

p = W*x;

% reshape the vector into a sinogram
sinogram = reshape(p, W.proj_size);
imshow(sinogram, []);


%% Reconstruction
% We use a least squares solver lsqr from Matlab to solve the 
% equation W*x = p.
% Max number of iterations is 100, convergence tolerance of 1e-6.
y = lsqr(W, p, 1e-6, 100);

% the output is a vector, so we reshape it into an image
reconstruction = reshape(y, W.vol_size);

subplot(1,3,1);
imshow(reconstruction, []);
title('Reconstruction');

subplot(1,3,2);
imshow(im, []);
title('Ground truth');

% The transpose of the operator corresponds to the backprojection.
backProjection = W'*p;
subplot(1,3,3);
imshow(reshape(backProjection, W.vol_size), []);
title('Backprojection');
