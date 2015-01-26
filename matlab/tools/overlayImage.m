function im = overlayImage(reconstruction, ground_truth)
%------------------------------------------------------------------------
% im = overlayImage(reconstruction, ground_truth)
% 
% Produces an overlay image of two images, useful for image comparison.
%
% reconstruction: first input image matrix.
% ground_truth: second input image matrix.
% im: output 3-component image, third channel is 0.
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

im(:,:,1) = reconstruction ./ max(reconstruction(:));
im(:,:,2) = ground_truth ./ max(ground_truth(:));
im(:,:,3) = zeros(size(reconstruction));
