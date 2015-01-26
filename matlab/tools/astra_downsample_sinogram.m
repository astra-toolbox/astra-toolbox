function [sinogram_new, proj_geom_new] = astra_downsample_sinogram(sinogram, proj_geom, factor)

%------------------------------------------------------------------------
% [sinogram_new, proj_geom_new] = astra_downsample_sinogram(sinogram, proj_geom, factor)
% 
% Downsample the sinogram with some factor and adjust projection geometry
% accordingly
% 
% sinogram: MATLAB data version of the sinogram.
% proj_geom: MATLAB struct containing the projection geometry.
% factor: the factor by which the number of detectors is divided.
% sinogram_new: MATLAB data version of the resampled sinogram.
% proj_geom_new: MATLAB struct containing the new projection geometry.
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

if mod(size(sinogram,2),factor) ~= 0
	disp('size of the sinogram must be a divisor of the factor');
end

sinogram_new = zeros(size(sinogram,1),size(sinogram,2)/factor);
for i = 1:size(sinogram,2)/factor
	sinogram_new(:,i) = sum(sinogram(:,(factor*(i-1)+1):factor*i),2);
end

proj_geom_new = proj_geom;
proj_geom_new.DetectorCount = proj_geom_new.DetectorCount / factor;
