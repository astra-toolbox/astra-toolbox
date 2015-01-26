function rayOrder = createOrderART(proj_geom, mode)

%------------------------------------------------------------------------
% rayOrder = createOrderART(proj_geom, mode)
% 
% Creates an array defining the order in which ART will iterate over the
% projections and projection rays 
%
% proj_geom: MATLAB struct containing the projection geometry.
% mode: string defining the wanted ray order, can be either 'sequential',
% 'randomray' or 'randomproj'.
% rayOrder: array of two columns of length angle_count * det_count, with
% the first column being the index of the projection and the second column 
% the index of the ray.
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

angle_count = length(proj_geom.projection_angles);
det_count = proj_geom.detector_count;

% create order
rayOrder = zeros(angle_count * det_count, 2);
if strcmp(mode,'sequential') == 1
	index = 1;
	for i = 1:angle_count
		for j = 1:det_count
			rayOrder(index,1) = i;
			rayOrder(index,2) = j;			
			index = index + 1;
		end
	end
elseif strcmp(mode,'randomray') == 1
	index = 1;
	for i = 1:angle_count
		for j = 1:det_count
			rayOrder(index,1) = i;
			rayOrder(index,2) = j;			
			index = index + 1;
		end
	end
	r = randperm(angle_count * det_count);
	rayOrder(:,1) = rayOrder(r,1);
	rayOrder(:,2) = rayOrder(r,2);	
elseif strcmp(mode,'randomproj') == 1
	index = 1;
	r = randperm(angle_count);
	for i = 1:angle_count
		for j = 1:det_count
			rayOrder(index,1) = r(i);
			rayOrder(index,2) = j;			
			index = index + 1;
		end
	end
else
	disp('mode not known');
end

