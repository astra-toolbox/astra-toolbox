% -----------------------------------------------------------------------
% This file is part of the ASTRA Toolbox
% 
% Copyright: 2010-2015, iMinds-Vision Lab, University of Antwerp
%            2014-2015, CWI, Amsterdam
% License: Open Source under GPLv3
% Contact: astra@uantwerpen.be
% Website: http://sf.net/projects/astra-toolbox
% -----------------------------------------------------------------------

vol_geom = astra_create_vol_geom(64, 64, 64);


% There are two main 3d projection geometry types: cone beam and parallel beam.
% Each has a regular variant, and a 'vec' variant.
% The 'vec' variants are completely free in the placement of source/detector,
% while the regular variants assume circular trajectories around the z-axis.


% -------------
% Parallel beam
% -------------


% Circular

% Parameters: width of detector column, height of detector row, #rows, #columns
angles = linspace2(0, 2*pi, 48);
proj_geom = astra_create_proj_geom('parallel3d', 1.0, 1.0, 32, 64, angles);


% Free

% We generate the same geometry as the circular one above. 
vectors = zeros(numel(angles), 12);
for i = 1:numel(angles)
  % ray direction
  vectors(i,1) = sin(angles(i));
  vectors(i,2) = -cos(angles(i));
  vectors(i,3) = 0;	

  % center of detector
  vectors(i,4:6) = 0;

  % vector from detector pixel (0,0) to (0,1)
  vectors(i,7) = cos(angles(i));
  vectors(i,8) = sin(angles(i));
  vectors(i,9) = 0;

  % vector from detector pixel (0,0) to (1,0)
  vectors(i,10) = 0;
  vectors(i,11) = 0;
  vectors(i,12) = 1;
end

% Parameters: #rows, #columns, vectors
proj_geom = astra_create_proj_geom('parallel3d_vec', 32, 64, vectors);

% ----------
% Cone beam
% ----------


% Circular

% Parameters: width of detector column, height of detector row, #rows, #columns,
%             angles, distance source-origin, distance origin-detector
angles = linspace2(0, 2*pi, 48);
proj_geom = astra_create_proj_geom('cone', 1.0, 1.0, 32, 64, ...
                                   angles, 1000, 0);

% Free

vectors = zeros(numel(angles), 12);
for i = 1:numel(angles)

	% source
	vectors(i,1) = sin(angles(i)) * 1000;
	vectors(i,2) = -cos(angles(i)) * 1000;
	vectors(i,3) = 0;	

	% center of detector
	vectors(i,4:6) = 0;

	% vector from detector pixel (0,0) to (0,1)
	vectors(i,7) = cos(angles(i));
	vectors(i,8) = sin(angles(i));
	vectors(i,9) = 0;

	% vector from detector pixel (0,0) to (1,0)
	vectors(i,10) = 0;
	vectors(i,11) = 0;
	vectors(i,12) = 1;		
end

% Parameters: #rows, #columns, vectors
proj_geom = astra_create_proj_geom('cone_vec', 32, 64, vectors);

