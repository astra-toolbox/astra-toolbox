% -----------------------------------------------------------------------
% This file is part of the ASTRA Toolbox
% 
% Copyright: 2010-2015, iMinds-Vision Lab, University of Antwerp
%            2014-2015, CWI, Amsterdam
% License: Open Source under GPLv3
% Contact: astra@uantwerpen.be
% Website: http://sf.net/projects/astra-toolbox
% -----------------------------------------------------------------------

vol_geom = astra_create_vol_geom(256, 256);
proj_geom = astra_create_proj_geom('parallel', 1.0, 384, linspace2(0,pi,180));

% For CPU-based algorithms, a "projector" object specifies the projection
% model used. In this case, we use the "strip" model.
proj_id = astra_create_projector('strip', proj_geom, vol_geom);

% Generate the projection matrix for this projection model.
% This creates a matrix W where entry w_{i,j} corresponds to the
% contribution of volume element j to detector element i.
matrix_id = astra_mex_projector('matrix', proj_id);

% Get the projection matrix as a Matlab sparse matrix.
W = astra_mex_matrix('get', matrix_id);


% Manually use this projection matrix to do a projection:
P = phantom(256)';
s = W * P(:);
s = reshape(s, [proj_geom.DetectorCount size(proj_geom.ProjectionAngles, 2)])';
figure(1), imshow(s,[]);

% Because Matlab's matrices are stored transposed in memory compared to C++,
% reshaping them to a vector doesn't give the right ordering for multiplication
% with W. We have to take the transpose of the input and output to get the same
% results (up to numerical noise) as using the toolbox directly.

% Each row of the projection matrix corresponds to a detector element.
% Detector t for angle p is for row 1 + t + p*proj_geom.DetectorCount.
% Each column corresponds to a volume pixel.
% Pixel (x,y) corresponds to column 1 + x + y*vol_geom.GridColCount.


astra_mex_projector('delete', proj_id);
astra_mex_matrix('delete', matrix_id);
