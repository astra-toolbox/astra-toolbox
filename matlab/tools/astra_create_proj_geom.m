function proj_geom = astra_create_proj_geom(type, varargin)

%--------------------------------------------------------------------------
% proj_geom = astra_create_proj_geom('parallel', det_width, det_count, angles)
%
% Create a 2D parallel beam geometry.  See the API for more information.
% det_width: distance between two adjacent detectors
% det_count: number of detectors in a single projection
% angles: projection angles in radians, should be between -pi/4 and 7pi/4
% proj_geom: MATLAB struct containing all information of the geometry
%--------------------------------------------------------------------------
% proj_geom = astra_create_proj_geom('parallel3d', det_spacing_x, det_spacing_y, det_row_count, det_col_count, angles)
%
% Create a 3D parallel beam geometry.  See the API for more information.
% det_spacing_x: distance between two horizontally adjacent detectors
% det_spacing_y: distance between two vertically adjacent detectors
% det_row_count: number of detector rows in a single projection
% det_col_count: number of detector columns in a single projection
% angles: projection angles in radians, should be between -pi/4 and 7pi/4
% proj_geom: MATLAB struct containing all information of the geometry
%--------------------------------------------------------------------------
% proj_geom = astra_create_proj_geom('fanflat', det_width, det_count, angles, source_origin, origin_det)
%
% Create a 2D flat fan beam geometry.  See the API for more information.
% det_width: distance between two adjacent detectors
% det_count: number of detectors in a single projection
% angles: projection angles in radians, should be between -pi/4 and 7pi/4
% source_origin: distance between the source and the center of rotation
% origin_det: distance between the center of rotation and the detector array
% proj_geom: MATLAB struct containing all information of the geometry
%--------------------------------------------------------------------------
% proj_geom = astra_create_proj_geom('fanflat_vec', det_count, vectors)
%
% Create a 2D flat fan beam geometry specified by 2D vectors.
%   See the API for more information.
% det_count: number of detectors in a single projection
% vectors: a matrix containing the actual geometry. Each row corresponds
%          to a single projection, and consists of:
%          ( srcX, srcY, dX, dY, uX, uY )
%          src: the ray source
%          d  : the center of the detector
%          u  : the vector from detector pixel 0 to 1
% proj_geom: MATLAB struct containing all information of the geometry
%--------------------------------------------------------------------------
% proj_geom = astra_create_proj_geom('cone',  det_spacing_x, det_spacing_y, det_row_count, det_col_count, angles, source_origin, origin_det)
%
% Create a 3D cone beam geometry.  See the API for more information.
% det_spacing_x: distance between two horizontally adjacent detectors
% det_spacing_y: distance between two vertically adjacent detectors
% det_row_count: number of detector rows in a single projection
% det_col_count: number of detector columns in a single projection
% angles: projection angles in radians, should be between -pi/4 and 7pi/4
% source_origin: distance between the source and the center of rotation
% origin_det: distance between the center of rotation and the detector array
% proj_geom: MATLAB struct containing all information of the geometry
%--------------------------------------------------------------------------
% proj_geom = astra_create_proj_geom('cone_vec',  det_row_count, det_col_count, vectors)
%
% Create a 3D cone beam geometry specified by 3D vectors.
%   See the API for more information.
% det_row_count: number of detector rows in a single projection
% det_col_count: number of detector columns in a single projection
% vectors: a matrix containing the actual geometry. Each row corresponds
%          to a single projection, and consists of:
%          ( srcX, srcY, srcZ, dX, dY, dZ, uX, uY, uZ, vX, vY, vZ )
%          src: the ray source
%          d  : the center of the detector
%          u  : the vector from detector pixel (0,0) to (0,1)
%          v  : the vector from detector pixel (0,0) to (1,0)
% proj_geom: MATLAB struct containing all information of the geometry
%--------------------------------------------------------------------------
% proj_geom = astra_create_proj_geom('parallel3d_vec',  det_row_count, det_col_count, vectors)
%
% Create a 3D parallel beam geometry specified by 3D vectors.
%   See the API for more information.
% det_row_count: number of detector rows in a single projection
% det_col_count: number of detector columns in a single projection
% vectors: a matrix containing the actual geometry. Each row corresponds
%          to a single projection, and consists of:
%          ( rayX, rayY, rayZ, dX, dY, dZ, uX, uY, uZ, vX, vY, vZ )
%          ray: the ray direction
%          d  : the center of the detector
%          u  : the vector from detector pixel (0,0) to (0,1)
%          v  : the vector from detector pixel (0,0) to (1,0)
% proj_geom: MATLAB struct containing all information of the geometry
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
% This file is part of the ASTRA Toolbox
% 
% Copyright: 2010-2015, iMinds-Vision Lab, University of Antwerp
%            2014-2015, CWI, Amsterdam
% License: Open Source under GPLv3
% Contact: astra@uantwerpen.be
% Website: http://sf.net/projects/astra-toolbox
%--------------------------------------------------------------------------
% $Id$


if strcmp(type,'parallel')
	if numel(varargin) < 3
		error('not enough variables: astra_create_proj_geom(parallel, detector_spacing, det_count, angles)');
	end
	proj_geom = struct( ...
		'type',					'parallel', ...
		'DetectorWidth',		varargin{1}, ...
		'DetectorCount',		varargin{2}, ...
		'ProjectionAngles',		varargin{3}  ...
	);

elseif strcmp(type,'fanflat')
	if numel(varargin) < 5
		error('not enough variables: astra_create_proj_geom(fanflat, det_width, det_count, angles, source_origin, source_det)');
	end	
	proj_geom = struct( ...
		'type',						'fanflat', ...
		'DetectorWidth',			varargin{1}, ...
		'DetectorCount',			varargin{2}, ...
		'ProjectionAngles',			varargin{3},  ...
		'DistanceOriginSource',		varargin{4},  ...		
		'DistanceOriginDetector',	varargin{5}  ...
	);

elseif strcmp(type,'fanflat_vec')
	if numel(varargin) < 2
		error('not enough variables: astra_create_proj_geom(fanflat_vec, det_count, V')
	end
	if size(varargin{2}, 2) ~= 6
		error('V should be a Nx6 matrix, with N the number of projections')
	end
	proj_geom = struct( ...
		'type',					'fanflat_vec',  ...
		'DetectorCount',		varargin{1}, ...
		'Vectors',				varargin{2}  ...
	);

elseif strcmp(type,'parallel3d')
	if numel(varargin) < 5
		error('not enough variables: astra_create_proj_geom(parallel3d, detector_spacing_x, detector_spacing_y, det_row_count, det_col_count, angles)');
	end
	proj_geom = struct( ...
		'type',					'parallel3d', ...
		'DetectorSpacingX',		varargin{1}, ...
		'DetectorSpacingY',		varargin{2}, ...
		'DetectorRowCount',		varargin{3}, ...
		'DetectorColCount',		varargin{4}, ...
		'ProjectionAngles',		varargin{5}  ...
	);
elseif strcmp(type,'cone')
	if numel(varargin) < 7
		error('not enough variables: astra_create_proj_geom(cone, detector_spacing_x, detector_spacing_y, det_row_count, det_col_count, angles, source_origin, source_det)');
	end
	proj_geom = struct( ...
		'type',					'cone', ...
		'DetectorSpacingX',		varargin{1}, ...
		'DetectorSpacingY',		varargin{2}, ...
		'DetectorRowCount',		varargin{3}, ...
		'DetectorColCount',		varargin{4}, ...
		'ProjectionAngles',		varargin{5}, ...
		'DistanceOriginSource',	varargin{6},  ...		
		'DistanceOriginDetector',varargin{7}  ...		
	);
elseif strcmp(type,'cone_vec')
	if numel(varargin) < 3
		error('not enough variables: astra_create_proj_geom(cone_vec, det_row_count, det_col_count, V')
	end
	if size(varargin{3}, 2) ~= 12
		error('V should be a Nx12 matrix, with N the number of projections')
	end
	proj_geom = struct( ...
		'type',					'cone_vec',  ...
		'DetectorRowCount',		varargin{1}, ...
		'DetectorColCount',		varargin{2}, ...
		'Vectors',				varargin{3}  ...
	);
elseif strcmp(type,'parallel3d_vec')
	if numel(varargin) < 3
		error('not enough variables: astra_create_proj_geom(parallel3d_vec, det_row_count, det_col_count, V')
	end
	if size(varargin{3}, 2) ~= 12
		error('V should be a Nx12 matrix, with N the number of projections')
	end
	proj_geom = struct( ...
		'type',					'parallel3d_vec',  ...
		'DetectorRowCount',		varargin{1}, ...
		'DetectorColCount',		varargin{2}, ...
		'Vectors',				varargin{3}  ...
	);
elseif strcmp(type,'sparse_matrix')
	if numel(varargin) < 3
		error('not enough variables: astra_create_proj_geom(sparse_matrix, det_width, det_count, angles, matrix_id)')
	end
	proj_geom = struct( ...
		'type',					'sparse_matrix', ...
		'DetectorWidth',		varargin{1}, ...
		'DetectorCount',		varargin{2}, ...
		'ProjectionAngles',		varargin{3}, ...
		'MatrixID',				varargin{4} ...
	);

else
	disp(['Error: unknown type ' type]);
	proj_geom = struct();
end

