function vol_geom = astra_create_vol_geom(varargin)

%--------------------------------------------------------------------------
% vol_geom = astra_create_vol_geom([row_count col_count]);
% vol_geom = astra_create_vol_geom(row_count, col_count);
% vol_geom = astra_create_vol_geom(row_count, col_count, min_x, max_x, min_y, max_y);
%
% Create a 2D volume geometry.  See the API for more information.
% row_count: number of rows.
% col_count: number of columns.
% min_x: minimum value on the x-axis.
% max_x: maximum value on the x-axis.
% min_y: minimum value on the y-axis.
% max_y: maximum value on the y-axis.
% vol_geom: MATLAB struct containing all information of the geometry.
%--------------------------------------------------------------------------
% vol_geom = astra_create_vol_geom(row_count, col_count, slice_count);
% vol_geom = astra_create_vol_geom(row_count, col_count, slice_count, min_x, max_x, min_y, max_y, min_z, max_z);
%
% Create a 3D volume geometry.  See the API for more information.
% row_count: number of rows.
% col_count: number of columns.
% slice_count: number of slices.
% vol_geom: MATLAB struct containing all information of the geometry.
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

% astra_create_vol_geom([row_and_col_count ])
if numel(varargin) == 1 && numel(varargin{1}) == 1
	vol_geom = struct();
	vol_geom.GridRowCount = varargin{1}(1);
	vol_geom.GridColCount =	varargin{1}(1);
	vol_geom.option.WindowMinX = -varargin{1}(1) / 2;
	vol_geom.option.WindowMaxX = varargin{1}(1) / 2;
	vol_geom.option.WindowMinY = -varargin{1}(1) / 2;
	vol_geom.option.WindowMaxY = varargin{1}(1) / 2;


% astra_create_vol_geom([row_count col_count])
elseif numel(varargin) == 1 && numel(varargin{1}) == 2
	vol_geom = struct();
	vol_geom.GridRowCount = varargin{1}(1);
	vol_geom.GridColCount =	varargin{1}(2);
	vol_geom.option.WindowMinX = -varargin{1}(2) / 2;
	vol_geom.option.WindowMaxX = varargin{1}(2) / 2;
	vol_geom.option.WindowMinY = -varargin{1}(1) / 2;
	vol_geom.option.WindowMaxY = varargin{1}(1) / 2;

% astra_create_vol_geom([row_count col_count slice_count])
elseif numel(varargin) == 1 && numel(varargin{1}) == 3
	vol_geom = struct();
	vol_geom.GridRowCount = varargin{1}(1);
	vol_geom.GridColCount =	varargin{1}(2);
	vol_geom.GridSliceCount = varargin{1}(3);	
	vol_geom.option.WindowMinX = -varargin{1}(2) / 2;
	vol_geom.option.WindowMaxX = varargin{1}(2) / 2;
	vol_geom.option.WindowMinY = -varargin{1}(1) / 2;
	vol_geom.option.WindowMaxY = varargin{1}(1) / 2;
	vol_geom.option.WindowMinZ = -varargin{1}(3) / 2;
	vol_geom.option.WindowMaxZ = varargin{1}(3) / 2;	
	
% astra_create_vol_geom(row_count, col_count)
elseif numel(varargin) == 2
	vol_geom = struct();
	vol_geom.GridRowCount = varargin{1};
	vol_geom.GridColCount =	varargin{2};
	vol_geom.option.WindowMinX = -varargin{2} / 2;
	vol_geom.option.WindowMaxX = varargin{2} / 2;
	vol_geom.option.WindowMinY = -varargin{1} / 2;
	vol_geom.option.WindowMaxY = varargin{1} / 2;

% astra_create_vol_geom(row_count, col_count, min_x, max_x, min_y, max_y)
elseif numel(varargin) == 6
	vol_geom = struct();
	vol_geom.GridRowCount = varargin{1};
	vol_geom.GridColCount = varargin{2};
	vol_geom.option.WindowMinX = varargin{3};
	vol_geom.option.WindowMaxX = varargin{4};
	vol_geom.option.WindowMinY = varargin{5};
	vol_geom.option.WindowMaxY = varargin{6};

% astra_create_vol_geom(row_count, col_count, slice_count)
elseif numel(varargin) == 3
	vol_geom = struct();
	vol_geom.GridRowCount =	varargin{1};
	vol_geom.GridColCount = varargin{2};
	vol_geom.GridSliceCount = varargin{3};

% astra_create_vol_geom(row_count, col_count, slice_count, min_x, max_x, min_y, max_y, min_z, max_z)
elseif numel(varargin) == 9
	vol_geom = struct();
	vol_geom.GridRowCount = varargin{1};
	vol_geom.GridColCount = varargin{2};
	vol_geom.GridSliceCount = varargin{3};
	vol_geom.option.WindowMinX = varargin{4};
	vol_geom.option.WindowMaxX = varargin{5};
	vol_geom.option.WindowMinY = varargin{6};
	vol_geom.option.WindowMaxY = varargin{7};
	vol_geom.option.WindowMinZ = varargin{8};
	vol_geom.option.WindowMaxZ = varargin{9};

end
