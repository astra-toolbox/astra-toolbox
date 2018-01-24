function s = astra_geom_size(geom, dim)
%--------------------------------------------------------------------------
% s = astra_geom_size(geom, dim)
%
% Get the size of a volume or projection geometry.
%
% geom: volume or projection geometry
% dim (optional): which dimension
% s: output
%--------------------------------------------------------------------------
% This file is part of the ASTRA Toolbox
%
% Copyright: 2010-2018, imec Vision Lab, University of Antwerp
%            2014-2018, CWI, Amsterdam
% License: Open Source under GPLv3
% Contact: astra@astra-toolbox.com
% Website: http://www.astra-toolbox.com/
%--------------------------------------------------------------------------

	if isfield(geom, 'GridSliceCount')
		% 3D Volume geometry?
		s = [ geom.GridColCount, geom.GridRowCount, geom.GridSliceCount ];
	elseif isfield(geom, 'GridColCount')
		% 2D Volume geometry?
		s = [ geom.GridRowCount, geom.GridColCount ];
	elseif strcmp(geom.type,'parallel') || strcmp(geom.type,'fanflat')
		s = [numel(geom.ProjectionAngles), geom.DetectorCount];

	elseif strcmp(geom.type,'parallel3d') || strcmp(geom.type,'cone')
		s =  [geom.DetectorColCount, numel(geom.ProjectionAngles), geom.DetectorRowCount];

	elseif strcmp(geom.type,'fanflat_vec') || strcmp(geom.type,'parallel_vec')
		s = [size(geom.Vectors,1), geom.DetectorCount];

	elseif strcmp(geom.type,'parallel3d_vec') || strcmp(geom.type,'cone_vec')
		s = [geom.DetectorColCount, size(geom.Vectors,1), geom.DetectorRowCount];

	end

	if nargin == 2
		s = s(dim);
	end

end

