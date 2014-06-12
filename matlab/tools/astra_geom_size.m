function s = astra_geom_size(geom, dim)

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
		
	elseif strcmp(geom.type,'fanflat_vec')
		s = [size(geom.Vectors,1), geom.DetectorCount];
		
	elseif strcmp(geom.type,'parallel3d_vec') || strcmp(geom.type,'cone_vec') 		
		s = [geom.DetectorColCount, size(geom.Vectors,1), geom.DetectorRowCount];
		
	end

	if nargin == 2
		s = s(dim);
	end
	
end

