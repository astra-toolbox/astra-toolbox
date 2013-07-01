function proj_geom = astra_geom_superresolution(proj_geom, factor)

	if  strcmp(proj_geom.type,'parallel')
			proj_geom.DetectorWidth = proj_geom.DetectorWidth/factor;
			proj_geom.DetectorCount = proj_geom.DetectorCount * factor;
	elseif strcmp(proj_geom.type,'fanflat') 
			proj_geom.DetectorWidth = proj_geom.DetectorWidth/factor;
			proj_geom.DetectorCount = proj_geom.DetectorCount * factor;
	elseif strcmp(proj_geom.type,'fanflat_vec')
			proj_geom.Vectors(:,5:6) = proj_geom.Vectors(:,5:6) / factor; % DetectorSize			
			proj_geom.DetectorCount = proj_geom.DetectorCount * factor;
	else
		error('Projection geometry not suited for super-resolution (or not implemented).')
	end
