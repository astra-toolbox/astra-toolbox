function proj_geom = astra_geom_postalignment(proj_geom, factor)

	if strcmp(proj_geom.type,'fanflat_vec')
		proj_geom.Vectors(:,3:4) = proj_geom.Vectors(:,3:4) + factor * proj_geom.Vectors(:,5:6);
		
	elseif strcmp(proj_geom.type,'cone_vec') || strcmp(proj_geom.type,'parallel3d_vec')
		proj_geom.Vectors(:,4:6) = proj_geom.Vectors(:,4:6) + factor * proj_geom.Vectors(:,7:9);

	else
		error('Projection geometry not suited for postalignment correction.')
	end
