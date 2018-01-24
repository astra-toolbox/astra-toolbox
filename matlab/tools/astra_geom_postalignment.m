function proj_geom = astra_geom_postalignment(proj_geom, factor)
%--------------------------------------------------------------------------
% proj_geom = astra_geom_postalignment(proj_geom, factorU)
% proj_geom = astra_geom_postalignment(proj_geom, [factorU factorV])
%
% Apply a postalignment to a vector-based projection geometry.  Can be used to model the rotation axis offset.
%
% For 2D geometries, the argument factor is a single float specifying the
% distance to shift the detector (measured in detector pixels).
%
% For 3D geometries, factor is a pair of floats specifying the horizontal
% resp. vertical distances to shift the detector. If only a single float is
% specified, this is treated as an horizontal shift.
%
% proj_geom: input projection geometry (vector-based only, use astra_geom_2vec to convert conventional projection geometries)
% factor: number of pixels to shift the detector
% proj_geom: output
%--------------------------------------------------------------------------
% This file is part of the ASTRA Toolbox
%
% Copyright: 2010-2018, imec Vision Lab, University of Antwerp
%            2014-2018, CWI, Amsterdam
% License: Open Source under GPLv3
% Contact: astra@astra-toolbox.com
% Website: http://www.astra-toolbox.com/
%--------------------------------------------------------------------------

	proj_geom = astra_geom_2vec(proj_geom);

	if strcmp(proj_geom.type,'fanflat_vec') || strcmp(proj_geom.type,'parallel_vec')
		proj_geom.Vectors(:,3:4) = proj_geom.Vectors(:,3:4) + factor(1) * proj_geom.Vectors(:,5:6);

	elseif strcmp(proj_geom.type,'cone_vec') || strcmp(proj_geom.type,'parallel3d_vec')
		if numel(factor) == 1
			proj_geom.Vectors(:,4:6) = proj_geom.Vectors(:,4:6) + factor * proj_geom.Vectors(:,7:9);
		elseif numel(factor) > 1
			proj_geom.Vectors(:,4:6) = proj_geom.Vectors(:,4:6) + factor(1) * proj_geom.Vectors(:,7:9) + factor(2) * proj_geom.Vectors(:,10:12);
		end

	else
		error('Projection geometry not suited for postalignment correction.')
	end

end
