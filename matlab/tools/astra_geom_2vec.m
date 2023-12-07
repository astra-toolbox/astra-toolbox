function proj_geom_vec = astra_geom_2vec(proj_geom)
%--------------------------------------------------------------------------
% proj_geom_vec = astra_geom_2vec(proj_geom)
%
% Convert a conventional projection geometry to a corresponding vector-base
% projection geometry
%
% proj_geom: input projection geometry (parallel, fanflat, parallel3d, cone)
% proj_geom_vec: output vector-base projection geometry
%--------------------------------------------------------------------------
% This file is part of the ASTRA Toolbox
%
% Copyright: 2010-2022, imec Vision Lab, University of Antwerp
%            2014-2022, CWI, Amsterdam
% License: Open Source under GPLv3
% Contact: astra@astra-toolbox.com
% Website: http://www.astra-toolbox.com/
%--------------------------------------------------------------------------

	% PARALLEL 2D
	if strcmp(proj_geom.type,'parallel')

		vectors = zeros(numel(proj_geom.ProjectionAngles), 6);
		for i = 1:numel(proj_geom.ProjectionAngles)

			% ray direction
			vectors(i,1) = sin(proj_geom.ProjectionAngles(i));
			vectors(i,2) = -cos(proj_geom.ProjectionAngles(i));

			% center of detector
			vectors(i,3) = 0;
			vectors(i,4) = 0;

			% vector from detector pixel 0 to 1
			vectors(i,5) = cos(proj_geom.ProjectionAngles(i)) * proj_geom.DetectorWidth;
			vectors(i,6) = sin(proj_geom.ProjectionAngles(i)) * proj_geom.DetectorWidth;

		end

		proj_geom_vec = astra_create_proj_geom('parallel_vec', proj_geom.DetectorCount, vectors);

	% FANFLAT
	elseif strcmp(proj_geom.type,'fanflat')

		vectors = zeros(numel(proj_geom.ProjectionAngles), 6);
		for i = 1:numel(proj_geom.ProjectionAngles)

			% source
			vectors(i,1) = sin(proj_geom.ProjectionAngles(i)) * proj_geom.DistanceOriginSource;
			vectors(i,2) = -cos(proj_geom.ProjectionAngles(i)) * proj_geom.DistanceOriginSource;

			% center of detector
			vectors(i,3) = -sin(proj_geom.ProjectionAngles(i)) * proj_geom.DistanceOriginDetector;
			vectors(i,4) = cos(proj_geom.ProjectionAngles(i)) * proj_geom.DistanceOriginDetector;

			% vector from detector pixel 0 to 1
			vectors(i,5) = cos(proj_geom.ProjectionAngles(i)) * proj_geom.DetectorWidth;
			vectors(i,6) = sin(proj_geom.ProjectionAngles(i)) * proj_geom.DetectorWidth;
		end

		proj_geom_vec = astra_create_proj_geom('fanflat_vec', proj_geom.DetectorCount, vectors);

	% CONE
	elseif strcmp(proj_geom.type,'cone')

		vectors = zeros(numel(proj_geom.ProjectionAngles), 12);
		for i = 1:numel(proj_geom.ProjectionAngles)

			% source
			vectors(i,1) = sin(proj_geom.ProjectionAngles(i)) * proj_geom.DistanceOriginSource;
			vectors(i,2) = -cos(proj_geom.ProjectionAngles(i)) * proj_geom.DistanceOriginSource;
			vectors(i,3) = 0;

			% center of detector
			vectors(i,4) = -sin(proj_geom.ProjectionAngles(i)) * proj_geom.DistanceOriginDetector;
			vectors(i,5) = cos(proj_geom.ProjectionAngles(i)) * proj_geom.DistanceOriginDetector;
			vectors(i,6) = 0;

			% vector from detector pixel (0,0) to (0,1)
			vectors(i,7) = cos(proj_geom.ProjectionAngles(i)) * proj_geom.DetectorSpacingX;
			vectors(i,8) = sin(proj_geom.ProjectionAngles(i)) * proj_geom.DetectorSpacingX;
			vectors(i,9) = 0;

			% vector from detector pixel (0,0) to (1,0)
			vectors(i,10) = 0;
			vectors(i,11) = 0;
			vectors(i,12) = proj_geom.DetectorSpacingY;
		end

		proj_geom_vec = astra_create_proj_geom('cone_vec', proj_geom.DetectorRowCount, proj_geom.DetectorColCount, vectors);

	% PARALLEL 3D
	elseif strcmp(proj_geom.type,'parallel3d')

		vectors = zeros(numel(proj_geom.ProjectionAngles), 12);
		for i = 1:numel(proj_geom.ProjectionAngles)

			% ray direction
			vectors(i,1) = sin(proj_geom.ProjectionAngles(i));
			vectors(i,2) = -cos(proj_geom.ProjectionAngles(i));
			vectors(i,3) = 0;

			% center of detector
			vectors(i,4) = 0;
			vectors(i,5) = 0;
			vectors(i,6) = 0;

			% vector from detector pixel (0,0) to (0,1)
			vectors(i,7) = cos(proj_geom.ProjectionAngles(i)) * proj_geom.DetectorSpacingX;
			vectors(i,8) = sin(proj_geom.ProjectionAngles(i)) * proj_geom.DetectorSpacingX;
			vectors(i,9) = 0;

			% vector from detector pixel (0,0) to (1,0)
			vectors(i,10) = 0;
			vectors(i,11) = 0;
			vectors(i,12) = proj_geom.DetectorSpacingY;
		end

		proj_geom_vec = astra_create_proj_geom('parallel3d_vec', proj_geom.DetectorRowCount, proj_geom.DetectorColCount, vectors);

	elseif ismember(proj_geom.type, {'parallel_vec', 'fanflat_vec', 'parallel3d_vec', 'cone_vec'})
		proj_geom_vec = proj_geom;

	else
		error(['No suitable vector geometry found for type: ' proj_geom.type '.'])
	end

end
