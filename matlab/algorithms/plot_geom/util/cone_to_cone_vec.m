function [ cone_vec_geom] = cone_to_cone_vec( cone_geom )
%% cone_to_cone_vec.m
% brief                 convert a cone beam projection geometry into a 
%                       cone_vec geometry according to:
%   http://www.astra-toolbox.com/docs/geom3d.html#projection-geometries
%
% param cone_geom       the cone beam geometry to convert
% return cone_vec_geom  the converted geometry
% 
% date      21.06.2018
% author    Tim Elberfeld
%           imec VisionLab
%           University of Antwerp
%%
    num_angles = numel(cone_geom.ProjectionAngles);
    vectors = zeros(num_angles, 12);

    for idx = 1:num_angles
        % source
        vectors(idx, 1) = sin(cone_geom.ProjectionAngles(idx)) * cone_geom.DistanceOriginSource;
        vectors(idx, 2) = -cos(cone_geom.ProjectionAngles(idx)) * cone_geom.DistanceOriginSource;
        vectors(idx, 3) = 0;

        % center of detector
        vectors(idx, 4) = -sin(cone_geom.ProjectionAngles(idx)) * cone_geom.DistanceOriginDetector;
        vectors(idx, 5) = cos(cone_geom.ProjectionAngles(idx)) * cone_geom.DistanceOriginDetector;
        vectors(idx, 6) = 0;

        % vector from detector pixel (0,0) to (0,1)
        vectors(idx, 7) = cos(cone_geom.ProjectionAngles(idx)) * cone_geom.DetectorSpacingX;
        vectors(idx, 8) = sin(cone_geom.ProjectionAngles(idx)) * cone_geom.DetectorSpacingX;
        vectors(idx, 9) = 0;

        % vector from detector pixel (0,0) to (1,0)
        vectors(idx, 10) = 0;
        vectors(idx, 11) = 0;
        vectors(idx, 12) = cone_geom.DetectorSpacingY;
    end
    
    cone_vec_geom = astra_create_proj_geom('cone_vec', cone_geom.DetectorRowCount, cone_geom.DetectorColCount, vectors);
end
