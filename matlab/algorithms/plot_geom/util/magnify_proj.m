
function [ magnified_vec_geom ] = magnify_proj( vec_geom, dsdd )
%%   generate magnified vector geometry
%   param type      vec_geom            -   example vector geometry
%                   dsdd                -   deviation of SDD 
%   return          magnified_vec_geom  -   the geometry that was created
%
%   date            09.07.2018
%   author          Van Nguyen
%                   imec VisionLab
%                   University of Antwerp
%%
    magnified_vec_geom = vec_geom;
    vec_sd_direction = vec_geom(:,1:3) - vec_geom(:,4:6);
    norm_sd = sqrt(sum(vec_sd_direction.^2,2));
    vec_norm_sd(:,1) = norm_sd;
    vec_norm_sd(:,2) = norm_sd;
    vec_norm_sd(:,3) = norm_sd;
    vec_sd_direction = vec_sd_direction ./ vec_norm_sd;
    magnified_vec_geom(:,4:6) = vec_geom(:,4:6) - dsdd * vec_sd_direction;
    clearvars -except magnified_vec_geom
end

