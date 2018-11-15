function [ proj_geom ] = create_example_parallel3d( type )
%% create_example_parallel3d.m
% brief         create an example geometry of type 'parallel3d'
% param type    type of geometry to create. provided as one of the
%               following strings (not case sensitive).
%               no arguments will create a standard fanflat geometry.
%               'vec'           -   example vector geometry

% return        proj_geom       -   the geometry that was created
% date          22.06.2018
% author        Tim Elberfeld
%               imec VisionLab
%               University of Antwerp
% last mod      07.11.2018
%%
    if nargin < 1
        type = 'nothing';
    end
    
    det_spacing = [0.035, 0.035];
    detector_px = [1000, 1000];
    angles = linspace(0, 2*pi, 100);

    proj_geom = astra_create_proj_geom('parallel3d', det_spacing(1),...
        det_spacing(2), detector_px(1), detector_px(2), angles);

    if strcmp(type, 'vec')
        proj_geom = astra_geom_2vec(proj_geom);
    end
end
