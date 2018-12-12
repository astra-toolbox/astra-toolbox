function [ geom ] = create_example_fanflat( type )
%% create_example_fanflat.m
% brief         create an example geometry of type 'fanflat'
% param type    type of geometry to create. provided as one of the
%               following strings (not case sensitive):
%               'normal'        -   standard fanflat geometry
%               'vec'           -   example vector geometry
% return        proj_geom       -   the geometry that was created
% date          22.06.2018
% author        Tim Elberfeld
%               imec VisionLab
%               University of Antwerp
% last mod      07.11.2018
%%
    if nargin < 1
        type = 'normal';
    end
    
    if strcmpi(type, 'normal')
         geom = make_normal_geometry();
    elseif strcmpi(type, 'vec')
        geom = astra_create_example_fanflat('normal');
        geom = astra_geom_2vec(geom);
        
    else
        geom = make_normal_geometry();
    end    
    
    function [geom] = make_normal_geometry()
        % first, give measurements in mm
        det_spacing = 0.035;
        detector_px = 1200;
        angles = linspace2(0, 2*pi, 100);
        source_origin = 30;
        origin_det = 200;
        phantom_size = 5;

        phantom_px = 150; % voxels for the phantom
        vx_size = phantom_size / phantom_px; % voxel size

        % now express all measurements in terms of the voxel size
        det_spacing = det_spacing ./ vx_size;
        origin_det = origin_det ./ vx_size;
        source_origin = source_origin ./ vx_size;

        geom = astra_create_proj_geom('fanflat',  det_spacing, ...
            detector_px, angles, source_origin, origin_det);        
    end
end

