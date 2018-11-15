function [ ] = draw_parallel3d_geom( h_ax, geom, options)
%%  draw_parallel3d_geom.m
%   draw an astra parallel3d projection geometry
%   param h_ax                  handle to axis to draw into
%   param geom                  the geometry to draw
%   param options               options struct with additional settings
%         .SourceDistance       distance of source to origin
%         .SourceMarker         marker for the source locations
%         .SourceMarkerColor    color specifier for the source marker
%         .DetectorMarker       marker for the detector locations.
%                               Default = '.'
%         .DetectorMarkerColor  color specifier for the detector marker.
%                               Default = 'k'
%         .DetectorLineColor    color for the lines drawing the detector
%                               outline
%         .OpticalAxisColor     Color of the line representing the optical
%                               axis.
%   date                        22.06.2018
%   author                      Tim Elberfeld
%                               imec VisionLab
%                               University of Antwerp
%
% - last update                 09.07.2018
%%
    dist_origin_detector = options.SourceDistance;
    dist_origin_source = options.SourceDistance;    
    hold on;
    
    % draw source
    scatter3(h_ax, 0, -dist_origin_source, 0, options.SourceMarker,...
        options.SourceMarkerColor);
        
    % draw detector
    detector = draw_detector(h_ax, geom, dist_origin_detector, options);

    % draw origin
    h_origin = scatter3(h_ax, 0,0,0, '+k');
    h_origin.SizeData = 120;
    h_origin.LineWidth = 2;
    
     % draw lines between source, origin and detector
    line(h_ax, [0, 0], [0, dist_origin_detector], [0, 0],...
        'Color', options.OpticalAxisColor);
    line(h_ax, [0, 0], [0, -dist_origin_source], [0, 0],...
        'Color', options.OpticalAxisColor);
    
    % connect source to detector edges
    for idx = 1:4
        line(h_ax,[detector.vertices(1, idx),...
            detector.vertices(1, idx)],...
            [-dist_origin_source, dist_origin_detector],...
            [detector.vertices(3, idx), detector.vertices(3, idx)],...
            'Color', 'k', 'LineStyle', ':');
    end

    % draw rotation axis
    line(h_ax, [0,0],[0,0], 0.6*[-detector.height, detector.height],...
        'LineWidth', 2, 'Color', 'k', 'LineStyle', '--');

    perc = 0.05;
    text(h_ax, perc*detector.width, perc*detector.width,...
        0.8*detector.height, 'rotation axis');
    text(h_ax, detector.width*perc, 0, perc*detector.height, 'origin');
    text(h_ax, detector.width*perc, -dist_origin_source,...
        perc*detector.height, 'x-ray source');
    text(h_ax, 0, dist_origin_detector, 0, 'detector');
    hold off;

    function [detector] = draw_detector(h_ax, geom,...
            dist_origin_detector, options)
        detector = struct;
        detector.height = geom.DetectorRowCount * geom.DetectorSpacingY;
        detector.width = geom.DetectorColCount * geom.DetectorSpacingX;
        
        vertices = zeros(3, 5);
        vertices(1, :) = 0.5*[-detector.width, -detector.width,...
            detector.width, detector.width, -detector.width];
        vertices(2, :) = repmat(dist_origin_detector, 5, 1);
        vertices(3, :) = 0.5*[detector.height, -detector.height,...
            -detector.height, detector.height, detector.height];
        
        detector.vertices = vertices;
        plot3(h_ax, detector.vertices(1, :), detector.vertices(2, :),...
            detector.vertices(3, :), options.DetectorLineColor);
    end

end

