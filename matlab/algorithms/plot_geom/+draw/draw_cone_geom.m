function [] = draw_cone_geom(h_ax, geom, options)
%%  draw_cone_geom.m
%   draw an astra cone beam projection geometry
%
%   param h_ax              handle to axis to draw into
%   param geom              the geometry to draw
%   param options           struct containing the options for this function
%       SourceMarker        marker to use to mark the source location
%       SourceMarkerColor   Color of the marker for the source
%       DetectorLineColor   Color of the lines that draw the detector
%       DetectorLineWidth   Width of the lines that draw the detector
%       OpticalAxisColor    color of the lines representing the optical axis
%
%   date                    21.06.2018
%   author                  Tim Elberfeld
%                           imec VisionLab
%                           University of Antwerp
%%
    hold on;
    % draw origin
    h_origin = scatter3(h_ax, 0,0,0, '+k');
    h_origin.SizeData = 120;
    h_origin.LineWidth = 2;

    % draw lines between source, origin and detector
    line(h_ax, [0, 0], [0, geom.DistanceOriginDetector], [0, 0], 'Color', options.OpticalAxisColor);
    line(h_ax, [0, 0], [0, -geom.DistanceOriginSource], [0, 0], 'Color', options.OpticalAxisColor);

    % draw source
    scatter3(h_ax, 0, -geom.DistanceOriginSource, 0, options.SourceMarker,...
        options.SourceMarkerColor);

    detector = draw_detector(h_ax, geom, options);

    % connect source to detector edges
    for idx = 1:4
        line(h_ax,[0, detector.vertices(1, idx)], ...
                  [-geom.DistanceOriginSource, geom.DistanceOriginDetector],...
                  [0, detector.vertices(3, idx)],...
                  'Color', 'k', 'LineStyle', ':');
    end

    % draw rotation axis
    line(h_ax, [0,0],[0,0], 0.6*[-detector.height, detector.height],...
        'LineWidth', options.DetectorLineWidth, 'Color',...
        'k', 'LineStyle', '--');

    perc = 0.05;
    text(h_ax, perc*detector.width, perc*detector.width,...
        0.8*detector.height, 'rotation axis');
    text(h_ax, detector.width*perc, 0, perc*detector.height, 'origin');
    text(h_ax, detector.width*perc, -geom.DistanceOriginSource,...
        perc*detector.height, 'x-ray source');
    text(h_ax, 0, geom.DistanceOriginDetector, 0, 'detector');
    hold off;

    function [detector] = draw_detector(h_ax, geom, options)
        detector = struct;
        detector.height = geom.DetectorRowCount * geom.DetectorSpacingY;
        detector.width = geom.DetectorColCount * geom.DetectorSpacingX;

        vertices = zeros(3, 5);
        vertices(1, :) = 0.5*[-detector.width, -detector.width,...
            detector.width, detector.width, -detector.width];
        vertices(2, :) = repmat([geom.DistanceOriginDetector], 5, 1);
        vertices(3, :) = 0.5*[detector.height, -detector.height,...
            -detector.height, detector.height, detector.height];

        detector.vertices = vertices;
        plot3(h_ax, detector.vertices(1, :), detector.vertices(2, :),...
            detector.vertices(3, :), options.DetectorLineColor);
    end
end
