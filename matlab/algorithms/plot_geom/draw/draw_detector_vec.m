function [ vertices] = draw_detector_vec( h_ax, detector, options)
%%  draw_detector_vec.m
%   draw a detector for a vector geometry
%   param h_ax                  handle to axis to draw into
%   param detector              the struct specifying the detector
%         .origin               3d coordinates of the detector origin
%         .u                    vector pointing from detector origin to
%                               pixel (1,0)
%         .v                    vector pointing from detector origin to
%                               pixel (0,1)
%         .width                width of the detector (number of px in u
%                               direction)
%         .height               height of the detector (number of px in v
%                               direction)
%   param options               struct with options
%         .Color                Color for the line work
%         .DetectorLineColor    Color of the detector rectangle outline
%   return                      The vertices of the detector rectangle that
%                               were plotted
%
%   date                        09.07.2018
%   author                      Tim Elberfeld
%                               imec VisionLab
%                               University of Antwerp
%%
    % draw the detector rectangle
    vertices = zeros(3, 5);
    vertices(:, 1) = detector.origin - detector.u * detector.width / 2 + ...
        detector.v * detector.height / 2;
    vertices(:, 2) = detector.origin + detector.u * detector.width / 2 + ...
        detector.v * detector.height / 2;
    vertices(:, 3) = detector.origin + detector.u * detector.width / 2 - ...
        detector.v * detector.height / 2;
    vertices(:, 4) = detector.origin - detector.u * detector.width / 2 - ...
        detector.v * detector.height / 2;
    vertices(:, 5) = vertices(:, 1);

    detector.vertices = vertices;
    plot3(h_ax, detector.vertices(1, :), detector.vertices(2, :),...
        detector.vertices(3, :), options.DetectorLineColor);

end
