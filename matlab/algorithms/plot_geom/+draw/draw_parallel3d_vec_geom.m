function [ ] = draw_parallel3d_vec_geom( h_ax, geom, options)
%%  draw_parallel3d_vec_geom.m
%   draw an astra parallel3d projection geometry
%   param h_ax          handle to axis to draw into
%   param geom          the geometry to draw
%   param options       struct containing the options for this function
%       idx             index of the vector to draw in more detail
%       SourceDistance  distance of the source to the origin
%
%   date                22.06.2018
%   author              Tim Elberfeld
%                       imec VisionLab
%                       University of Antwerp
%%
    vectors = geom.Vectors;

    % source
    xray_source = vectors(:, 1:3)*options.SourceDistance;

    % center of detector
    detector_center = vectors(:, 4:6);

    % draw the points and connect with lines
    hold on;
    num_angles = size(vectors, 1);
    scatter3(h_ax, xray_source(:, 1), xray_source(:, 2),...
        xray_source(:, 3), options.SourceMarkerColor,...
        options.SourceMarker);
    scatter3(h_ax, detector_center(:, 1), detector_center(:, 2),...
        detector_center(:, 3), 'k.');

    detector = struct;
    detector.u = vectors(options.VectorIdx, 7:9);
    detector.v = vectors(options.VectorIdx, 10:12);
    detector.height = geom.DetectorColCount;
    detector.width = geom.DetectorRowCount;
    detector.origin = detector_center(options.VectorIdx, :);
    detector.vertices = draw.draw_detector_vec(h_ax, detector, options);

    connect_source_detector(h_ax, detector, detector_center, xray_source,...
        options);
    
    % rotation axis will be roughly as long as the source detector distance
    distances = eucl_dist3d(detector_center, xray_source);
    mean_sdd = mean(distances(:)); % mean source detector distance
    draw_rotation_axis(h_ax, mean_sdd, options);
    
    text(h_ax, xray_source(options.VectorIdx, 1),...
        xray_source(options.VectorIdx, 2), ...
        xray_source(options.VectorIdx, 3), 'x-ray source');
    text(h_ax, detector_center(options.VectorIdx, 1),...
        detector_center(options.VectorIdx, 2),...
        detector_center(options.VectorIdx, 3), 'detector');
    hold off;

    function [] = connect_source_detector(h_ax, detector,...
            detector_center, xray_source, options)
        % connect source to detector origin
        idx = options.VectorIdx;
        line(h_ax, [detector_center(idx, 1), xray_source(idx, 1)],...
                   [detector_center(idx, 2), xray_source(idx, 2)],...
                   [detector_center(idx, 3), xray_source(idx, 3)],...
                   'Color', options.OpticalAxisColor, 'LineStyle', '--');

        % compute normal of detector plane
        n = null([detector.u; detector.v]);

        % connect source to detector edges
        for kk = 1:4
            a = detector.vertices(1, kk) - n(1)*xray_source(idx, 1);
            b = detector.vertices(2, kk) - n(2)*xray_source(idx, 2);
            c = detector.vertices(3, kk) - n(3)*xray_source(idx, 3);
            line(h_ax,[a, detector.vertices(1, kk)], ...
                      [b, detector.vertices(2, kk)],...
                      [c, detector.vertices(3, kk)],...
                      'Color', 'k',...
                      'LineStyle', ':');
        end
    end

    function [] = draw_rotation_axis(h_ax, scaling, options)
        % draw rotation axis
        rot_axis = options.RotationAxis;
        if(~isnan(rot_axis(1)))
            rot_axis = options.RotationAxis + options.RotationAxisOffset;
            origin = options.RotationAxisOffset;
            % origin of the geometry is assumed to be [0, 0, 0] always!
            line(h_ax, [origin(1), (scaling/2)*rot_axis(1)],...
                       [origin(2), (scaling/2)*rot_axis(2)],...
                       [origin(3), (scaling/2)*rot_axis(3)],...
                       'Color', options.OpticalAxisColor,...
                       'LineStyle', '-.');
            line(h_ax, [origin(1), -(scaling/2)*rot_axis(1)],...
                       [origin(2), -(scaling/2)*rot_axis(2)],...
                       [origin(3), -(scaling/2)*rot_axis(3)],...
                       'Color', options.OpticalAxisColor,...
                       'LineStyle', '-.');
        end
    end
end
