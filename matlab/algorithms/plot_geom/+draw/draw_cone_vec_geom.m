function [] = draw_cone_vec_geom(h_ax, geom, options)
%%  draw_cone_vec_geom.m
%   draw an astra cone beam vectorized projection geometry
%   param h_ax                  handle to axis to draw into
%   param geom                  the geometry to draw
%   param options               struct holding options for drawing
%       RotationAxis            if specified, will change the rotation axis
%                               from z-axis to provided axis. 
%                               Must be 3-vector. Default value [0, 0, 1]
%       VectorIdx               the index of the angle to draw
%       Color                   global color for all lines and markers.
%                               is overwritten by specialized color options
%       SourceMarker            Marker to use to mark the source
%       SourceMarkerColor       Color of the source marker
%       DetectorMarker          Marker to use to mark the source
%       DetectorMarkerColor     Color of the source marker
%       DetectorLineColor       Color of the outline of the detector
%       OpticalAxisColor        color for drawing the optical axis
%
%   date                        21.06.2018
%   author                      Tim Elberfeld, Van Nguyen
%                               imec VisionLab
%                               University of Antwerp
%
% - last update                 07.11.2018
%%
    vectors = geom.Vectors;

    % source
    xray_source = vectors(:, 1:3);
    % center of detector
    detector_center = vectors(:, 4:6);

    % draw the points and connect with lines
    hold on;
    num_angles = size(vectors, 1);
    s_source = scatter3(h_ax, xray_source(:, 1), xray_source(:, 2),...
        xray_source(:, 3));
    s_source.Marker = options.SourceMarker;
    s_source.MarkerEdgeColor = options.SourceMarkerColor;

    s_det = scatter3(h_ax, detector_center(:, 1),...
        detector_center(:, 2), detector_center(:, 3));
    s_det.MarkerEdgeColor = options.DetectorMarkerColor;
    s_det.Marker = options.DetectorMarker;

    detector = struct;
    detector.u = vectors(options.VectorIdx, 7:9);
    detector.v = vectors(options.VectorIdx, 10:12);
    detector.height = geom.DetectorColCount;
    detector.width = geom.DetectorRowCount;
    detector.origin = detector_center(options.VectorIdx, :);

    vertices = draw.draw_detector_vec(h_ax, detector, options);
    connect_source_detector(h_ax, vertices, detector_center, ...
        xray_source, options);
    
    % rotation axis will be roughly as long as the source detector distance
    distances = eucl_dist3d(detector_center, xray_source);
    mean_sdd = mean(distances(:)); % mean source detector distance    
    draw_rotation_axis(h_ax, mean_sdd, options);
    
    text(h_ax, xray_source(options.VectorIdx, 1),...
        xray_source(options.VectorIdx, 2),...
        xray_source(options.VectorIdx, 3), 'x-ray source');
    text(h_ax, detector_center(options.VectorIdx, 1),...
        detector_center(options.VectorIdx, 2),...
        detector_center(options.VectorIdx, 3), 'detector');
    hold off;

    function [] = connect_source_detector(h_ax, vertices,...
            detector_center,  xray_source, options)
        % connect source to detector origin
        idx = options.VectorIdx;
        line(h_ax, [detector_center(idx, 1), xray_source(idx, 1)],...
                   [detector_center(idx, 2), xray_source(idx, 2)],...
                   [detector_center(idx, 3), xray_source(idx, 3)],...
                   'Color', options.OpticalAxisColor, 'LineStyle', '--');       
        
        % connect source to detector edges
        for kk = 1:4
            line(h_ax,[xray_source(idx, 1), vertices(1, kk)], ...
                      [xray_source(idx, 2), vertices(2, kk)],...
                      [xray_source(idx, 3), vertices(3, kk)],...
                      'Color', 'k', 'LineStyle', ':');
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
