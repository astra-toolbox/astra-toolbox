function [] = draw_proj_geom(geom, varargin)
%% draw_proj_geom.m
% brief                         rendering function for astra geometries.
% param geom                    the geometry to plot. If geometry type
%                               is not supported, throws error
% ------------------------------
% optional parameters that can be provided as string value pairs:
%
% param RotationAxis            if specified, will change the drawn
%                               rotation axis to provided axis.
%                               Must be 3-vector. Default value is
%                               [NaN, NaN, NaN], (meaning do not draw).
% param RotationAxisOffset      if specified, will translate the drawn 
%                               rotation axis by the provided vector.
%                               Default = [0, 0, 0]
% param VectorIdx               index of the vector to visualize if geom
%                               is a vector geometry type. Default = 1
% param Color                   Color for all markers and lines if not 
%                               otherwise specified
% param DetectorMarker          marker for the detector locations.
%                               Default = '.'
% param DetectorMarkerColor     color specifier for the detector marker.
%                               Default = 'k'
% param DetectorLineColor       color for the lines drawing the detector
%                               outline
% param DetectorLineWidth       line width of detector rectangle
% param SourceMarker            marker for the source locations
% param SourceMarkerColor       color specifier for the source marker
% param SourceDistance          (only for parallel3d and parallel3d_vec)
%                               distance of source to origin
% param OpticalAxisColor        Color for drawing the optical axis
%
% date                          20.06.2018
% author                        Tim Elberfeld
%                               imec VisionLab
%                               University of Antwerp
%
% - last update                 07.11.2018
%%
    h_ax = gca;
    options = parseoptions(varargin);

    switch geom.type
        case 'parallel3d'
            disp('type: parallel3d')
            disp(['detector spacing: [' num2str(geom.DetectorSpacingX), ', '...
                num2str(geom.DetectorSpacingY) ']']);
            disp(['detector px: [' num2str(geom.DetectorRowCount), ', ' ...
                num2str(geom.DetectorColCount) ']']);
            disp(['angle lo: ' num2str(geom.ProjectionAngles(1))]);
            disp(['angle hi: ' num2str(geom.ProjectionAngles(end))]);
            disp(['# angles: ' num2str(numel(geom.ProjectionAngles))]);
            disp('DistanceOriginDetector inf');
            disp('DistanceOriginSource inf');

            draw.draw_parallel3d_geom(h_ax, geom, options);
        case 'parallel3d_vec'
            disp('type: parallel3d_vec')
            disp(['detector px: [' num2str(geom.DetectorRowCount), ', '...
                num2str(geom.DetectorColCount) ']']);
            disp(['# angles: ' num2str(size(geom.Vectors, 1))]);

            draw.draw_parallel3d_vec_geom(h_ax, geom, options);
        case 'cone'
            disp('type: cone');
            disp(['detector spacing: [' num2str(geom.DetectorSpacingX), ', '...
                num2str(geom.DetectorSpacingY) ']']);
            disp(['detector px: [' num2str(geom.DetectorRowCount), ', ' ...
                num2str(geom.DetectorColCount) ']']);
            disp(['angle lo: ' num2str(geom.ProjectionAngles(1))]);
            disp(['angle hi: ' num2str(geom.ProjectionAngles(end))]);
            disp(['# angles: ' num2str(numel(geom.ProjectionAngles))]);
            disp(['DistanceOriginDetector ' num2str(geom.DistanceOriginDetector)]);
            disp(['DistanceOriginSource ' num2str(geom.DistanceOriginSource)]);

            draw.draw_cone_geom(h_ax, geom, options);
        case 'cone_vec'
            disp('type: cone_vec');
            disp(['detector px: [' num2str(geom.DetectorRowCount), ', ' ...
                num2str(geom.DetectorColCount) ']']);
            disp(['# angles: ' num2str(size(geom.Vectors, 1))]);

            draw.draw_cone_vec_geom(h_ax, geom, options);
        case 'fanflat'
            disp('type: fanflat');
            disp(['detector px: ' num2str(geom.DetectorCount)]);
            disp(['angle lo: ' num2str(geom.ProjectionAngles(1))]);
            disp(['angle hi: ' num2str(geom.ProjectionAngles(end))]);
            disp(['# angles: ' num2str(numel(geom.ProjectionAngles))]);
            disp(['DistanceOriginDetector '...
                num2str(geom.DistanceOriginDetector)]);
            disp(['DistanceOriginSource '...
                num2str(geom.DistanceOriginSource)]);

            draw.draw_fanflat_geom(h_ax, geom, options);
        case 'fanflat_vec'
            disp('type: fanflat_vec');
            disp(['detector px: ' num2str(geom.DetectorCount)]);
            disp(['# angles: ' num2str(size(geom.Vectors, 1))]);

            draw.draw_fanflat_vec_geom(h_ax, geom, options);
        otherwise
            error(['Unknown geometry type ' geom.type])
    end
    view(45, 25);    % gives nicer default view angle

    function [options] = parseoptions(input_args)
        % make an options struct
        options = struct;
        options.RotationAxis = [NaN, NaN, NaN];
        options.RotationAxisOffset = [0, 0, 0];
        options.VectorIdx = 1;
        options.Color = 'k';
        options.DetectorMarker = '.';
        options.DetectorMarkerColor = '';
        options.DetectorLineColor = '';
        options.DetectorLineWidth = 1;
        options.SourceMarker = '*';
        options.SourceMarkerColor = '';
        options.SourceDistance = 100;
        options.OpticalAxisColor = '';
        options = parseargs.parseargs(options, input_args{:});

        % if the color is still empty, replace by global color
        if strcmpi(options.DetectorMarkerColor , '')
            options.DetectorMarkerColor = options.Color;
        end
        if strcmpi(options.DetectorLineColor , '')
            options.DetectorLineColor = options.Color;
        end
        if strcmpi(options.SourceMarkerColor , '')
            options.SourceMarkerColor = options.Color;
        end
        if strcmpi(options.OpticalAxisColor , '')
            options.OpticalAxisColor = options.Color;
        end
    end
end
