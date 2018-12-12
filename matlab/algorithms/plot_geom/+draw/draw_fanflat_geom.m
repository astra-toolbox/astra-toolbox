function [] = draw_fanflat_geom( h_ax, geom, options)
%%  draw_fanflat_geom.m
%   draw an astra cone beam projection geometry
%   param h_ax      handle to axis to draw into
%   param geom      the geometry to draw
%   param options               struct holding options for drawing
%       SourceMarker            Marker to use to mark the source
%       SourceMarkerColor       Color of the source marker
%       DetectorMarker          Marker to use to mark the source
%       DetectorMarkerColor     Color of the source marker
%       DetectorLineColor       Color of the outline of the detector
%       OpticalAxisColor        color for drawing the optical axis
%   date            28.06.2018
%   author          Tim Elberfeld
%                   imec VisionLab
%                   University of Antwerp
%%
    % convert to faux cone geometry so we don't have to write more code :)!
    cone_geom = astra_create_proj_geom('cone', geom.DetectorWidth,...
        geom.DetectorWidth, 1, geom.DetectorCount, geom.ProjectionAngles,...
        geom.DistanceOriginSource, geom.DistanceOriginDetector);
    
    draw.draw_cone_geom(h_ax, cone_geom, options);
end

