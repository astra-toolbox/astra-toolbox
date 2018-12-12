function [] = astra_plot_geom(geometry, varargin)
%--------------------------------------------------------------------------
% [] = astra_plot_geometry(geometry, varargin)
%
% plot an astra geometry
%
% geometry: any astra geometry, either volume geometry, projection
%           geometry or an *.stl file (powered by stlRead).
% varargin: supports a variable number of (ordered and unordered)
%           arguments.
%
%           the remaining arguments depend on the input:
%           if 'geometry' is
%           - a volume geometry
%             vx_size               voxel size in unit of preference. Must
%                                   be same unit that was used to scale the
%                                   projection geometry.
%            and as unorderd string-value-pairs
%             Magnification         magnification factor for the phantom.
%                                   For small phantoms it might be
%                                   necessary to scale the render up as 
%                                   otherwise it won't show up in the plot.
%                                   Default = 1
%             LineWidth             line width for the box wireframe.
%                                   Default = 2
%             Color                 color of the wireframe. Default = 'r'
%                  
%           - a projection geometry (as unordered string-value-pairs)
%             RotationAxis          if specified, will change the drawn
%                                   rotation axis to provided axis.
%                                   Must be 3-vector. Default value is
%                                   [NaN, NaN, NaN], (meaning do not draw).
%             RotationAxisOffset    if specified, will translate the drawn 
%                                   rotation axis by the provided vector.
%                                   Default = [0, 0, 0]
%             VectorIdx             index of the vector to visualize if 
%                                   vector geometry type. Default = 1
%             Color                 Color for all markers and lines if not 
%                                   otherwise specified
%             DetectorMarker        marker for the detector locations.
%                                   Default = '.'
%             DetectorMarkerColor   color specifier for the detector marker.
%                                   Default = 'k'
%             DetectorLineColor     color for the lines drawing the 
%                                   detector outline
%             DetectorLineWidth     line width of detector rectangle
%             SourceMarker          marker for the source locations
%             SourceMarkerColor     color specifier for the source marker
%             SourceDistance        (only for parallel3d and parallel3d_vec)
%                                   distance of source to origin
%             OpticalAxisColor      Color for drawing the optical axis
%                  
%           - a path to an *.stl file
%             magn    -             magnification factor for vertices in
%                                   CAD file. Default value = 1
%
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
% This file is part of the ASTRA Toolbox
%
% Copyright: 2010-2018, imec Vision Lab, University of Antwerp
%            2014-2018, CWI, Amsterdam
% License: Open Source under GPLv3
% Contact: astra@astra-toolbox.com
% Website: http://www.astra-toolbox.com/
%--------------------------------------------------------------------------
    if exist('astra_create_example_cone') ~= 2
        error('Please add astra/algorithms/plot_geom to your path to use this function')
    end

    if is_vol_geom(geometry)
        draw.draw_vol_geom(geometry, varargin{:});
    elseif is_proj_geom(geometry)
        draw.draw_proj_geom(geometry, varargin{:});
    elseif ischar(geometry) % assume 'geometry' is a path to a CAD file
        draw.draw_cad_phantom(geometry, varargin{:});
    end

    % ---- helper functions ----
    function [ res ] = is_vol_geom(geom)
        res = false;
        if sum(isfield(geom, {'GridRowCount', 'GridColCount'})) == 2
            res = true;
        end
    end

    function [ res ] = is_proj_geom(geom)
        res = false;
        if isfield(geom, 'type')
            res = true;
        end
    end
end
