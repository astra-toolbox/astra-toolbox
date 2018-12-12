function [] = draw_vol_geom( vol_geom, varargin)
%% draw_vol_geom.m
% brief                 rendering function for astra volume geometries
%                       describing a phantom.
% param vol_geom        volume geometry describing the phantom
% param vx_size         voxel size in unit of preference. must be same unit
%                       that was used to scale the projection geometry
% ------------------------------
% optional parameters that can be provided as string value pairs:
%
% param Magnification   magnification factor for the phantom. for small
%                       phantoms it might be necessary to scale the render 
%                       up as otherwise it won't show up in the plot.
%                       Default = 1
% param LineWidth       line width for the box wireframe. Default = 2
% param Color           color of the wireframe. Default = 'r'
%
% date                  20.06.2018
% author                Tim Elberfeld
%                       imec VisionLab
%                       University of Antwerp
%
% - last update         16.11.2018
%%
    h_ax = gca;

    if mod(size(varargin), 2) ~= 0
        vx_size = varargin{1};
        varargin = varargin(2:end); % consumed vx_size from arg list
    else
        vx_size = 1;
    end
    
    
    options = struct;
    options.Color = 'r';
    options.LineWidth = 2;
    options.Magnification = 1;
    options = parseargs.parseargs(options, varargin{:});

    hold on;
    phantom_height = vol_geom.GridRowCount * vx_size;
    phantom_width = vol_geom.GridColCount * vx_size;
    phantom_depth = vol_geom.GridSliceCount * vx_size;

    if isfield(vol_geom, 'option')
        minx = vol_geom.option.WindowMinX * vx_size;
        maxx = vol_geom.option.WindowMaxX * vx_size;

        miny = vol_geom.option.WindowMinY * vx_size;
        maxy = vol_geom.option.WindowMaxY * vx_size;

        minz = vol_geom.option.WindowMinZ * vx_size;
        maxz = vol_geom.option.WindowMaxZ * vx_size;
    else
        minx = phantom_width / 2 * vx_size;
        maxx = phantom_width / 2 * vx_size;

        miny = phantom_height / 2 * vx_size;
        maxy = phantom_height / 2 * vx_size;

        minz = phantom_depth / 2 * vx_size;
        maxz = phantom_depth / 2 * vx_size;
    end

    xx_phantom = options.Magnification*[minx, minx, minx, minx, maxx, maxx, maxx, maxx];
    yy_phantom = options.Magnification*[miny, miny, maxy, maxy, miny, miny, maxy, maxy];
    zz_phantom = options.Magnification*[minz, maxz, minz, maxz, minz, maxz, minz, maxz];

    face1 = [xx_phantom(1:4); yy_phantom(1:4); zz_phantom(1:4)];
    face2 = [[xx_phantom(1:2), xx_phantom(5:6)];...
             [yy_phantom(1:2), yy_phantom(5:6)];...
             [zz_phantom(1:2), zz_phantom(5:6)]];
    face3 = [[xx_phantom(3:4), xx_phantom(7:8)];...
             [yy_phantom(3:4), yy_phantom(7:8)];...
             [zz_phantom(3:4), zz_phantom(7:8)]];
    face4 = [[xx_phantom(5:6), xx_phantom(7:8)];...
             [yy_phantom(5:6), yy_phantom(7:8)];...
             [zz_phantom(5:6), zz_phantom(7:8)]];

    % as we draw only a wire frame, we need only to draw 4 of the faces
    draw_face(h_ax, face1, options);
    draw_face(h_ax, face2, options);
    draw_face(h_ax, face3, options);
    draw_face(h_ax, face4, options);

    hold off;

    function [] = draw_face(h_ax, face_coords, options)
        line(h_ax, face_coords(1, 1:2), face_coords(2, 1:2),...
                   face_coords(3, 1:2), 'LineWidth', options.LineWidth,...
                   'Color', options.Color);
        line(h_ax, face_coords(1, 3:4), face_coords(2, 3:4),...
                   face_coords(3, 3:4), 'LineWidth', options.LineWidth,...
                   'Color', options.Color);
        line(h_ax, [face_coords(1, 4),face_coords(1, 2)],...
                   [face_coords(2, 4),face_coords(2, 2)],...
                   [face_coords(3, 4),face_coords(3, 2)],...
                    'LineWidth', options.LineWidth, 'Color', options.Color);
        line(h_ax, [face_coords(1, 1),face_coords(1, 3)],...
                   [face_coords(2, 1),face_coords(2, 3)],...
                   [face_coords(3, 1),face_coords(3, 3)],...
                   'LineWidth', options.LineWidth, 'Color', options.Color);
    end
end
