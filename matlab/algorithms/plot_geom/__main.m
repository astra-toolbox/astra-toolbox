%% main.m
% brief             small tool to render astra geometries in matlab
%
% date              20.06.2018
% author            Tim Elberfeld
%                   imec VisionLab
%                   University of Antwerp
%
% - last update     09.07.2018
%%
% close all;

[h_gui, is_running] = create_gui_figure();

%proj_geom = create_example_cone('vec');
%proj_geom = create_example_cone('normal');
%proj_geom = create_example_cone('helix');
proj_geom = create_example_parallel3d('vec');
%proj_geom = create_example_fanflat('vec');
%proj_geom = create_example_fanflat();
%proj_geom = create_example_parallel3d();

draw_proj_geometry(proj_geom, h_gui);
hold on;

vol_magn = 10;
phantom_size = 5;
phantom_px = 1500;
vx_size = phantom_size / phantom_px; % voxel size
vol_geom = astra_create_vol_geom(phantom_px, phantom_px, phantom_px);
line_width = 1; % line width for phantom
draw_vol_geom(vol_geom, vx_size, h_gui, 'Magnification', vol_magn,...
    'LineWidth', line_width, 'Color', 'r');

% this magnification is empirically chosen to fit the stl file
cad_magn = 350;
draw_cad_phantom('stl/bunny.stl', cad_magn, h_gui);

proj_geom = create_example_cone('deform_vec');
draw_proj_geometry(proj_geom, h_gui, 'VectorIdx', 3, 'Color', 'b',...
    'RotationAxis', [0,0,1]);

hold off;
axis equal;
