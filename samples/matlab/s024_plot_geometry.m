%% s024_plot_geometry.m
% brief             example of usage for astra_plot_geom command
% - last update     16.11.2018
% -----------------------------------------------------------------------
% This file is part of the ASTRA Toolbox
% 
% Copyright: 2010-2018, imec Vision Lab, University of Antwerp
%            2014-2018, CWI, Amsterdam
% License: Open Source under GPLv3
% Contact: astra@astra-toolbox.com
% Website: http://www.astra-toolbox.com/
% -----------------------------------------------------------------------
%%
close all;

if exist('astra_create_example_cone') ~= 2
    error('Please add astra/algorithms/plot_geom to your path to use this function')
end


% proj_geom = astra_create_example_cone('vec');
% proj_geom = astra_create_example_cone('normal');
proj_geom = astra_create_example_cone('helix');
% proj_geom = astra_create_example_parallel3d('vec');
% proj_geom = astra_create_example_fanflat('vec');
% proj_geom = astra_create_example_fanflat();
% proj_geom = astra_create_example_parallel3d();
% proj_geom = astra_create_example_cone('deform_vec');

astra_plot_geom(proj_geom);
hold on;

vol_magn = 20;
phantom_size = 5;
phantom_px = 1500;
vx_size = phantom_size / phantom_px; % voxel size
vol_geom = astra_create_vol_geom(phantom_px, phantom_px, phantom_px);
line_width = 1; % line width for phantom
astra_plot_geom(vol_geom, vx_size, 'Magnification', vol_magn,...
    'LineWidth', line_width, 'Color', 'r');

% this magnification is empirically chosen to fit the stl file
cad_magn = 900;
astra_plot_geom('bunny.stl', cad_magn);

hold off;
axis equal;
