function [] = draw_cad_phantom(filename, magn, h_ax)
%% draw_cad_phantom.m
% brief             render an stl model into a 3d axis object
% param vol_geom    volume geometry describing the phantom
% param magn        magnification multiplier of the phantom. default = 1
% param h_ax        handle to axis to plot into
%
% date              02.07.2018
% author            Alice Presenti
%                   imec VisionLab
%                   University of Antwerp
% Modified by Tim Elberfeld
%%
    if nargin == 1
        magn = 1;
    end
    if nargin == 2
        h_ax = axes(gcf);
    end    

    [v,f,~,~] = stlRead(filename);
    m = mean(v); % to center the CAD model!
    for i=1:3
        v(:,i) = (v(:,i)- m(i)) .* magn;
    end
    object.vertices = v;
    object.faces = f;
    patch(h_ax, object,'FaceColor',       [0.8 0.8 1.0], ...
                       'EdgeColor',       'none',        ...
                       'FaceLighting',    'gouraud',     ...
                       'AmbientStrength', 0.15);
    % Add a camera light, and tone down the specular highlighting
    camlight('headlight');
    material('dull');
    hold off;

end