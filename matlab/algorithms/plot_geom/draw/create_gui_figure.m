function [h_ax, running] = create_gui_figure()
%% create_gui_figure.m
% brief     gui for the geometry drawing functions
% return    h_ax        axis to draw into with draw_*() functions
%           running     state of the rotation, needs to be passed so that
%                       the callback will work
% date      20.06.2018
% author    Tim Elberfeld
%           imec VisionLab
%           University of Antwerp
%%
    h_figure = figure('name', 'geometry render', 'ButtonDownFcn', @toggle_rotation);
    h_ax = axes(h_figure);
    
    set(h_figure,'CloseRequestFcn', @handle_close_fig)

    xlabel('x axis')
    ylabel('y axis')
    zlabel('z axis')

    grid on
    box off
    axis vis3d
    axis equal
    view(0,0)

    running = false;
    do_rotation();

    function [] = handle_close_fig(h_figure,~)
        % this is necessary to stop the rotation before closing the figure
        if running
            toggle_rotation()
        end
        
        delete(h_figure);
    end

    function [] = toggle_rotation(~, ~)
        % toggle rotation state
        running = ~running;         
        if running
            view(0,0)
            do_rotation();
        else
            view(45,45);
        end
    end

    function [] = do_rotation()
        % rotate the rendered geometry around the origin
        camtarget([0,0,0]); % make origin the camera target and point around which to rotate
        while running
            camorbit(0.5,0,'camera')
            drawnow 
        end       
    end
end