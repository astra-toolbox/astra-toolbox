classdef astra_projector_handle < handle
    %ASTRA_PROJECTOR_HANDLE Handle class around an astra_mex_projector id
    %   Automatically deletes the projector when deleted.

    %------------------------------------------------------------------------
    % This file is part of the
    % All Scale Tomographic Reconstruction Antwerp Toolbox ("ASTRA-Toolbox")
    %
    % Copyright: iMinds-Vision Lab, University of Antwerp
    % License: Open Source under GPLv3
    % Contact: mailto:astra@ua.ac.be
    % Website: http://astra.ua.ac.be
    %------------------------------------------------------------------------

    properties
        id
    end

    methods
        function obj = astra_projector_handle(proj_id)
            obj.id = proj_id;
        end
        function delete(obj)
            astra_mex_projector('delete', obj.id);
        end
    end

end

