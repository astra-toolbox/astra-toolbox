classdef astra_projector_handle < handle
    %ASTRA_PROJECTOR_HANDLE Handle class around an astra_mex_projector id
    %   Automatically deletes the projector when deleted.

    %------------------------------------------------------------------------
    % This file is part of the ASTRA Toolbox
    % 
    % Copyright: 2010-2016, iMinds-Vision Lab, University of Antwerp
    %            2014-2016, CWI, Amsterdam
    % License: Open Source under GPLv3
    % Contact: astra@uantwerpen.be
    % Website: http://www.astra-toolbox.com/
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

