function [varargout] = astra_mex_projector3d(varargin)
%------------------------------------------------------------------------
% Reference page in Help browser
%    <a href="matlab:docsearch('astra_mex_projector3d' )">astra_mex_projector3d</a>.
%------------------------------------------------------------------------
%------------------------------------------------------------------------
% This file is part of the
% All Scale Tomographic Reconstruction Antwerp Toolbox ("ASTRA-Toolbox")
%
% Copyright: iMinds-Vision Lab, University of Antwerp
% License: Open Source under GPLv3
% Contact: mailto:astra@ua.ac.be
% Website: http://astra.ua.ac.be
%------------------------------------------------------------------------
% $Id$
if nargout == 0
    astra_mex_projector3d_c(varargin{:});
    if exist('ans','var')
        varargout{1} = ans;
    end
else
    varargout = cell(1,nargout);
    [varargout{:}] = astra_mex_projector3d_c(varargin{:});
end
