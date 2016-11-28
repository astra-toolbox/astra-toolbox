function [varargout] = astra_mex(varargin)
%------------------------------------------------------------------------
% Reference page in Help browser
%    <a href="matlab:docsearch('astra_mex' )">astra_mex</a>.
%------------------------------------------------------------------------
%------------------------------------------------------------------------
% This file is part of the ASTRA Toolbox
% 
% Copyright: 2010-2016, iMinds-Vision Lab, University of Antwerp
%            2014-2016, CWI, Amsterdam
% License: Open Source under GPLv3
% Contact: astra@uantwerpen.be
% Website: http://www.astra-toolbox.com/
%------------------------------------------------------------------------
if nargout == 0
    astra_mex_c(varargin{:});
    if exist('ans','var')
        varargout{1} = ans;
    end
else
    varargout = cell(1,nargout);
    [varargout{:}] = astra_mex_c(varargin{:});
end
