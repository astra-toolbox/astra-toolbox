function [varargout] = astra_mex_log(varargin)
%------------------------------------------------------------------------
% Reference page in Help browser
%    <a href="matlab:docsearch('astra_mex_log' )">astra_mex_log</a>.
%------------------------------------------------------------------------
%------------------------------------------------------------------------
% This file is part of the ASTRA Toolbox
% 
% Copyright: 2010-2015, iMinds-Vision Lab, University of Antwerp
%            2014-2015, CWI, Amsterdam
% License: Open Source under GPLv3
% Contact: astra@uantwerpen.be
% Website: http://sf.net/projects/astra-toolbox
%------------------------------------------------------------------------
% $Id$
if size(varargin,2)==2 && (strcmp(varargin{1},'debug') || strcmp(varargin{1},'info') || strcmp(varargin{1},'warn') || strcmp(varargin{1},'error'))
    d = dbstack(1);
    if size(d,1)==0
        astra_mex_log_c(varargin{1},'Unknown',0,varargin{2})
    else
        astra_mex_log_c(varargin{1},d(1).file,d(1).line,varargin{2})
    end
else
    if nargout == 0
        astra_mex_log_c(varargin{:});
        if exist('ans','var')
            varargout{1} = ans;
        end
    else
        varargout = cell(1,nargout);
        [varargout{:}] = astra_mex_log_c(varargin{:});
    end
end