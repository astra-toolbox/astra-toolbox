function out = linspace2(a,b,c)

%------------------------------------------------------------------------
% out = linspace2(a,b,c)
% 
% Generates linearly spaced vectors 
%
% a: lower limit.
% b: upper limit (exclusive).
% c: number of elements.
% out: linearly spaced vector.
%------------------------------------------------------------------------
%------------------------------------------------------------------------
% This file is part of the ASTRA Toolbox
% 
% Copyright: 2010-2018, imec Vision Lab, University of Antwerp
%            2014-2018, CWI, Amsterdam
% License: Open Source under GPLv3
% Contact: astra@astra-toolbox.com
% Website: http://www.astra-toolbox.com/
%------------------------------------------------------------------------

out = linspace(a,b,c+1);
out = out(1:end-1);
