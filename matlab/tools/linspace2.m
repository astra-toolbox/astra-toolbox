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
% Copyright: 2010-2015, iMinds-Vision Lab, University of Antwerp
%            2014-2015, CWI, Amsterdam
% License: Open Source under GPLv3
% Contact: astra@uantwerpen.be
% Website: http://sf.net/projects/astra-toolbox
%------------------------------------------------------------------------
% $Id$

out = linspace(a,b,c+1);
out = out(1:end-1);
