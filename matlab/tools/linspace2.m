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
% This file is part of the
% All Scale Tomographic Reconstruction Antwerp Toolbox ("ASTRA-Toolbox")
%
% Copyright: iMinds-Vision Lab, University of Antwerp
% License: Open Source under GPLv3
% Contact: mailto:astra@ua.ac.be
% Website: http://astra.ua.ac.be
%------------------------------------------------------------------------
% $Id$

out = linspace(a,b,c+1);
out = out(1:end-1);
