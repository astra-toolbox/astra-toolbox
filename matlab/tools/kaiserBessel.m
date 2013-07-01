function res = kaiserBessel(m,alpha,a,r)

%------------------------------------------------------------------------
% res = kaiserBessel(m,alpha,a,r)
% 
% Calculates the Kaiser windowing function
%
% a: length of the sequence.
% m: order.
% alpha: determines shape of window.
% r: input values for which to compute window value.
% res: the window values.
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

sq = sqrt(1 - (r./a).^2);

res1 = 1 ./ besseli(m, alpha);
res2 = sq .^ m;
res3 = besseli(m, alpha .* sq);

res = res1 .* res2 .* res3;
