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
% This file is part of the ASTRA Toolbox
% 
% Copyright: 2010-2018, imec Vision Lab, University of Antwerp
%            2014-2018, CWI, Amsterdam
% License: Open Source under GPLv3
% Contact: astra@astra-toolbox.com
% Website: http://www.astra-toolbox.com/
%------------------------------------------------------------------------

sq = sqrt(1 - (r./a).^2);

res1 = 1 ./ besseli(m, alpha);
res2 = sq .^ m;
res3 = besseli(m, alpha .* sq);

res = res1 .* res2 .* res3;
