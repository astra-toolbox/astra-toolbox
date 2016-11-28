function out = imscale(in)

%------------------------------------------------------------------------
% out = imscale(in)
% 
% Rescales the image values between zero and one.
%
% in: input image.
% out: scaled output image.
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

mi = min(in(:));
ma = max(in(:));
if (ma-mi) == 0
	out = zeros(size(in));
else
	out = (in - mi) / (ma - mi);
end
