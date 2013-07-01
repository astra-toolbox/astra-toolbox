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
% This file is part of the
% All Scale Tomographic Reconstruction Antwerp Toolbox ("ASTRA-Toolbox")
%
% Copyright: iMinds-Vision Lab, University of Antwerp
% License: Open Source under GPLv3
% Contact: mailto:astra@ua.ac.be
% Website: http://astra.ua.ac.be
%------------------------------------------------------------------------
% $Id$

mi = min(in(:));
ma = max(in(:));
if (ma-mi) == 0
	out = zeros(size(in));
else
	out = (in - mi) / (ma - mi);
end
