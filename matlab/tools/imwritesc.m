function imwritesc(in, filename)

%------------------------------------------------------------------------
% imwritesc(in, filename)
%
% Rescale between zero and one and write image
%
% in: input image.
% filename: name of output image file.
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

imwrite(imscale(in),filename);
