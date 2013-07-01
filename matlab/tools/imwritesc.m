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
% This file is part of the
% All Scale Tomographic Reconstruction Antwerp Toolbox ("ASTRA-Toolbox")
%
% Copyright: iMinds-Vision Lab, University of Antwerp
% License: Open Source under GPLv3
% Contact: mailto:astra@ua.ac.be
% Website: http://astra.ua.ac.be
%------------------------------------------------------------------------
% $Id$

imwrite(imscale(in),filename);
