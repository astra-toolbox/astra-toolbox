function Im = imreadgs(filename)

%------------------------------------------------------------------------
% Im = imreadgs(filename)
% 
% Reads an image and transforms it into a grayscale image consisting of
% doubles.
%
% filename: name of the image file.
% Im: a grayscale image in double.
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

Im = double(imread(filename));
if size(Im,3) > 1
	Im = Im(:,:,1);
end	
