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
% This file is part of the ASTRA Toolbox
% 
% Copyright: 2010-2018, imec Vision Lab, University of Antwerp
%            2014-2018, CWI, Amsterdam
% License: Open Source under GPLv3
% Contact: astra@astra-toolbox.com
% Website: http://www.astra-toolbox.com/
%------------------------------------------------------------------------

Im = double(imread(filename));
if size(Im,3) > 1
	Im = Im(:,:,1);
end	
