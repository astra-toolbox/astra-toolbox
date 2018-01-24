function slice = sliceExtractor(data, dir, slicenr)

%------------------------------------------------------------------------
% slice = sliceExtractor(data, dir, slicenr)
% 
% Outputs a specified slice from a three dimensional matrix/volume
%
% data: the 3D volume.
% dir: the direction in which the volume is sliced.
% slicenr: the index of the slice to retrieve.
% slice: 2D image matrix containing the slice.
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

slicenr = round(slicenr);

if strcmp(dir,'z')
	slice = squeeze(data(:,:,slicenr));
end
if strcmp(dir,'x')
	slice = squeeze(data(:,slicenr,:));
end
if strcmp(dir,'y')
	slice = squeeze(data(slicenr,:,:));
end
