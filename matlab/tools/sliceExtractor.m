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
% This file is part of the
% All Scale Tomographic Reconstruction Antwerp Toolbox ("ASTRA-Toolbox")
%
% Copyright: iMinds-Vision Lab, University of Antwerp
% License: Open Source under GPLv3
% Contact: mailto:astra@ua.ac.be
% Website: http://astra.ua.ac.be
%------------------------------------------------------------------------
% $Id$

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
