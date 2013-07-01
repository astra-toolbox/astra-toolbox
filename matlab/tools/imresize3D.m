function out = imresize3D(in, s_out, method)

%------------------------------------------------------------------------
% out = imresize3D(in, s_out, method)
% 
% Resizes a 3-component image
%
% in: input image.
% s_out: 2 element array with the wanted image size, [rows columns].
% out: the resized 3-component image.
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

out = zeros(s_out);
for i = 1:size(in,3)
	out(:,:,i) = imresize(in(:,:,i), s_out(1:2), method);
end
