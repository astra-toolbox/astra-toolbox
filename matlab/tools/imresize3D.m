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
% This file is part of the ASTRA Toolbox
% 
% Copyright: 2010-2018, imec Vision Lab, University of Antwerp
%            2014-2018, CWI, Amsterdam
% License: Open Source under GPLv3
% Contact: astra@astra-toolbox.com
% Website: http://www.astra-toolbox.com/
%------------------------------------------------------------------------

out = zeros(s_out);
for i = 1:size(in,3)
	out(:,:,i) = imresize(in(:,:,i), s_out(1:2), method);
end
