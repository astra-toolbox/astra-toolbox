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
% Copyright: 2010-2015, iMinds-Vision Lab, University of Antwerp
%            2014-2015, CWI, Amsterdam
% License: Open Source under GPLv3
% Contact: astra@uantwerpen.be
% Website: http://sf.net/projects/astra-toolbox
%------------------------------------------------------------------------
% $Id$

out = zeros(s_out);
for i = 1:size(in,3)
	out(:,:,i) = imresize(in(:,:,i), s_out(1:2), method);
end
