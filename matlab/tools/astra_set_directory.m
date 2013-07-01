function in = astra_set_directory(in)

%------------------------------------------------------------------------
% in = astra_set_directory(in)
% 
% Creates the directories present in the input path if they do not exist
% already
% 
% in: input path.
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

a = find(in == '/' | in == '\');
for i = 1:numel(a)
	if ~isdir(in(1:a(i)))
		mkdir(in(1:a(i)));
	end
end
