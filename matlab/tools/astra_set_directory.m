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
% This file is part of the ASTRA Toolbox
% 
% Copyright: 2010-2018, imec Vision Lab, University of Antwerp
%            2014-2018, CWI, Amsterdam
% License: Open Source under GPLv3
% Contact: astra@astra-toolbox.com
% Website: http://www.astra-toolbox.com/
%------------------------------------------------------------------------

a = find(in == '/' | in == '\');
for i = 1:numel(a)
	if ~isdir(in(1:a(i)))
		mkdir(in(1:a(i)));
	end
end
