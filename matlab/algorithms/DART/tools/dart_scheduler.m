%--------------------------------------------------------------------------
% This file is part of the ASTRA Toolbox
%
% Copyright: 2010-2022, imec Vision Lab, University of Antwerp
%            2014-2022, CWI, Amsterdam
% License: Open Source under GPLv3
% Contact: astra@astra-toolbox.com
% Website: http://www.astra-toolbox.com/
%--------------------------------------------------------------------------

function dart_scheduler(D_tmpl, iterations, settings)


for base_index = 1:numel(settings.base_loop)
	
	% create new DART object
	D = DART(settings.base_loop{base_index});

	dart_scheduler1D(D, iterations, settings.parameter1);
	
	% copy from templates
	D.tomography = D_tmpl.tomography;
	D.smoothing = D_tmpl.smoothing;
	D.segmentation = D_tmpl.segmentation;
	D.masking = D_tmpl.masking;
	D.statistics = D_tmpl.statistics;
	D.output = D_tmpl.output;	
	
	% set output options
	D.output = OutputScheduler();
	D.output.directory = output_folder{base_index};
	
	% run DART
	D = D.initialize();
	D = D.iterate(iterations);
	
end

end

function dart_scheduler1d(D, iterations, parameter_loop1)

end










