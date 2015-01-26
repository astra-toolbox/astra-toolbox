%--------------------------------------------------------------------------
% This file is part of the ASTRA Toolbox
%
% Copyright: 2010-2015, iMinds-Vision Lab, University of Antwerp
%            2014-2015, CWI, Amsterdam
% License: Open Source under GPLv3
% Contact: astra@uantwerpen.be
% Website: http://sf.net/projects/astra-toolbox
%--------------------------------------------------------------------------

classdef OutputDefault < matlab.mixin.Copyable
	 
	% Default policy class for output for DART.
	
	properties (Access=public)
		
		directory				= '';		% SETTING: Directory to save output.
		pre						= '';		% SETTING: Prefix of output.

		save_images				= 'no';		% SETTING: Save the images. 'no', 'yes' (='S') OR {'S', 'I', 'Mask', 'P'}
		save_results			= 'no';		% SETTING: Save the results. 'yes', 'no' OR {'base', 'stats', 'settings', 'S', 'V', 'V0'}
		save_object				= 'no';		% SETTING: Save the DART object. {'no','yes'}

		save_interval			= 1;		% SETTING: # DART iteration between saves.
		save_images_interval	= [];		% SETTING: Overwrite interval for save_images.
		save_results_interval	= [];		% SETTING: Overwrite interval for save_results.
		save_object_interval	= [];		% SETTING: Overwrite interval for save_object.

		slices					= 1;		% SETTING: In case of 3D, which slices to save?		
		
		verbose					= 'no';		% SETTING: Verbose? {'no','yes'}
		
	end
	
	methods (Access=public)
		
		%------------------------------------------------------------------
		function pre_initial_iteration(this, ~)
			if strcmp(this.verbose,'yes')
				tic
				fprintf(1, 'initial iteration...');
			end
		end

		%------------------------------------------------------------------
		function post_initial_iteration(this, ~)
			if strcmp(this.verbose,'yes')
				t = toc;
				fprintf(1, 'done in %f s.\n', t);
			end
		end
		
		%------------------------------------------------------------------
		function pre_iteration(this, DART)
			if strcmp(this.verbose,'yes')
				tic;
				fprintf(1, '%s dart iteration %d...', this.pre, DART.iterationcount);
			end
		end
		
		%------------------------------------------------------------------
		function post_iteration(this, DART)

			% print output
			if strcmp(this.verbose,'yes')
				t = toc;
				s = DART.statistics.tostring(DART.stats);
				fprintf(1, 'done in %0.2fs %s.\n', t, s);
			end
			
			% save DART object	
			if do_object(this, DART)
				save(sprintf('%s%sobject_%i.mat', this.directory, this.pre, DART.iterationcount), '-v7.3', 'DART');					
			end
			
			% save .mat
			if do_results(this, DART)
				base = DART.base;
				stats = DART.stats;
				S = DART.S;
				V = DART.V;
				V0 = DART.V0;
				settings = DART.getsettings();
				if ~iscell(this.save_results) 
					save(sprintf('%s%sresults_%i.mat', this.directory, this.pre, DART.iterationcount), '-v7.3', 'base', 'stats', 'S', 'V', 'V0', 'settings');
				else
					string = [];
					for i = 1:numel(this.save_results)
						string = [string this.save_results{i} '|'];
					end
					save(sprintf('%s%sresults_%i.mat', this.directory, this.pre, DART.iterationcount), '-v7.3', '-regexp', string(1:end-1));
				end
			end	
			
			% save images
			if do_images(this, DART)
			
				if ~iscell(this.save_images) && strcmp(this.save_images, 'yes')
					output_image(this, DART, 'S')
				elseif iscell(this.save_images)
					for i = 1:numel(this.save_images)
						output_image(this, DART, this.save_images{i});
					end
				end
				
			end					
			
		end
		%------------------------------------------------------------------
		
	end
	
	%----------------------------------------------------------------------
	methods (Access=private)
		
		function output_image(this, DART, data)
			% 2D
			if numel(size(DART.S)) == 2 
				eval(['imwritesc(DART.' data ', sprintf(''%s%s' data '_%i.png'', this.directory, this.pre, DART.iterationcount))']);	
			% 3D
			elseif numel(size(DART.S)) == 3  
				for slice = this.slices
					eval(['imwritesc(DART.' data '(:,:,slice), sprintf(''%s%s' data '_%i_slice%i.png'', this.directory, this.pre, DART.iterationcount, slice))']);
				end
			end
		end
	
		%------------------------------------------------------------------
		function out = do_object(this, DART)
			if strcmp(this.save_object,'no')
				out = 0;
				return
			end			
			if numel(this.save_object_interval) == 0 && mod(DART.iterationcount, this.save_interval) == 0 
				out = 1;
			elseif mod(DART.iterationcount, this.save_object_interval) == 0 
				out = 1;
			else
				out = 0;
			end
		end				
		%------------------------------------------------------------------
		function out = do_results(this, DART)
			if strcmp(this.save_results,'no')
				out = 0;
				return
			end
			if numel(this.save_results_interval) == 0 && mod(DART.iterationcount, this.save_interval) == 0 
				out = 1;
			elseif mod(DART.iterationcount, this.save_results_interval) == 0 
				out = 1;
			else
				out = 0;
			end
		end		
		
		%------------------------------------------------------------------
		function out = do_images(this, DART)
			if numel(this.save_images_interval) == 0 && mod(DART.iterationcount, this.save_interval) == 0 
				out = 1;
			elseif mod(DART.iterationcount, this.save_images_interval) == 0 
				out = 1;
			else
				out = 0;				
			end
		end
		%------------------------------------------------------------------
		
	end
	
end

