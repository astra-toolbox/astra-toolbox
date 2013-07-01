% This file is part of the
% All Scale Tomographic Reconstruction Antwerp Toolbox ("ASTRA-Toolbox")
%
% Copyright: iMinds-Vision Lab, University of Antwerp
% License: Open Source under GPLv3
% Contact: mailto:astra@ua.ac.be
% Website: http://astra.ua.ac.be
%
% Author of this DART Algorithm: Wim van Aarle


classdef TomographyDefault3D < IterativeTomography3D

	% Policy class for 3D tomography for DART.

	%----------------------------------------------------------------------
	properties (Access=public)
		t					= 5;		% SETTING: # ARMiterations, each DART iteration.
		t0					= 100;		% SETTING: # ARM iterations at DART initialization.
	end
	%----------------------------------------------------------------------

	methods

		%------------------------------------------------------------------
		function settings = getsettings(this)
			% Returns a structure containing all settings of this object.
			% >> settings = DART.tomography.getsettings();
%			settings = getsettings@IterativeTomography();
			settings.t					= this.t;
			settings.t0					= this.t0;
		end

		%------------------------------------------------------------------
		function initialize(this, DART)
			% Initializes this object. 
			% >> DART.tomography.initialize();
			this.proj_geom = DART.base.proj_geom;
			this.initialize@IterativeTomography3D();
		end
		
		%------------------------------------------------------------------
		function P = createForwardProjection(this, ~, volume)
			% Compute forward projection. 
			% >> DART.tomography.createForwardProjection(DART, volume);
			P = this.project_c(volume);
		end
		
		%------------------------------------------------------------------
		function I = createReconstruction(this, ~, sinogram, V0, mask)
			% Compute reconstruction (with mask).
			% >> DART.tomography.createReconstruction(DART, sinogram, V0, mask);			
			if strcmp(this.inner_circle,'yes')
				mask = ROIselectfull(mask, size(mask,1));
			end
			I = this.reconstruct_c(sinogram, V0, mask, this.t);
		end	
		
		%------------------------------------------------------------------
		function I = createInitialReconstruction(this, ~, sinogram)
			% Compute reconstruction (initial).
			% >> DART.tomography.createInitialReconstruction(DART, sinogram);						
			I = this.reconstruct_c(sinogram, [], [], this.t0);
			if strcmp(this.inner_circle,'yes')
				I = ROIselectfull(I, size(I,1));
			end
		end			
		%------------------------------------------------------------------
	
	end		
		
end

