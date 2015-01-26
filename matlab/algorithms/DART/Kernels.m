%--------------------------------------------------------------------------
% This file is part of the ASTRA Toolbox
%
% Copyright: 2010-2015, iMinds-Vision Lab, University of Antwerp
%            2014-2015, CWI, Amsterdam
% License: Open Source under GPLv3
% Contact: astra@uantwerpen.be
% Website: http://sf.net/projects/astra-toolbox
%--------------------------------------------------------------------------

classdef Kernels
	%KERNELS Summary of this class goes here
	%   Detailed explanation goes here
	
	properties
		
	end
	
	methods(Static)

		function K = BinaryPixelKernel(radius, conn)
			
			if nargin < 2
				conn = 8;
			end
			
			% 2D, 4conn
			if conn == 4
				K = [0 1 0; 1 1 1; 0 1 0];
				for i = 2:radius
					K = conv2(K,K);
				end
				K = double(K >= 1);
			
			% 2D, 8conn
			elseif conn == 8
				K = ones(2*radius+1, 2*radius+1);
				
			% 3D, 6conn	
			elseif conn == 6
				K = zeros(3,3,3);
				K(:,:,1) = [0 0 0; 0 1 0; 0 0 0];
				K(:,:,2) = [0 1 0; 1 1 1; 0 1 0];
				K(:,:,3) = [0 0 0; 0 1 0; 0 0 0];
				for i = 2:radius
					K = convn(K,K);
				end
				K = double(K >= 1);				
				
			% 2D, 27conn
			elseif conn == 26
				K = ones(2*radius+1, 2*radius+1, 2*radius+1);
				
			else
				disp('Invalid conn')
			end
		end

		

	
		
		
	end
	
end

