function sinogram_out = astra_add_noise_to_sino(sinogram_in,I0)

%--------------------------------------------------------------------------
% sinogram_out = astra_add_noise_to_sino(sinogram_in,I0)
%
% Add poisson noise to a sinogram.
%
% sinogram_in: input sinogram, can be either MATLAB-data or an
% astra-identifier.  In the latter case, this operation is inplace and the
% result will also be stored in this data object.
% I0: background intensity, used to set noise level, lower equals more
% noise
% sinogram_out: output sinogram in MATLAB-data.
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
% This file is part of the ASTRA Toolbox
% 
% Copyright: 2010-2015, iMinds-Vision Lab, University of Antwerp
%            2014-2015, CWI, Amsterdam
% License: Open Source under GPLv3
% Contact: astra@uantwerpen.be
% Website: http://sf.net/projects/astra-toolbox
%--------------------------------------------------------------------------
% $Id$

if numel(sinogram_in) == 1
	sinogramRaw = astra_mex_data2d('get', sinogram_in);
else
	sinogramRaw = sinogram_in;
end

% scale to [0,1]
max_sinogramRaw = max(sinogramRaw(:));
sinogramRawScaled = sinogramRaw ./ max_sinogramRaw;
% to detector count
sinogramCT = I0 * exp(-sinogramRawScaled);
% add poison noise
sinogramCT_A = sinogramCT * 1e-12;
sinogramCT_B = double(imnoise(sinogramCT_A, 'poisson'));
sinogramCT_C = sinogramCT_B * 1e12;
% to density
sinogramCT_D = sinogramCT_C / I0;
sinogram_out = -max_sinogramRaw * log(sinogramCT_D);

if numel(sinogram_in) == 1
	astra_mex_data2d('store', sinogram_in, sinogram_out);
end
