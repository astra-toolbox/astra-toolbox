% -----------------------------------------------------------------------
% This file is part of the ASTRA Toolbox
% 
% Copyright: 2010-2016, iMinds-Vision Lab, University of Antwerp
%            2014-2016, CWI, Amsterdam
% License: Open Source under GPLv3
% Contact: astra@uantwerpen.be
% Website: http://www.astra-toolbox.com/
% -----------------------------------------------------------------------

% This sample illustrates the use of bicubic interpolation to achieve
% forward projections, that are continuously differentiable with respect
% to the projection geometry parameters.
%
% For example, this enables tomographic alignment by projection-matching
% methods, where the acquisition geometry parameters are fitted by
% matching forward projections of preliminary object reconstructions
% to the measured tomographic data.


% Make hollow cube test object
N_obj = [256,256,256];
obj = zeros(N_obj);
obj(end/4+1:3*end/4, end/4+1:3*end/4, end/4+1:3*end/4) = ones(N_obj/2) + 0.2*rand(N_obj/2);
obj(3*end/8+1:5*end/8, 3*end/8+1:5*end/8, 3*end/8+1:5*end/8) = 0;


% Projection- and volume geometry
t = rand()*360;     % Random tomographic angle
proj_geom = astra_create_proj_geom('parallel3d', 1, 1, N_obj(2), N_obj(1), t * (pi/180));
%proj_geom = astra_create_proj_geom('cone', 2, 2, N_obj(2), N_obj(1), t * (pi/180), 2*N_obj(1), 2*N_obj(1));    % uncomment for cone-beam test case
vol_geom  = astra_create_vol_geom(N_obj);


% Compute forward projections
[~, fp_bilin] = astra_create_sino3d_cuda(obj, proj_geom, vol_geom);                                 % Bilinear texture-interpolation
[~, fp_bicubic] = astra_create_sino3d_cuda(obj, proj_geom, vol_geom, 'bicubic');                    % Bicubic interpolation
[~, fp_bicubic_dd1] = astra_create_sino3d_cuda(obj, proj_geom, vol_geom, 'bicubic_derivative_1');   % Bicubic derivative along dimension 1
[~, fp_bicubic_dd2] = astra_create_sino3d_cuda(obj, proj_geom, vol_geom, 'bicubic_derivative_2');   % Bicubic derivative along dimension 2


% Plot results
figure('name', 'Forward projection with default bilinear texture-interpolation'); 
imagesc(squeeze(fp_bilin));
colorbar;

figure('name', 'Forward projection with bicubic texture-interpolation'); 
imagesc(squeeze(fp_bicubic));
colorbar;

figure('name', 'Forward projection using bicubic differentiation along dimension 1'); 
imagesc(squeeze(fp_bicubic_dd1));
colorbar;

figure('name', 'Forward projection using bicubic differentiation along dimension 2'); 
imagesc(squeeze(fp_bicubic_dd2));
colorbar;
