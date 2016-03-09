% -----------------------------------------------------------------------
% This file is part of the ASTRA Toolbox
% 
% Copyright: 2010-2015, iMinds-Vision Lab, University of Antwerp
%            2014-2015, CWI, Amsterdam
% License: Open Source under GPLv3
% Contact: astra@uantwerpen.be
% Website: http://sf.net/projects/astra-toolbox
% -----------------------------------------------------------------------
addpath matlab/tools/
addpath bin/x64/Release/

vol_geom = astra_create_vol_geom(256, 256);
proj_geom = astra_create_proj_geom('parallel', 1.0, 384, linspace2(0,pi,90));
% vol_geom = astra_create_vol_geom(4,4);
% proj_geom = astra_create_proj_geom('parallel', 1.0, 10, linspace2(0,pi,4));

% For CPU-based algorithms, a "projector" object specifies the projection
% model used. In this case, we use the "strip" model.
proj_id = astra_create_projector('linear', proj_geom, vol_geom);

% Create a sinogram from a phantom
P = phantom(256);
% P = phantom(4);
[sinogram_id, sinogram] = astra_create_sino(P, proj_id);
% figure(1); imshow(P, []); axis image; axis off;
% figure(2); imshow(sinogram, []); axis image; axis off;
% imshow(sinogram, []);

% add sinogram noise
rng(123);
% sinogram = sinogram + randn(size(sinogram)) * 0.5;

astra_mex_data2d('delete', sinogram_id);

% We now re-create the sinogram data object as we would do when loading
% an external sinogram
sinogram_id = astra_mex_data2d('create', '-sino', proj_geom, sinogram);

% Create a data object for the reconstruction
rec_id = astra_mex_data2d('create', '-vol', vol_geom);

% Set up the parameters for a reconstruction algorithm using the CPU
% The main difference with the configuration of a GPU algorithm is the
% extra ProjectorId setting.
cfg = astra_struct('SART');
cfg.ReconstructionDataId = rec_id;
cfg.ProjectionDataId = sinogram_id;
cfg.ProjectorId = proj_id;
cfg.option.UseMinConstraint = 1;
cfg.option.MinConstraintValue = 0;
cfg.option.UseMaxConstraint = 1;
cfg.option.MaxConstraintValue = 1;
cfg.option.ClearRayLength = 1;
cfg.option.Alpha = 2;

% Available algorithms:
% ART, SART, SIRT, CGLS, FBP

%
% See the effect of ClearRayLength
%
iterations = [5, 10, 15, 20];
ray_lengths = [1, 0];

snr_res = {[]; []};

% run
for i_it=1:length(iterations)
  for i_rl=1:length(ray_lengths)
    
    cfg.option.ClearRayLength = ray_lengths(i_rl);
    alg_id = astra_mex_algorithm('create', cfg);
    astra_mex_algorithm('iterate', alg_id, iterations(i_it));
    astra_mex_algorithm('delete', alg_id);
    
    % Get the result
    rec = astra_mex_data2d('get', rec_id);

    snr_res{i_rl} = [snr_res{i_rl}, snr(P, (P-rec))];
  end
end

% plot
figure(1)
plot(iterations, snr_res{1}, '-r', iterations, snr_res{2}, '-g');
legend({'With ClearRayLength', 'Without ClearRayLength'}, ...
  'Location','SouthEast');
xlabel('iteration');
ylabel('SNR (db)');


% 
% See the effect of Alpha
%
alphas = [0.2, 0.5, 0.9, 1, 1.2, 1.5, 1.9, 2];
snr_res = [];

% run
for i_alpha=1:length(alphas)
    
  cfg.option.Alpha = alphas(i_alpha);
  cfg.option.ClearRayLength = 1;
  alg_id = astra_mex_algorithm('create', cfg);
  astra_mex_algorithm('iterate', alg_id, 20);
  astra_mex_algorithm('delete', alg_id);

  % Get the result
  rec = astra_mex_data2d('get', rec_id);

  snr_res = [snr_res, snr(P, (P-rec))];
end

% plot
figure(2)
plot(alphas, snr_res, '-r');
xlabel('Alpha');
ylabel('SNR (db)');

% Clean up. 
astra_mex_projector('delete', proj_id);
astra_mex_data2d('delete', sinogram_id);
astra_mex_data2d('delete', rec_id);

