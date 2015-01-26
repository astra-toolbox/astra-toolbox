function backProj = astra_create_backprojection_cuda(sinogramData, proj_geom, vol_geom)
    %--------------------------------------------------------------------------
    % backProj = astra_create_backprojection_cuda(sinogramData, proj_geom, vol_geom)
    % 
    % Creates a CUDA-based simple backprojection
    %
    % sinogramData: 2D matrix with projections stored row-based
    % theta: projection angles, length should be equal to the number of rows in
    % sinogramData
    % reconstructionSize: vector with length 2 with the row and column count of
    % the reconstruction image
    % backProj: 2D back projection from sinogram data
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

    recon_id = astra_mex_data2d('create', '-vol', vol_geom, 0);
    sinogram_id = astra_mex_data2d('create', '-sino', proj_geom, sinogramData);

    cfg = astra_struct('BP_CUDA');
    cfg.ProjectionDataId = sinogram_id;
    cfg.ReconstructionDataId = recon_id;

    alg_id = astra_mex_algorithm('create', cfg);
    astra_mex_algorithm('run', alg_id);
    backProj = astra_mex_data2d('get', recon_id);

    astra_mex_data2d('delete', sinogram_id);
    astra_mex_data2d('delete', recon_id);
    astra_mex_algorithm('delete', alg_id);
end
