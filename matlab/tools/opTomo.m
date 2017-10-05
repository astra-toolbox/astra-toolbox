%OPTOMO Wrapper for ASTRA tomography projector
%
%   OP = OPTOMO(TYPE, PROJ_GEOM, VOL_GEOM) generates a Spot operator OP for
%   the ASTRA forward and backprojection operations. The string TYPE
%   determines the model used for the projections. Possible choices are:
%       TYPE:  * using the CPU
%                'line'   - use a line kernel
%                'linear' - use a Joseph kernel
%                'strip'  - use the strip kernel
%              * using the GPU
%                'cuda'  - use a Joseph kernel, on the GPU, currently using
%                          'cuda' is the only option in 3D.
%   The PROJ_GEOM and VOL_GEOM structures are projection and volume
%   geometries as used in the ASTRA toolbox.
%
%   OP = OPTOMO(TYPE, PROJ_GEOM, VOL_GEOM, GPU_INDEX) also specify the
%   index of the GPU that should be used, if multiple GPUs are present in
%   the host system. By default GPU_INDEX is 0.
%
%   Note: this code depends on the Matlab toolbox 
%   "Spot - A Linear-Operator Toolbox" which can be downloaded from
%   http://www.cs.ubc.ca/labs/scl/spot/
%--------------------------------------------------------------------------
% This file is part of the ASTRA Toolbox
% 
% Copyright: 2014-2016, CWI, Amsterdam
% License:   Open Source under GPLv3
% Author:    Folkert Bleichrodt
% Contact:   F.Bleichrodt@cwi.nl
% Website:   http://www.astra-toolbox.com/
%--------------------------------------------------------------------------

classdef opTomo < opSpot
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    properties ( Access = private )
        % multiplication function
        funHandle
        % ASTRA identifiers
        sino_id
        vol_id
        fp_alg_id
        bp_alg_id
        proj_id
        % ASTRA IDs handle
        astra_handle
    end % properties
    
    properties ( SetAccess = private, GetAccess = public )
        proj_size
        vol_size
        proj_geom
        vol_geom
    end % properties
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Methods - public
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % opTomo - constructor
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function op = opTomo(type, proj_geom, vol_geom, gpu_index)
            
            if nargin < 4 || isempty(gpu_index), gpu_index = 0; end
            
            proj_size = astra_geom_size(proj_geom);
            vol_size  = astra_geom_size(vol_geom);
            
            % construct operator
            op = op@opSpot('opTomo', prod(proj_size), prod(vol_size));
            
            % determine the dimension
            is2D = ~isfield(vol_geom, 'GridSliceCount');
            gpuEnabled = strcmpi(type, 'cuda');
            
            if is2D
                % create a projector
                proj_id = astra_create_projector(type, proj_geom, vol_geom);
                
                % create a function handle
                op.funHandle = @opTomo_intrnl2D;
                
                % Initialize ASTRA data objects.
                % projection data
                sino_id = astra_mex_data2d('create', '-sino', proj_geom, 0);
                % image data
                vol_id  = astra_mex_data2d('create', '-vol', vol_geom, 0);
                
                % Setup forward and back projection algorithms.
                if gpuEnabled
                    fp_alg = 'FP_CUDA';
                    bp_alg = 'BP_CUDA';
                else
                    fp_alg = 'FP';
                    bp_alg = 'BP';
                end
                
                % configuration for ASTRA fp algorithm
                cfg_fp = astra_struct(fp_alg);
                cfg_fp.ProjectorId      = proj_id;
                cfg_fp.ProjectionDataId = sino_id;
                cfg_fp.VolumeDataId     = vol_id;
                
                % configuration for ASTRA bp algorithm
                cfg_bp = astra_struct(bp_alg);
                cfg_bp.ProjectionDataId     = sino_id;
                cfg_bp.ProjectorId          = proj_id;
                cfg_bp.ReconstructionDataId = vol_id;
                
                % set GPU index
                if gpuEnabled
                    cfg_fp.option.GPUindex = gpu_index;
                    cfg_bp.option.GPUindex = gpu_index;
                end
                
                fp_alg_id = astra_mex_algorithm('create', cfg_fp);
                bp_alg_id = astra_mex_algorithm('create', cfg_bp);
                
                % Create handle to ASTRA objects, so they will be deleted
                % if opTomo is deleted.
                op.astra_handle = opTomo_helper_handle([sino_id, ...
                    vol_id, proj_id, fp_alg_id, bp_alg_id]);
                
                op.fp_alg_id   = fp_alg_id;
                op.bp_alg_id   = bp_alg_id;
                op.sino_id     = sino_id;
                op.vol_id      = vol_id;
                
            else
                % 3D
                % only gpu/cuda code for 3D
                if ~gpuEnabled
                    error(['Only type ' 39 'cuda' 39 ' is supported ' ...
                           'for 3D geometries.'])
                end

                % setup projector
                cfg = astra_struct('cuda3d');
                cfg.ProjectionGeometry = proj_geom;
                cfg.VolumeGeometry = vol_geom;
                cfg.option.GPUindex = gpu_index;

                % create projector
                op.proj_id = astra_mex_projector3d('create', cfg);
                % create handle to ASTRA object, for cleaning up
                op.astra_handle = opTomo_helper_handle(op.proj_id);
                
                % create a function handle
                op.funHandle = @opTomo_intrnl3D;
            end
      
            
            % pass object properties
            op.proj_size   = proj_size;
            op.vol_size    = vol_size;
            op.cflag       = false;
            op.sweepflag   = false;
            op.proj_geom   = proj_geom;
            op.vol_geom   = vol_geom;

        end % opTomo - constructor
        
    end % methods - public

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Methods - protected
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods( Access = protected )

        % multiplication
        function y = multiply(op,x,mode)
            
            % ASTRA cannot handle sparse vectors
            if issparse(x)
                x = full(x);
            end

            if isa(x, 'double')
                isdouble = true;
                x = single(x);
            else
                isdouble = false;
            end
            
            % the multiplication
            y = op.funHandle(op, x, mode);
            
            % make sure output is column vector
            y = y(:);

            if isdouble
                y = double(y);
            end
            
        end % multiply
        
    end % methods - protected
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Methods - private
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods( Access = private )
        
        % 2D projection code
        function y = opTomo_intrnl2D(op,x,mode)
                       
            if mode == 1              
                % x is passed as a vector, reshape it into an image.             
                x = reshape(x, op.vol_size);
                
                % Matlab data copied to ASTRA data
                astra_mex_data2d('store', op.vol_id, x);
                
                % forward projection
                astra_mex_algorithm('iterate', op.fp_alg_id);
                
                % retrieve Matlab array
                y = astra_mex_data2d('get_single', op.sino_id);
            else
                % x is passed as a vector, reshape it into a sinogram.
                x = reshape(x, op.proj_size);
                
                % Matlab data copied to ASTRA data
                astra_mex_data2d('store', op.sino_id, x);
                
                % backprojection
                astra_mex_algorithm('iterate', op.bp_alg_id);
                
                % retrieve Matlab array
                y = astra_mex_data2d('get_single', op.vol_id);
            end

        end % opTomo_intrnl2D
        
        
        % 3D projection code
        function y = opTomo_intrnl3D(op,x,mode)
            
            if mode == 1
                % x is passed as a vector, reshape it into an image
                x = reshape(x, op.vol_size);
                
                % forward projection
                y = astra_mex_direct('FP3D', op.proj_id, x);
            else
                % x is passed as a vector, reshape it into projection data
                x = reshape(x, op.proj_size);
                                
                y = astra_mex_direct('BP3D', op.proj_id, x);
            end 
        end % opTomo_intrnl3D
        
    end % methods - private
 
end % classdef
