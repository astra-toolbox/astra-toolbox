function [ proj_geom ] = create_example_cone(type)
%%  create_example_cone.m
%   create an example standard cone beam geometry
%   param type      type of geometry to create. provided as one of the
%                   following strings (not case sensitive):
%                   'normal'        -   standard (non vector) geometry
%                   'vec'           -   example vector geometry
%                   'helix'         -   helical trajectory vector geometry
%                   'deform_vec'    -   deformed vector geometry example, 
%                                       courtesy of Van Nguyen
%   return          proj_geom       -   the geometry that was created
%
%   date            21.06.2018
%   author          Tim Elberfeld, Alice Presenti, Van Nguyen
%                   imec VisionLab
%                   University of Antwerp
%   last mod        07.11.2018
%%
    if(nargin < 1)
        type = 'normal';
    end
    if strcmpi(type, 'normal')
        % first, give measurements in mm
        det_spacing = [0.035, 0.035];
        detector_px = [1200, 1200];
        angles = linspace2(0, 2*pi, 100);
        source_origin = 30;
        origin_det = 200;
        phantom_size = 5;

        phantom_px = 150; % voxels for the phantom
        vx_size = phantom_size / phantom_px; % voxel size

        % now express all measurements in terms of the voxel size
        det_spacing = det_spacing ./ vx_size;
        origin_det = origin_det ./ vx_size;
        source_origin = source_origin ./ vx_size;

        proj_geom = astra_create_proj_geom('cone',  det_spacing(1), ...
        det_spacing(2), detector_px(1), detector_px(2), angles,...
        source_origin, origin_det);

    elseif strcmpi(type, 'vec')
        proj_geom = create_standard_vec_geom();

    elseif strcmpi(type, 'deform_vec')
        proj_geom = create_deform_vec_geom();

    elseif strcmpi(type, 'helix')
        detRows = 220;
        detCols = 220;
        detPitchX = 10^-3 * 11; % microns to mm
        detPitchY = 10^-3 * 11; % microns to mm
        objSrcDist = 500; % mm
        rotStep = 3.6; % deg
        numAngles = 200;
        zTranslation = 0.5; % mm
        zDist = zTranslation * numAngles;
        vectors = zeros(numAngles, 12);
        translation = -zDist + (zTranslation * (0:(numAngles-1)));

        minAngle = 0; % just assume we start at 0
        maxAngle = (numAngles * rotStep)/180 * pi; % convert deg to rad
        angles = linspace(minAngle, maxAngle, 200);

        % source position per angle
        vectors(:, 1) = sin(angles) * objSrcDist;
        vectors(:, 2) = -cos(angles) * objSrcDist;
        vectors(:, 3) = translation;

        % detector position per angle
        vectors(:, 4) = 0;
        vectors(:, 5) = 0;
        vectors(:, 6) = translation;

        % vector from detector pixel (0,0) to (0,1)
        vectors(:, 7) = cos(angles);
        vectors(:, 8) = sin(angles);
        vectors(:, 9) = zeros(numAngles, 1);

        % vector from detector pixel (0,0) to (1, 0)
        vectors(:, 10) = zeros(numAngles, 1);
        vectors(:, 11) = zeros(numAngles, 1);
        vectors(:, 12) = ones(numAngles, 1);

        proj_geom = astra_create_proj_geom('cone_vec', detCols, detRows, vectors);

    else
        proj_geom = create_standard_vec_geom();
    end

    function [proj_geom, z_axis] = create_standard_vec_geom()
        % geometry settings taken from code from Alice Presenti
        settings = struct;
        settings.detectorPixelSize = 0.0748;
        % detector size and number of projections
        settings.projectionSize = [1536 1944 21];
        settings.SOD = 679.238020;  %[mm]
        settings.SDD = 791.365618;  %[mm]
        settings.voxelSize = 1;     %[mm]
        settings.gamma = linspace(0,300,21)*pi/180;

        S0 = zeros(settings.projectionSize(3), 12);
        Sorig = [-settings.SOD, 0, 0,... % the ray origin vector (source)
                 (settings.SDD-settings.SOD), 0, 0,... % detector center
                 0, -settings.detectorPixelSize, 0,... % detector u axis
                 0, 0, settings.detectorPixelSize]; % detector v axis

        z_axis = [0, 0, 1];
        for i = 1:settings.projectionSize(3)
            S0(i,:) = Sorig(:);
            S0(i,1:3) = rotate_around3d(S0(i,1:3),...
                z_axis, settings.gamma(i));
            S0(i,4:6) = rotate_around3d(S0(i,4:6), z_axis,...
                settings.gamma(i));
            S0(i,7:9) = rotate_around3d(S0(i,7:9), z_axis,...
                settings.gamma(i));
            S0(i,10:12) = rotate_around3d(S0(i,10:12),...
                z_axis, settings.gamma(i));
        end
        proj_geom = astra_create_proj_geom('cone_vec', ...
            settings.projectionSize(2), settings.projectionSize(1), S0);
    end

    function [proj_geom, z_axis] = create_deform_vec_geom()
        settings = struct;
        settings.detectorPixelSize = 0.0748;
        settings.projectionSize = [1536 1944 21];
        settings.SOD = 679.238020;  %[mm]
        settings.SDD = 791.365618;  %[mm]
        settings.voxelSize = 1;     %[mm]
        settings.gamma = linspace(0,300,21)*pi/180;

        S0 = zeros(settings.projectionSize(3), 12);
        Sorig = [-settings.SOD, 0, 0,... % the ray origin vector (source)
                 (settings.SDD-settings.SOD), 0, 0,... % detector center
                 0, -settings.detectorPixelSize, 0,... % detector u axis
                 0, 0, settings.detectorPixelSize]; % detector v axis

        z_axis = [0, 0, 1];
        for i = 1:settings.projectionSize(3)
            S0(i,:) = Sorig(:);
        end

        S0(:, 1:3) = translate_3d(S0(:, 1:3), [100, 150, 0]);
        S0(:, 4:6) = translate_3d(S0(:, 4:6), [100, 150, 0]);        
        
        S0 = rotate_detector(S0, [0.48,0.32,0]);
        S0 = magnify_proj(S0, 100);
        for i = 1:settings.projectionSize(3)
            S0(i,1:3) = rotate_around3d(S0(i,1:3), z_axis,...
                settings.gamma(i));
            S0(i,4:6) = rotate_around3d(S0(i,4:6), z_axis,...
                settings.gamma(i));
            S0(i,7:9) = rotate_around3d(S0(i,7:9), z_axis,...
                settings.gamma(i));
            S0(i,10:12) = rotate_around3d(S0(i,10:12), z_axis,...
                settings.gamma(i));
        end

        proj_geom = astra_create_proj_geom('cone_vec',...
            settings.projectionSize(2), settings.projectionSize(1), S0);
    end

end
