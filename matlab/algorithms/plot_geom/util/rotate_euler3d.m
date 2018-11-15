function [ vectors_rot ] = rotate_euler3d(vectors, rot_angles)
%%  rotate some vectors by euler angles around an axis
%   param vectors           - vectors to transform
%   param rot_angles        - rotation euler angles
%
%   return vectors_rot      - copy of input vectors, but rotated
%
%   date            09.07.2018
%   author          Van Nguyen, Tim Elberfeld
%                   imec VisionLab
%                   University of Antwerp
%   last mod        07.11.2018
%%
    roll = rot_angles(1);
    yaw = rot_angles(2);
    pitch = rot_angles(3);
    roll_mat = [1 0 0; ...
                0 cos(roll) -sin(roll); ...
                0 sin(roll) cos(roll)];

    yaw_mat = [cos(yaw) 0 -sin(yaw); ...
               0 1 0; ...
               sin(yaw) 0 cos(yaw)];

    pitch_mat = [cos(pitch) -sin(pitch) 0; ...
                sin(pitch) cos(pitch) 0; ...
                0 0 1];
    % simulate rotation and translation of the DETECTOR
    rot_mat = roll_mat * yaw_mat * pitch_mat;
    vectors_rot = vectors;

    for i = 1:size(vectors, 1)
        vectors_rot(i, :) = (rot_mat * vectors(i, :)')';
    end
end
