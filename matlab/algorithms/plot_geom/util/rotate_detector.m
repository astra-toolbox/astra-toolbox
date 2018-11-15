function [ vectors_rot ] = rotate_detector(vectors, rot_angles)
%%  rotate the detector part of the vectors of a vector geometry
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
    vectors_rot = vectors;
    vectors_rot(:,  1: 3) = rotate_euler3d(vectors(:,  1: 3), rot_angles);
    vectors_rot(:,  4: 6) = rotate_euler3d(vectors(:,  4: 6), rot_angles);
    vectors_rot(:,  7: 9) = rotate_euler3d(vectors(:,  7: 9), rot_angles);
    vectors_rot(:, 10:12) = rotate_euler3d(vectors(:, 10:12), rot_angles);
end
