function [rot_vec] = rotate_around3d(vec, ax, angle)
%% rotate_around.m
%   rotate a 3d vector around an axis by an angle
%   param vec:    3 x 1 vector to be rotated
%   param ax:     3 x 1 vector specifying the axis
%   param angle:  scalar, angle to rotate by
%   return:       rotated vector
%
%   date:         21.06.2018
%   author:       someone at VisionLab, modified by Tim Elberfeld
%                 imec VisionLab
%                 University of Antwerp
%%
    rot_vec = zeros(3, 1);

    rot_vec(1) = (ax(1) * ax(1) * (1-cos(angle)) + cos(angle)) * vec(1) +...
                 (ax(1) * ax(2) * (1-cos(angle)) - ax(3) * sin(angle)) * vec(2) +...
                 (ax(1) * ax(3) * (1-cos(angle)) + ax(2) * sin(angle)) * vec(3);
  
    rot_vec(2) = (ax(1) * ax(2) * (1-cos(angle)) + ax(3) * sin(angle)) * vec(1) +...
                 (ax(2) * ax(2) * (1-cos(angle)) + cos(angle)) * vec(2) +...
                 (ax(2) * ax(3) * (1-cos(angle)) - ax(1) * sin(angle)) * vec(3);
  
    rot_vec(3) = (ax(1) * ax(3) * (1-cos(angle)) - ax(2) * sin(angle)) * vec(1) +...
                 (ax(2) * ax(3) * (1-cos(angle)) + ax(1) * sin(angle)) * vec(2) +...
                 (ax(3) * ax(3) * (1-cos(angle)) + cos(angle)) * vec(3);
end
