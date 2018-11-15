function [dist] = eucl_dist3d(a, b)
%%  eucl_dist3d.m
%   3d euclidean distance for a nx3 matrix holding n 3-vectors
%   param a     -   first vectors 
%   param b     -   second vectors 
%   date            07.11.2018
%   author          Tim Elberfeld
%                   imec VisionLab
%                   University of Antwerp
%   last update     07.11.2018
%%
    dist = sqrt((a(:, 1) - b(:, 1)).^2 + ...
           (a(:, 2) - b(:, 2)).^2 + ...
           (a(:, 3) - b(:, 3)).^2);
end

