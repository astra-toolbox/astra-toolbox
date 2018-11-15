function [ transl_vectors ] = translate_3d( vectors, transl_vec)
%%   translate some vectors by a translation vector
%   param vectors               -   the vectors to translate
%   param transl_vec            -   vector geometry translation
%   return translated_vec_geom  -   copy of input geometry with translated
%                                   vectors
%
%   date            09.07.2018
%   author          Van Nguyen, Tim Elberfeld
%                   imec VisionLab
%                   University of Antwerp
%   last mod        07.11.2018
%%
    transl_vectors = vectors; % copy input
    transl = repmat(transl_vec, size(vectors, 1), 1);
    transl_vectors = transl_vectors + transl;
end

