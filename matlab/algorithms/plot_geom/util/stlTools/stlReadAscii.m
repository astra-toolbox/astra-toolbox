function [v, f, n, name] = stlReadAscii(fileName)
%STLREADASCII reads a STL file written in ASCII format
%V are the vertices
%F are the faces
%N are the normals
%NAME is the name of the STL object (NOT the name of the STL file)

%======================
% STL ascii file format
%======================
% ASCII STL files have the following structure.  Technically each facet
% could be any 2D shape, but in practice only triangular facets tend to be
% used.  The present code ONLY works for meshes composed of triangular
% facets.
%
% solid object_name
% facet normal x y z
%   outer loop
%     vertex x y z
%     vertex x y z
%     vertex x y z
%   endloop
% endfacet
%
% <Repeat for all facets...>
%
% endsolid object_name

fid = fopen(fileName);
cellcontent = textscan(fid,'%s','delimiter','\n'); % read all the file and put content in cells
content = cellcontent{:}(logical(~strcmp(cellcontent{:},''))); % remove all blank lines
fclose(fid);

% read the STL name
line1 = char(content(1));
if (size(line1,2) >= 7)
    name = line1(7:end);
else
    name = 'Unnamed Object';
end

% read the vector normals
normals = char(content(logical(strncmp(content,'facet normal',12))));
n = str2num(normals(:,13:end));

% read the vertex coordinates (vertices)
vertices = char(content(logical(strncmp(content,'vertex',6))));
v = str2num(vertices(:,7:end));
nvert = size(vertices,1); % number of vertices
nfaces = sum(strcmp(content,'endfacet')); % number of faces
if (nvert == 3*nfaces)
    f = reshape(1:nvert,[3 nfaces])'; % create faces
end

% slim the file (delete duplicated vertices)
[v,f] = stlSlimVerts(v,f);