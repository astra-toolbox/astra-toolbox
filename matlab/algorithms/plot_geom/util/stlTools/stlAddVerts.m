function [vnew, fnew] = stlAddVerts(v, f, list)
%STLADDVERTS adds new vertices (and consequently, new faces) to a STL object
%V is the Nx3 array of vertices
%F is the Mx3 array of faces
%LIST is the list of vertices to be added to the object
%VNEW is the new array of vertices
%FNEW is the new array of faces

% triangulation just with the slice
faces = delaunay(list(:,1),list(:,2)); % calculate new faces
% update object
nvert = length(v); % number of original vertices
v = [v; list]; % update vertices with the ones in the list
f = [f; faces+nvert]; % update faces with the new ones
[vnew,fnew] = stlSlimVerts(v,f); % clear repeated vertices