function list = stlGetVerts(v, f, mode)
%GETVERTS returns the vertices that are 'opened' or 'closed' depending on 
%the 'mode'. An 'open' vertice is the one that defines an open side. An 
%open side is the one that only takes part of one triangle
%V is the Nx3 array of vertices
%F is the Mx3 array of faces
%MODE can be 'opened' or 'closed' depending of the kind of vertices to list
%LIST is the list of 'opened' or 'closed' vertices

sides = sort([[f(:,1) f(:,2)]; ...
    [f(:,2) f(:,3)]; ...
    [f(:,3) f(:,1)]],2);

[C,ia,ic] = unique(sides,'rows');
ind_all = sort(ic); % open and closed sides
ind_rep = find(diff(ind_all) == 0);
ind_cls = ind_all(ind_rep); % closed sides
sides_cls = C(ind_cls,:);
ind_rep = [ind_rep; ind_rep+1];
ind_opn = ind_all;
ind_opn(ind_rep) = []; % open sides
sides_opn = C(ind_opn,:);

switch mode,
    case'opened',
        list = v(unique(sides_opn(:)),:);
    case 'closed',
        list = v(unique(sides_cls(:)),:);
    otherwise,
        error('getVerts:InvalidMode','The ''mode'' valid values are ''opened'' or ''closed''');
end