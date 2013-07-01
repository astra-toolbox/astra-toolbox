function V_out = ROIselectfull(input, ROI)

	s1 = size(input,1);
	s2 = size(input,2);
	[x y] = meshgrid(-(s2-1)/2:(s2-1)/2,(s1-1)/2:-1:-(s1-1)/2);
	A = Afstand(x,y,0,0);

	V_out = zeros(size(input));
	for slice = 1:size(input,3);
		V = input(:,:,slice);
		V(A > ROI/2) = 0;
		V_out(:,:,slice) = V; 
	end
end

function A = Afstand(x1,y1,x2,y2)
	A = sqrt((x1-x2).^2+(y1-y2).^2);
end