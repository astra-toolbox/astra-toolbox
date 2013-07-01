function [rnmp, nmp] = compute_rnmp(phantom, S)

	phantom = double(phantom == max(phantom(:))); 
	S = double(S == max(S(:)));

	%u1 = sort(unique(phantom));
	%u2 = sort(unique(S));
	%for i = 1:numel(u1)
	%	phantom_(phantom == u1(i)) = i;
	%	S_(S == u2(i)) = i;
	%end
	%phantom = phantom_;
	%S = S_;
	
	if numel(size(phantom)) == 2
		S = imresize(S, size(phantom), 'nearest');
	elseif numel(size(phantom)) == 3
		S2 = zeros(size(phantom));
		for slice = 1:size(phantom,3)
			S2(:,:,slice) = imresize(S(:,:,slice), [size(phantom,1) size(phantom,2)], 'nearest');
		end
		S = S2;
	end

	nmp = sum(abs(phantom(:) ~= S(:)));
	rnmp = nmp / sum(phantom(:));

end

