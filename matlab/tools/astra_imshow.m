function V = astra_imshow(data, range)

if numel(data) == 1
	data = astra_mex_data2d('get', data);
end
imshow(data,range);

if nargout >= 1
	V = data;
end