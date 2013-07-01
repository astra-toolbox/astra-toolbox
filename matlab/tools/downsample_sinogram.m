function sinogram_out = downsample_sinogram(sinogram, ds)

	if ds == 1
		sinogram_out = sinogram;
		return;
	end

	sinogram_out = sinogram(:,1:ds:end,:);
	for i = 2:ds
		sinogram_out = sinogram_out + sinogram(:,i:ds:end,:);
	end
	sinogram_out = sinogram_out / (ds*ds);