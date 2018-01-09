function astra_geom_visualize(proj_geom, vol_geom)

	if strcmp(proj_geom.type,'parallel') ||  strcmp(proj_geom.type,'fanflat') ||strcmp(proj_geom.type,'parallel3d') ||  strcmp(proj_geom.type,'cone')
		proj_geom = astra_geom_2vec(proj_geom);
	end

	% open window
	f = figure('Visible','off');
	hold on

	% display projection 1
	displayProjection(1);

	% label
	txt = uicontrol('Style','text', 'Position',[10 10 70 20], 'String','Projection');

	% slider
	anglecount = size(proj_geom.Vectors,1);
	sld = uicontrol('Style', 'slider', ...
		'Min', 1, 'Max', anglecount, 'SliderStep', [1 1]/anglecount, 'Value', 1, ...
		'Position', [80 10 200 20], ...
		'Callback', @updateProjection); 

	f.Visible = 'on';

    function updateProjection(source, callbackdata)
		displayProjection(floor(source.Value));
    end

    function displayProjection(a)

		colours = get(gca,'ColorOrder');

    	
    	% set title
    	title(['projection ' num2str(a)]);

		if strcmp(proj_geom.type,'parallel_vec')

			v = proj_geom.Vectors;
			d = proj_geom.DetectorCount;

			if ~isfield(vol_geom, 'option')
				minx = -vol_geom.GridRowCount/2;
				miny = -vol_geom.GridColCount/2;
				minz = -vol_geom.GridSliceCount/2;
				maxx = vol_geom.GridRowCount/2;
			else
				minx = vol_geom.option.WindowMinX;
				miny = vol_geom.option.WindowMinY;
				maxx = vol_geom.option.WindowMaxX;
				maxy = vol_geom.option.WindowMaxY;
			end
		
			% axis
			cla
			axis([minx maxx miny maxy]*2.25)
			axis square

			% volume
			plot([minx minx maxx maxx minx], [miny maxy maxy miny miny], 'LineWidth', 1, 'Color', colours(1,:))

			% ray
			s = maxx - minx;
			plot([0 v(a,1)]*s*0.33, [0 v(a,2)]*s*0.33, 'LineWidth', 2, 'Color', colours(3,:))

			% detector
			s2 = s*0.75;
			plot([-d/2 d/2]*v(a,5) + v(a,3) + s2*v(a,1), [-d/2 d/2]*v(a,6) + v(a,4) + s2*v(a,2), 'LineWidth', 2, 'Color', colours(5,:))

		elseif strcmp(proj_geom.type,'fanflat_vec')

			v = proj_geom.Vectors;
			d = proj_geom.DetectorCount;

			if ~isfield(vol_geom, 'option')
				minx = -vol_geom.GridRowCount/2;
				miny = -vol_geom.GridColCount/2;
				minz = -vol_geom.GridSliceCount/2;
				maxx = vol_geom.GridRowCount/2;
			else
				minx = vol_geom.option.WindowMinX;
				miny = vol_geom.option.WindowMinY;
				maxx = vol_geom.option.WindowMaxX;
				maxy = vol_geom.option.WindowMaxY;
			end

			% axis
			cla
			axis([minx maxx miny maxy]*2.25)
			axis square

			% volume
			plot([minx minx maxx maxx minx], [miny maxy maxy miny miny], 'LineWidth', 1, 'Color', colours(1,:))

			% detector
			D1 = v(a,3:4) - d/2*v(a,5:6);
			D2 = v(a,3:4) + d/2*v(a,5:6);
			plot([D1(1) D2(1)], [D1(2) D2(2)], 'LineWidth', 2, 'Color', colours(5,:))

			% beam
			plot([v(a,1) D1(1)], [v(a,2) D1(2)], 'LineWidth', 1, 'Color', colours(3,:))
			plot([v(a,1) D2(1)], [v(a,2) D2(2)], 'LineWidth', 1, 'Color', colours(3,:))

		elseif strcmp(proj_geom.type,'parallel3d_vec')

			v = proj_geom.Vectors;
			d1 = proj_geom.DetectorRowCount;
			d2 = proj_geom.DetectorColCount;

			if ~isfield(vol_geom, 'option')
				minx = -vol_geom.GridRowCount/2;
				miny = -vol_geom.GridColCount/2;
				minz = -vol_geom.GridSliceCount/2;
				maxx = vol_geom.GridRowCount/2;
				maxy = vol_geom.GridColCount/2;
				maxz = vol_geom.GridSliceCount/2;
			else
				minx = vol_geom.option.WindowMinX;
				miny = vol_geom.option.WindowMinY;
				minz = vol_geom.option.WindowMinZ;
				maxx = vol_geom.option.WindowMaxX;
				maxy = vol_geom.option.WindowMaxY;
				maxz = vol_geom.option.WindowMaxZ;
			end

			% axis
			windowminx = min(v(:,4));
			windowminy = min(v(:,5));
			windowminz = max(v(:,6));
			windowmaxx = max(v(:,4));
			windowmaxy = max(v(:,5));
			windowmaxz = max(v(:,6));
			cla
			axis([minx maxx miny maxy minz maxz]*5.10)

			% volume
			plot3([minx minx maxx maxx minx], [miny maxy maxy miny miny], [minz minz minz minz minz], 'LineWidth', 1, 'Color', colours(1,:))
			plot3([minx minx maxx maxx minx], [miny maxy maxy miny miny], [maxz maxz maxz maxz maxz], 'LineWidth', 1, 'Color', colours(1,:))
			plot3([minx minx], [miny miny], [minz maxz], 'LineWidth', 1, 'Color', colours(1,:))
			plot3([maxx maxx], [miny miny], [minz maxz], 'LineWidth', 1, 'Color', colours(1,:))
			plot3([minx minx], [maxy maxy], [minz maxz], 'LineWidth', 1, 'Color', colours(1,:))
			plot3([maxx maxx], [maxy maxy], [minz maxz], 'LineWidth', 1, 'Color', colours(1,:))

			% detector
			D1 = v(a,4:6) - d1/2*v(a,7:9) - d2/2*v(a,10:12);
			D2 = v(a,4:6) + d1/2*v(a,7:9) - d2/2*v(a,10:12);
			D3 = v(a,4:6) + d1/2*v(a,7:9) + d2/2*v(a,10:12);
			D4 = v(a,4:6) - d1/2*v(a,7:9) + d2/2*v(a,10:12);
			plot3([D1(1) D2(1) D3(1) D4(1) D1(1)], [D1(2) D2(2) D3(2) D4(2) D1(2)], [D1(3) D2(3) D3(3) D4(3) D1(3)], 'LineWidth', 2, 'Color', colours(5,:))

			% ray
			s = maxx - minx;
			plot3([0 v(a,1)]*s*0.30, [0 v(a,2)]*s*0.30, [0 v(a,3)]*s*0.30, 'LineWidth', 2, 'Color', colours(3,:))

		elseif strcmp(proj_geom.type,'cone_vec')

			v = proj_geom.Vectors;
			d1 = proj_geom.DetectorRowCount;
			d2 = proj_geom.DetectorColCount;

			if ~isfield(vol_geom, 'option')
				minx = -vol_geom.GridRowCount/2;
				miny = -vol_geom.GridColCount/2;
				minz = -vol_geom.GridSliceCount/2;
				maxx = vol_geom.GridRowCount/2;
				maxy = vol_geom.GridColCount/2;
				maxz = vol_geom.GridSliceCount/2;
			else
				minx = vol_geom.option.WindowMinX;
				miny = vol_geom.option.WindowMinY;
				minz = vol_geom.option.WindowMinZ;
				maxx = vol_geom.option.WindowMaxX;
				maxy = vol_geom.option.WindowMaxY;
				maxz = vol_geom.option.WindowMaxZ;
			end

			% axis
			windowminx = min(v(:,4));
			windowminy = min(v(:,5));
			windowminz = max(v(:,6));
			windowmaxx = max(v(:,4));
			windowmaxy = max(v(:,5));
			windowmaxz = max(v(:,6));
			cla
			axis([minx maxx miny maxy minz maxz]*5.10)

			% volume
			plot3([minx minx maxx maxx minx], [miny maxy maxy miny miny], [minz minz minz minz minz], 'LineWidth', 1, 'Color', colours(1,:))
			plot3([minx minx maxx maxx minx], [miny maxy maxy miny miny], [maxz maxz maxz maxz maxz], 'LineWidth', 1, 'Color', colours(1,:))
			plot3([minx minx], [miny miny], [minz maxz], 'LineWidth', 1, 'Color', colours(1,:))
			plot3([maxx maxx], [miny miny], [minz maxz], 'LineWidth', 1, 'Color', colours(1,:))
			plot3([minx minx], [maxy maxy], [minz maxz], 'LineWidth', 1, 'Color', colours(1,:))
			plot3([maxx maxx], [maxy maxy], [minz maxz], 'LineWidth', 1, 'Color', colours(1,:))

			% detector
			D1 = v(a,4:6) - d1/2*v(a,7:9) - d2/2*v(a,10:12);
			D2 = v(a,4:6) + d1/2*v(a,7:9) - d2/2*v(a,10:12);
			D3 = v(a,4:6) + d1/2*v(a,7:9) + d2/2*v(a,10:12);
			D4 = v(a,4:6) - d1/2*v(a,7:9) + d2/2*v(a,10:12);
			plot3([D1(1) D2(1) D3(1) D4(1) D1(1)], [D1(2) D2(2) D3(2) D4(2) D1(2)], [D1(3) D2(3) D3(3) D4(3) D1(3)], 'LineWidth', 2, 'Color', colours(5,:))

			% beam
			plot3([v(a,1) D1(1)], [v(a,2) D1(2)], [v(a,3) D1(3)], 'LineWidth', 1, 'Color', colours(3,:))
			plot3([v(a,1) D2(1)], [v(a,2) D2(2)], [v(a,3) D2(3)], 'LineWidth', 1, 'Color', colours(3,:))
			plot3([v(a,1) D3(1)], [v(a,2) D3(2)], [v(a,3) D3(3)], 'LineWidth', 1, 'Color', colours(3,:))
			plot3([v(a,1) D4(1)], [v(a,2) D4(2)], [v(a,3) D4(3)], 'LineWidth', 1, 'Color', colours(3,:))


		else
			error('invalid projector type')

		end
    end

end
