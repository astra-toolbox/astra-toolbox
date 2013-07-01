function varargout = astra_data_gui(varargin)
% ASTRA_DATA_GUI M-file for ASTRA_DATA_GUI.fig
%      ASTRA_DATA_GUI, by itself, creates a new ASTRA_DATA_GUI or raises the existing
%      singleton*.
%
%      H = ASTRA_DATA_GUI returns the handle to a new ASTRA_DATA_GUI or the handle to
%      the existing singleton*.
%
%      ASTRA_DATA_GUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in ASTRA_DATA_GUI.M with the given input arguments.
%
%      ASTRA_DATA_GUI('Property','Value',...) creates a new ASTRA_DATA_GUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before ASTRA_DATA_GUI_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to ASTRA_DATA_GUI_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help ASTRA_DATA_GUI

% Last Modified by GUIDE v2.5 05-Mar-2012 14:34:03

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @astra_data_gui_OpeningFcn, ...
                   'gui_OutputFcn',  @astra_data_gui_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before astra_data_gui is made visible.
function astra_data_gui_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to astra_data_gui (see VARARGIN)

% Choose default command line output for astra_data_gui
handles.output = hObject;
handles.data = [];

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes astra_data_gui wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = astra_data_gui_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% Use this function to display a figure using the gui from any m-file
% example:
%       Handle = astra_data_gui();
%       astra_data_gui('loadVolume',guihandles(Handle),'rand(30,30,30)',15);
function loadVolume(handles,name,figure_number)
set(handles.txt_var, 'String', name);
set(handles.figure_number, 'String', num2str(figure_number));
btn_load_Callback(handles.txt_var, [], handles);





function txt_var_Callback(hObject, eventdata, handles) %#ok<*DEFNU>
% hObject    handle to txt_var (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of txt_var as text
%        str2double(get(hObject,'String')) returns contents of txt_var as a double


% --- Executes during object creation, after setting all properties.
function txt_var_CreateFcn(hObject, eventdata, handles)
% hObject    handle to txt_var (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in btn_load.
function btn_load_Callback(hObject, eventdata, handles)
% hObject    handle to btn_load (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

s = get(handles.txt_var, 'String');
data = evalin('base', s);
handles.data = data;
guidata(hObject, handles);

% Set Default Stuff
set(handles.sld_slice, 'Min',1);
set(handles.sld_slice, 'Max', size(data,3));
set(handles.sld_slice, 'SliderStep', [1/(size(data,3)-2) 1/(size(data,3)-2)]);
set(handles.sld_slice, 'Value', size(data,3)/2);

sliderValue = floor(get(handles.sld_slice, 'Value'));
set(handles.txt_slice, 'String', num2str(sliderValue));
set(handles.txt_min, 'String', num2str(1));
set(handles.txt_max, 'String', num2str(size(data,3)));

set(handles.sld_magnification, 'Min',1);
set(handles.sld_magnification, 'Max', 400);
set(handles.sld_magnification, 'SliderStep', [1/(400-2) 1/(400-2)]);
set(handles.sld_magnification, 'Value', 100);

sliderValue3 = floor(get(handles.sld_magnification, 'Value'));
set(handles.txt_mag, 'String', num2str(sliderValue3));


figure_number = floor(str2double(get(handles.figure_number, 'String')));
if(isnan(figure_number) || figure_number < 1)
    set(handles.figure_number, 'String', num2str(10));
end

showimage(handles);

% --- SHOW IMAGE
function showimage(handles)
	sliderValue = floor(get(handles.sld_slice, 'Value'));
    magnification = floor(get(handles.sld_magnification, 'Value'));
    figure_number = floor(str2double(get(handles.figure_number, 'String')));
    image_matrix = handles.data;
	if get(handles.btn_x, 'Value') == 1
		figure(figure_number), imshow(sliceExtractor((image_matrix(:,:,:)), 'y', sliderValue),[],'InitialMagnification', magnification);
        ylabel('y')
        xlabel('z')
        set(gcf,'Name','ASTRA DATA GUI')
	elseif get(handles.btn_y, 'Value') == 1
		figure(figure_number), imshow(sliceExtractor((image_matrix(:,:,:)), 'x', sliderValue),[],'InitialMagnification', magnification);
        ylabel('x')
        xlabel('z')
        set(gcf,'Name','ASTRA DATA GUI')
	else
		figure(figure_number), imshow(sliceExtractor((image_matrix(:,:,:)), 'z', sliderValue),[],'InitialMagnification', magnification);
        ylabel('x')
        xlabel('y')
        set(gcf,'Name','ASTRA DATA GUI')
    end


% --- Executes on slider movement.
function sld_slice_Callback(hObject, eventdata, handles)
% hObject    handle to sld_slice (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
	
sliderValue = floor(get(handles.sld_slice, 'Value'));
set(handles.txt_slice, 'String', num2str(sliderValue));
showimage(handles);

% --- Executes during object creation, after setting all properties.
function sld_slice_CreateFcn(hObject, eventdata, handles)
% hObject    handle to sld_slice (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in pushbutton3.
function pushbutton3_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in pushbutton4.
function pushbutton4_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


function txt_slice_Callback(hObject, eventdata, handles)
% hObject    handle to txt_slice (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
slice = str2double(get(handles.txt_slice, 'String'));
max = str2num(get(handles.txt_max,'String'));
min = str2num(get(handles.txt_min,'String'));
if(slice > max)
    set(handles.txt_slice, 'String', num2str(max));
    set(handles.sld_slice, 'Value', max);
elseif(slice < min)
    set(handles.txt_slice, 'String', num2str(min));
    set(handles.sld_slice, 'Value', min);
else
    set(handles.sld_slice, 'Value', slice);
end
showimage(handles);
	
% --- Executes during object creation, after setting all properties.
function txt_slice_CreateFcn(hObject, eventdata, handles)
% hObject    handle to txt_slice (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes on slider movement.
function sld_magnification_Callback(hObject, eventdata, handles)
% hObject    handle to sld_slice2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
sliderValue3 = floor(get(handles.sld_magnification, 'Value'));
set(handles.txt_mag, 'String', num2str(sliderValue3));

if(~isempty(handles.data))
    showimage(handles);
end



% --- Executes during object creation, after setting all properties.
function sld_magnification_CreateFcn(hObject, eventdata, handles)
% hObject    handle to sld_slice2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


function txt_mag_Callback(hObject, eventdata, handles)
% hObject    handle to txt_slice2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
magnification = str2double(get(handles.txt_mag, 'String'));
if(magnification > 400)
    set(handles.txt_mag, 'String', num2str(400));
    set(handles.sld_magnification, 'Value', 400);
elseif(magnification < 1)
    set(handles.txt_mag, 'String', num2str(1));
    set(handles.sld_magnification, 'Value', 1);
else
    set(handles.sld_magnification, 'Value', magnification);
end

if(~isempty(handles.data))
    showimage(handles);
end

% --- Executes during object creation, after setting all properties.
function txt_mag_CreateFcn(hObject, eventdata, handles)
% hObject    handle to txt_slice2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes on slider movement.
function figure_number_Callback(hObject, eventdata, handles)
% hObject    handle to sld_slice2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
number = floor(str2double(get(handles.figure_number, 'String')));
if(number < 1)
    set(handles.figure_number, 'String', num2str(1));
else
    set(handles.figure_number, 'String', num2str(number));
end

if(~isempty(handles.data))
    showimage(handles);
end


% --- Executes during object creation, after setting all properties.
function figure_number_CreateFcn(hObject, eventdata, handles)
% hObject    handle to sld_slice2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end



% --- Executes when selected object is changed in btn_dir.
function btn_dir_SelectionChangeFcn(hObject, eventdata, handles)
% hObject    handle to the selected object in btn_dir 
% eventdata  structure with the following fields (see UIBUTTONGROUP)
%	EventName: string 'SelectionChanged' (read only)
%	OldValue: handle of the previously selected object or empty if none was selected
%	NewValue: handle of the currently selected object
% handles    structure with handles and user data (see GUIDATA)

data = handles.data;

if(hObject == handles.btn_x)
    set(handles.btn_x, 'Value', 1);
    set(handles.btn_y, 'Value', 0);
    set(handles.btn_z, 'Value', 0);
elseif(hObject == handles.btn_y)
    set(handles.btn_x, 'Value', 0);
    set(handles.btn_y, 'Value', 1);
    set(handles.btn_z, 'Value', 0);
elseif(hObject == handles.btn_z)
    set(handles.btn_x, 'Value', 0);
    set(handles.btn_y, 'Value', 0);
    set(handles.btn_z, 'Value', 1);
end

if get(handles.btn_x, 'Value') == 1
	set(handles.sld_slice, 'Min',1);
	set(handles.sld_slice, 'Max', size(data,1));
	set(handles.sld_slice, 'SliderStep', [1/(size(data,1)-2) 1/(size(data,1)-2)]);
	set(handles.sld_slice, 'Value', size(data,1)/2);

	sliderValue = get(handles.sld_slice, 'Value');
	set(handles.txt_slice, 'String', num2str(sliderValue));
	set(handles.txt_min, 'String', num2str(1));
	set(handles.txt_max, 'String', num2str(size(data,1)));
	
elseif get(handles.btn_y, 'Value') == 1
	set(handles.sld_slice, 'Min',1);
	set(handles.sld_slice, 'Max', size(data,2));
	set(handles.sld_slice, 'SliderStep', [1/(size(data,2)-2) 1/(size(data,2)-2)]);
	set(handles.sld_slice, 'Value', size(data,2)/2);

	sliderValue = get(handles.sld_slice, 'Value');
	set(handles.txt_slice, 'String', num2str(sliderValue));
	set(handles.txt_min, 'String', num2str(1));
	set(handles.txt_max, 'String', num2str(size(data,2)));
else
	set(handles.sld_slice, 'Min',1);
	set(handles.sld_slice, 'Max', size(data,3));
	set(handles.sld_slice, 'SliderStep', [1/(size(data,3)-2) 1/(size(data,3)-2)]);
	set(handles.sld_slice, 'Value', size(data,3)/2);

	sliderValue = get(handles.sld_slice, 'Value');
	set(handles.txt_slice, 'String', num2str(sliderValue));
	set(handles.txt_min, 'String', num2str(1));
	set(handles.txt_max, 'String', num2str(size(data,3)));
end

showimage(handles);
