function format = stlGetFormat(fileName)
%STLGETFORMAT identifies the format of the STL file and returns 'binary' or
%'ascii'

fid = fopen(fileName);
% Check the file size first, since binary files MUST have a size of 84+(50*n)
fseek(fid,0,1);         % Go to the end of the file
fidSIZE = ftell(fid);   % Check the size of the file
if rem(fidSIZE-84,50) > 0
    format = 'ascii';
else
    % Files with a size of 84+(50*n), might be either ascii or binary...
    % Read first 80 characters of the file.
    % For an ASCII file, the data should begin immediately (give or take a few
    % blank lines or spaces) and the first word must be 'solid'.
    % For a binary file, the first 80 characters contains the header.
    % It is bad practice to begin the header of a binary file with the word
    % 'solid', so it can be used to identify whether the file is ASCII or
    % binary.
    fseek(fid,0,-1); % go to the beginning of the file
    header = strtrim(char(fread(fid,80,'uchar')')); % trim leading and trailing spaces
    isSolid = strcmp(header(1:min(5,length(header))),'solid'); % take first 5 char
    fseek(fid,-80,1); % go to the end of the file minus 80 characters
    tail = char(fread(fid,80,'uchar')');
    isEndSolid = findstr(tail,'endsolid');
    
    % Double check by reading the last 80 characters of the file.
    % For an ASCII file, the data should end (give or take a few
    % blank lines or spaces) with 'endsolid <object_name>'.
    % If the last 80 characters contains the word 'endsolid' then this
    % confirms that the file is indeed ASCII.
    if isSolid & isEndSolid
        format = 'ascii';
    else
        format = 'binary';
    end
end
fclose(fid);