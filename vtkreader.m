% load a vtk file
% needs to support SCALARS or VECTORS
% AND
% needs to load more than one dataset if it's there
fname = '/home/dtward/data/csh_data/Marmoset_CCF/outputs_female_v00/nissl_registered/nissl_to_nissl_registered/images/nissl_nissl_to_nissl_registered_rgb.vtk'
fname = '/home/dtward/data/csh_data/Marmoset_CCF/outputs_female_v00/nissl_registered/myelin_to_nissl_registered/images/myelin_myelin_to_nissl_registered_rgb.vtk'
fname = '/home/dtward/data/AllenInstitute/allen_vtk/annotation_50_bregma_LR.vtk'



verbose = 1;

% open it in binary mode
fid = fopen(fname,'rb');

% the first line should say the vtk version
line = fgetl(fid);
if verbose
    disp(line)
end

% the second line should give a title for the file
line = fgetl(fid);
if verbose
    disp(line)
end
title_ = line;

% the third line should say ASCII or BINARY
line = fgetl(fid);
if verbose
    disp(line)
end
if ~strcmp(line, 'BINARY')
    error('type must be BINARY')
end


% the fourth line should say DATASET STRUCTURED_POINTS
line = fgetl(fid);
if verbose
    disp(line)
end
if ~strcmp(line,'DATASET STRUCTURED_POINTS')
    error('must say DATASET STRUCTURED_POINTS')
end

% the fifth line will say the dimensions
line = fgetl(fid);
if verbose
    disp(line)
end
[T,line] = strtok(line);
n = str2num(line);
if verbose
    disp(n)
end

% the sixth line will say the origin
line = fgetl(fid);
if verbose
    disp(line)
end
[T,line] = strtok(line);
o = str2num(line);
if verbose
    disp(o)
end

% the seventh line will say the spacing
line = fgetl(fid);
if verbose
    disp(line)
end
[T,line] = strtok(line);
d = str2num(line);
if verbose
    disp(d)
end

% from here we can calculate the location of pixels
x0 = (0:n(1)-1)*d(1) + o(1);
x1 = (0:n(2)-1)*d(2) + o(2);
x2 = (0:n(3)-1)*d(3) + o(3);
% note vtk calls this xyz, and x are next to each other on disk, and vector
% components are in xyz order
% in matlab, next to each other is the first index
% this is opposite to python


% the next line will say point data
line = fgetl(fid);
if verbose
    disp(line)
end
[T,line] = strtok(line);
if ~strcmp(T,'POINT_DATA')
    error('should say POINT_DATA')
end
npoints = str2num(line);
if verbose
    disp(npoints)
end

if npoints ~= prod(n)
    error('number of points should equal size of grid')
end

% now we will loop over multiple datasets
ndatasets = 0;
I = {};
names = {};
while 1
    disp('hi');
    % this line should have SCALARS or VECTORS, I'll give it a few chances
    % to load blank lines, break if end of file
    line = fgetl(fid);
    disp('line is')
    disp(line)
    disp(size(line))    
    while isempty(line) || all(isspace(line))
        line = fgetl(fid);
        disp('line is')
        disp(line)
        disp(size(line))
    end    
    if verbose
        disp(line)
    end
    if line == -1
        break
    end
    [T,line] = strtok(line);
    if ~strcmp(T,'SCALARS') && ~strcmp(T,'VECTORS')
        disp('must say SCALARS or VECTORS')
        break
    end
    SCALARS_VECTORS = T;
    [T,line] = strtok(line);
    dataset_name = T;
    if verbose
        disp(dataset_name)
    end
    
    [T,line] = strtok(line);
    datatype = T;
    if verbose
        disp(T)
    end
    if strcmp(datatype ,'unsigned_char')
        matlab_datatype = 'uchar';
    elseif strcmp(datatype,'unsigned_int')
        matlab_datatype = 'uint32';
    else
        error('datatype is not recognized')
    end
    
    % for scalars only, we can have a number of components
    if strcmp(SCALARS_VECTORS,'SCALARS')
        ncomponents = 1; % default
        if ~isempty(line) % for vectors this should be empty
            ncomponents = str2num(line); 
        end
    else
        ncomponents = length(n); % this should be 3
        if ncomponents~=3
            error('we were expecting 3 vector components')
        end
    end
    
    % now we read the data
    I_ = fread(fid,prod([ncomponents,n]),[matlab_datatype '=>' matlab_datatype],0,'b'); % always big, don't skip anything
    I_ = reshape(I_,[ncomponents,n]);
    if verbose
        disp('read an image of size')
        disp(size(I_))
    end
    % test
    % imagesc(squeeze(permute(squeeze(I_(:,round(size(I_,2)/2),:,:)),[2,3,1])))
    I{end+1} = I_;
    names{end+1} = dataset_name;
    
end



fclose(fid);