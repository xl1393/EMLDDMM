import numpy as np
import matplotlib.pyplot as plt
import torch
import glob
import os
from os.path import join,split,splitext
from os import makedirs
import nibabel
import json
import re
import tifffile as tf
import PIL
try:    
    PIL.Image.MAX_IMAGE_PIXELS = None 
except:
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
from .utils import interp

def load_slices(target_name, xJ=None):
    """ Load a slice dataset.
    
    Load a slice dataset for histology registration. Slice datasets include pairs
    of images and json sidecar files, as well as one tsv file explaining the dataset.
    Note this code creates a 3D array by padding.
    
    Parameters
    ----------
    target_name : string
        Name of a directory containing slice dataset.
    xJ : list, optional
        list of numpy arrays containing voxel positions along each axis.
        Images will be resampled by interpolation on this 3D grid.

    Returns
    -------
    xJ : list of numpy arrays
        Location of v
    J : numpy array
        Numpy array of size C x nslices x nrows x ncols where C is the number of channels
        e.g. C=3 for RGB.
    W0 : numpy array
        A nslices x nrows x ncols numpy array containing weights.  Weights are 0 where there 
        was padding
    
    
    
    Raises
    ------
    Exception
        If the first image is not present in the image series.

    """
    #print('loading target images')
    fig,ax = plt.subplots()
    ax = [ax]
    # current limitation
    # requires the word 'present'
    # requires the first image to be present
    # expects data type to be in 0,1
    # assumes space directions are diagonal
    # todo: origin
    # we will need more control over the size, and we will need to maintain the origin of each slice
    # right now we have a heuristic for taking 99th percentile and expanding by 1%
    
    data = []
    # load the one tsv file
    tsv_name = join(target_name, 'samples.tsv' )
    with open(tsv_name,'rt') as f:
        for count,line in enumerate(f):
            line = line.strip()
            key = '\t' if '\t' in line else '    '
            if count == 0:
                headings = re.split(key,line)                
                continue
            data.append(re.split(key,line))
    data_ = np.zeros((len(data),len(data[0])),dtype=object)
    for i in range(data_.shape[0]):
        for j in range(data_.shape[1]):
            try:
                data_[i,j] = data[i][j]
            except:
                data_[i,j] = ''
    data = data_
    #print(f'dataset with shape {data.shape}')
    
    # now we will loop through the files and get the sizes 
    nJ_ = np.zeros((data.shape[0],3),dtype=int)
    origin = np.zeros((data.shape[0],3),dtype=float)
    slice_status = data[:,3]
    J_ = []    
    for i in range(data.shape[0]):
        #if not (slice_status[i].lower() == 'present' or slice_status[i].lower() == 'true'):
        if slice_status[i].lower() in ['missing','absent',False,'False','false']:
            # if i == 0:
            #     raise Exception('First image is not present')
            # J_.append(np.array([[[0.0,0.0,0.0]]]))
            continue
        namekey = data[i,0]
        #print(namekey)
        searchstring = join(target_name,'*'+os.path.splitext(namekey)[0]+'*.json')
        #print(searchstring)
        jsonfile = glob.glob(searchstring)
        #print(jsonfile)
        with open(jsonfile[0]) as f:
            jsondata = json.load(f)
        #nJ_[i] = np.array(jsondata['Sizes'])


        # this should contain an image and a json    

        image_name = jsondata['DataFile']
        _, ext = os.path.splitext(image_name)
        if ext == '.tif':
            J__ = tf.imread(os.path.join(target_name, image_name))
        else:
            J__ = plt.imread(os.path.join(target_name,image_name))

        if J__.dtype == np.uint8:
            J__ = J__.astype(float)/255.0
            J__ = J__[...,:3] # no alpha
        else:
            J__ = J__[...,:3].astype(float)
            J__ = J__ / np.mean(np.abs(J__.reshape(-1, J__.shape[-1])), axis=0)

        if not i%20:
            ax[0].cla()
            toshow = (J__- np.min(J__)) / (np.max(J__)-np.min(J__))
            ax[0].imshow(toshow)
            fig.suptitle(f'slice {i} of {data.shape[0]}: {image_name}')
            fig.canvas.draw()    

        nJ_[i] = np.array(J__.shape)

        J_.append(J__)
               


        # the domain
        # if this is the first file we want to set up a 3D volume
        if 'dJ' not in locals():
            dJ = np.diag(np.array(jsondata['SpaceDirections'][1:]))[::-1]
        # note the order needs to be reversed
        origin[i] = np.array(jsondata['SpaceOrigin'])
        x0 = origin[:,2] # z coordinates of slices
    if xJ == None:
        #print('building grid')
        # build 3D coordinate grid
        nJ0 = np.array(int((np.max(x0) - np.min(x0))//dJ[0]) + 1) # length of z axis on the grid (there may be missing slices)
        nJm = np.max(nJ_,0)
        nJm = (np.quantile(nJ_,0.95,axis=0)*1.01).astype(int) # this will look for outliers when there are a small number, really there just shouldn't be outliers
        nJ = np.concatenate(([nJ0],nJm[:-1]))
        # get the minimum coordinate on each axis
        xJmin = [-(n-1)*d/2.0 for n,d in zip(nJ[1:],dJ[1:])]
        xJmin.insert(0,np.min(x0))
        xJ = [(np.arange(n)*d + o) for n,d,o in zip(nJ,dJ,xJmin)]
        #print(xJ)
    XJ = np.stack(np.meshgrid(*xJ, indexing='ij'))

    # get the presence of a slice at z axis grid points. This is used for loading into a 3D volume. 
    # slice_status = []
    # i = 0
    # j = 0
    # while i < len(xJ[0]):
    #     if j == len(x0):
    #         slice_status = slice_status + [False]*(len(xJ[0])-i)
    #         break
    #     status = xJ[0][i] == x0[j]
    #     if status == False:
    #         i += 1
    #     else:
    #         i += 1
    #         j += 1
    #     slice_status.append(status)

    # resample slices on 3D grid    
    J = np.zeros(XJ.shape[1:] + tuple([3]))
    W0 = np.zeros(XJ.shape[1:])
    i = 0
    #print('starting to interpolate slice dataset')
    for j in range(XJ.shape[1]):
        # if slice_status[j] == False:
        if slice_status[j] in ['missing','absent',False,'False','false']:
            #print(f'slice {j} was missing')
            continue
        # getting an index out of range issue in the line below (problem was 'missing' versus 'absent')
        #print(dJ,J_[i].shape)
        xJ_ = [np.arange(n)*d - (n-1)*d/2.0 for n,d in zip(J_[i].shape[:-1], dJ[1:])]
        # note, padding mode border means weights will not be appropriate, change on jan 11, 2024
        #J[j] = np.transpose(interp(xJ_, J_[i].transpose(2,0,1), XJ[1:,0], interp2d=True, padding_mode="border"), (1,2,0))
        J[j] = np.transpose(interp(xJ_, J_[i].transpose(2,0,1), XJ[1:,0], interp2d=True, padding_mode="zeros"), (1,2,0))
        W0_ = np.zeros(W0.shape[1:])
        #W0_[J[i,...,0] > 0.0] = 1.0 # we check if the first channel is greater than 0
        # jan 11, 2024, I think there is a mistake above, I thnk it should be j
        W0_[J[j,...,0] > 0.0] = 1.0 # we check if the first channel is greater than 0
        #W0[i] = W0_
        W0[j] = W0_
        i += 1
    J = np.transpose(J,(3,0,1,2))

    #print(f'J shape {J.shape}')
    return xJ,J,W0
    

            
# resampling
def write_vtk_data(fname,x,out,title,names=None):
    
    '''    
    Write data as vtk file legacy format file. Note data is written in big endian.
    
    inputs should be numpy, but will check for tensor
    only structured points supported, scalars or vectors data type
    each channel is saved as a dataset (time for velocity field, or image channel for images)
    each channel is saved as a structured points with a vector or a scalar at each point        
    
    Parameters
    ----------
    fname : str
        filename to write to
    x : list of arrays
        Voxel locations along last three axes
    out : numpy array
        Imaging data to write out. If out is size nt x 3 x slices x height x width we assume vector
        if out is size n x slices x height x width we assume scalar     
    title : str
        Name of the dataset
    names : list of str or None
        List of names for each dataset or None to use a default.        

    Raises
    ------
    Exception
        If out is not the right size.
    
    '''
    
    if len(out.shape) == 5 and out.shape[1] == 3:
        type_ = 'VECTORS'
    elif len(out.shape) == 4:
        type_ = 'SCALARS'
    else:
        raise Exception('out is not the right size')
    if names is None:        
        names = [f'data_{t:03d}(b)' for t in range(out.shape[0])]
    else:
        # make sure we know it is big endian
        names = [n if '(b)' in n else n+'(b)' for n in names]
        
        

    if type(out) == torch.Tensor:        
        out = out.cpu().numpy()
    
    with open(fname,'wt') as f:
        f.writelines([
            '# vtk DataFile Version 3.0\n',
            title+'\n',
            'BINARY\n',
            'DATASET STRUCTURED_POINTS\n',
            f'DIMENSIONS {out.shape[-1]} {out.shape[-2]} {out.shape[-3]}\n',
            f'ORIGIN {x[-1][0]} {x[-2][0]} {x[-3][0]}\n',
            f'SPACING {x[-1][1]-x[-1][0]} {x[-2][1]-x[-2][0]} {x[-3][1]-x[-3][0]}\n',
            f'POINT_DATA {out.shape[-1]*out.shape[-2]*out.shape[-3]}\n'                  
            ])
    
    
    for i in range(out.shape[0]):
        with open(fname,'at') as f:
            f.writelines([
                f'{type_} {names[i]} {dtypes[out.dtype]}\n'
            ])
        with open(fname,'ab') as f:
            # make sure big endian 
            if type_ == 'VECTORS':
                # put the vector component at the end
                # on march 29, 2022, daniel flips zyx to xyz
                out_ = np.array(out[i].transpose(1,2,3,0)[...,::-1])
            else:
                f.write('LOOKUP_TABLE default\n'.encode())
                out_ = np.array(out[i])
            outtype = np.dtype(out_.dtype).newbyteorder('>')
            out_.astype(outtype).tofile(f)
        with open(fname,'at') as f:
            f.writelines([
                '\n'
            ])
            
dtypes_reverse = {
    'float':np.dtype('float32'),
    'double':np.dtype('float64'),
    'unsigned_char':np.dtype('uint8'),
    'unsigned_short':np.dtype('uint16'),
    'unsigned_int':np.dtype('uint32'),
    'unsigned_long':np.dtype('uint64'),
    'char':np.dtype('int8'),
    'short':np.dtype('int16'),
    'int':np.dtype('int32'),
    'long':np.dtype('int64'),
}    
def read_vtk_data(fname,normalize=False,endian='b'):
    '''
    Read vtk structured points legacy format data.
    
    Note endian should always be big, but we support little as well.
    
    Parameters
    ----------
    fname : str
        Name of .vtk file to read.
    normalize : bool
        Whether or not to divide an image by its mean absolute value. Defaults to True.
    endian : str
        Endian of data, with 'b' for big (default and only officially supported format)
        or 'l' for little (for compatibility if necessary).
        
    Returns
    -------
    x : list of numpy arrays
        Location of voxels along each spatial axis (last 3 axes)
    images : numpy array
        Image with last three axes corresponding to spatial dimensions.  If 4D,
        first axis is channel.  If 5D, first axis is time, and second is xyz 
        component of vector field.

    Raises
    ------
    Exception
        The first line should include vtk DataFile Version X.X
    Exception
        If the file contains data type other than BINARY.
    Exception
        If the dataset type is not STRUCTURED_POINTS.
    Exception
        If the dataset does not have either 3 or 4 axes.
    Exception
        If dataset does not contain POINT_DATA
    Exception
        If the file does not contain scalars or vectors.
    
    Warns
    -----
    If data not written in big endian
        Note (b) symbol not in data name {name}, you should check that it was written big endian. Specify endian="l" if you want little
        
    TODO
    ----
    Torch does not support negative strides.  This has lead to an error where x has a negative stride.
    I should flip instead of negative stride, or copy afterward.
    '''
    # TODO support skipping blank lines
    big = not (endian=='l')
    
    verbose = True
    verbose = False
    with open(fname,'rb') as f:        
        # first line should say vtk version
        line = f.readline().decode().strip()
        if verbose: print(line)
        if 'vtk datafile' not in line.lower():
            raise Exception('first line should include vtk DataFile Version X.X')
        # second line says title    
        line = f.readline().decode().strip()
        if verbose: print(line)
        title = line

        # third line should say type of data
        line = f.readline().decode().strip()
        if verbose: print(line)
        if not line.upper() == 'BINARY':
            raise Exception(f'Only BINARY data type supported, but this file contains {line}')
        data_format = line

        # next line says type of data
        line = f.readline().decode().strip()
        if verbose: print(line)
        if not line.upper() == 'DATASET STRUCTURED_POINTS':
            raise Exception(f'Only STRUCTURED_POINTS dataset supported, but this file contains {line}')
        geometry = line

        # next line says dimensions    
        # "ordered with x increasing fastest, theny,thenz"
        # this is the same as nrrd (fastest to slowest)
        # however our convention in python that we use channel z y x order
        # i.e. the first is channel
        line = f.readline().decode().strip()
        if verbose: print(line)
        dimensions = np.array([int(n) for n in line.split()[1:]])
        if len(dimensions) not in [3,4]:
            raise Exception(f'Only datasets with 3 or 4 axes supported, but this file contains {dimensions}')

        # next says origin
        line = f.readline().decode().strip()
        if verbose: print(line)
        origin = np.array([float(n) for n in line.split()[1:]])

        # next says spacing
        line = f.readline().decode().strip()
        if verbose: print(line)
        spacing = np.array([float(n) for n in line.split()[1:]])

        # now I can build axes
        # note I have to reverse the order for python
        x = [np.arange(n)*d+o for n,d,o in zip(dimensions[::-1],spacing[::-1],origin[::-1])]

        # next line must contain point_data
        line = f.readline().decode().strip()
        if verbose: print(line)
        if 'POINT_DATA' not in line:
            raise Exception(f'only POINT_DATA supported but this file contains {line}')                          
        N = int(line.split()[-1])

        # now we will loop over available datasets
        names = []
        images = []
        count = 0
        while True:
            
            # first line contains data type (scalar or vector), name, and format
            # it could be a blank line
            line = f.readline().decode()
            if line == '\n':
                line = f.readline().decode()        
            line = line.strip()
            
            if line is None or not line: # check if we've reached the end of the file
                break
                
            if verbose: print(f'starting to load dataset {count}')
                
            if verbose: print(line)            
            S_V = line.split()[0]
            name = line.split()[1]
            dtype = line.split()[2]
            names.append(name)

            if S_V.upper() not in ['SCALARS','VECTORS']:
                raise Exception(f'Only scalars or vectors supported but this file contains {S_V}')        
            
            if '(b)' not in name and big: 
                warn(f'Note (b) symbol not in data name {name}, you should check that it was written big endian. Specify endian="l" if you want little')
                            
            dtype_numpy = dtypes_reverse[dtype]
            if big:
                dtype_numpy_big = dtype_numpy.newbyteorder('>') # > means big endian
            else:
                dtype_numpy_big = dtype_numpy
            #
            # read the data
            if S_V == 'SCALARS':
                # there should be a line with lookup table
                line = f.readline().decode()
                if verbose: print(line)
                data = np.fromfile(f,dtype_numpy_big,N).astype(dtype_numpy)
                # shape it
                data = data.reshape(dimensions[::-1])
                # axis order is already correct because of slowest to fastest convention in numpy

            elif S_V == 'VECTORS':            
                data = np.fromfile(f,dtype_numpy_big,N*3).astype(dtype_numpy)
                # shape it
                data = data.reshape((dimensions[-1],dimensions[-2],dimensions[-3],3))
                # move vector components first
                data = data.transpose((3,0,1,2))
                # with vector data we should flip xyz (file) to zyx (python) (added march 29)
                data = np.copy(data[::-1])
            images.append(data)
            count += 1
        images = np.stack(images) # stack on axis 0
        if normalize:
            images = images / np.mean(np.abs(images)) # normalize

    return x,images,title,names
    
    
def read_data(fname, x=None, **kwargs):
    '''
    Read array data from several file types.
    
    This function will read array based data of several types
    and output x,images,title,names. Note we prefer vtk legacy format, 
    but accept some other formats as read by nibabel.
    
    Parameters
    ----------
    fname : str
        Filename (full path or relative) of array data to load. Can be .vtk or 
        nibabel supported formats (e.g. .nii)
    x : list of arrays, optional
        Coordinates for 2D series space
    **kwargs : dict
        Keyword parameters that are passed on to the loader function
    
    Returns
    -------
    
    x : list of numpy arrays
        Pixel locations where each element of the list identifies pixel
        locations in corresponding axis.
    images : numpy array
        Imaging data of size channels x slices x rows x cols, or of size
        time x 3 x slices x rows x cols for velocity fields
    title : str
        Title of the dataset (read from vtk files)        
    names : list of str
        Names of each dataset (channel or time point)
    
    Raises
    ------
    Exception
        If file type is nrrd.
    Exception
        If data is a single slice, json reader does not support it.
    Exception
        If opening with Nibabel and the affine matrix is not diagonal.
    
    '''
    # find the extension
    # if no extension use slice reader
    # if vtk use our reader
    # if nrrd use nrrd
    # otherwise try nibabel
    base,ext = os.path.splitext(fname)
    if ext == '.gz':
        base,ext_ = os.path.splitext(base)
        ext = ext_+ext
    #print(f'Found extension {ext}')
    
    if ext == '':
        x,J,W0 = load_slices(fname, xJ=x)
        images = np.concatenate((J,W0[None]))
        # set the names, I will separate out mask later
        names = ['red','green','blue','mask']
        title = 'slice_dataset'
    elif ext == '.vtk':
        x,images,title,names = read_vtk_data(fname,**kwargs)
    elif ext == '.nrrd':
        print('opening with nrrd')
        raise Exception('NRRD not currently supported')
    elif ext in ['.tif','.tiff','.jpg','.jpeg','.png']:
        # 2D image file, I can specify dx and ox
        # or I can search for a companion file
        print('opening 2D image file')
        if 'dx' not in kwargs and 'ox' not in kwargs:
            print('No geometry information provided')
            print('Searching for geometry information files')
            json_name = fname.replace(ext,'.json')
            geometry_name = join(os.path.split(fname)[0],'geometry.csv')
            if os.path.exists(json_name):
                print('Found json sidecar')
                raise Exception('json reader for single slice not implemented yet')            
            elif os.path.exists(geometry_name):                
                print('Found legacy geometry file')
                with open(geometry_name,'rt') as f:
                    for line in f:
                        if os.path.split(fname)[-1] in line:
                            #print(line)
                            parts = line.split(',')
                            # filename, nx,ny,nz,dx,dy,dz,ox,oy,oz
                            nx = np.array([int(p) for p in parts[1:4]])
                            #print(nx)
                            dx = np.array([float(p) for p in parts[4:7]])
                            #print(dx)
                            ox = np.array([float(p) for p in parts[7:10]])
                            #print(ox)
                            # change xyz to zyx
                            nx = nx[::-1]
                            dx = dx[::-1]
                            ox = ox[::-1]
                            kwargs['dx'] = dx
                            kwargs['ox'] = ox                                            
            else:
                print('did not found geomtery info, using some defaults')
        if 'dx' not in kwargs:
            warn('Voxel size dx not in keywords, using (1,1,1)')
            dx = np.array([1.0,1.0,1.0])
        if 'ox' not in kwargs:
            warn('Origin not in keywords, using 0 for z, and image center for xy')
            ox = [0.0,None,None]
        if ext in ['.tif','.tiff']:
            images = tf.imread(fname)
        else:
            images = plt.imread(fname)
        # convert to float
        if images.dtype == np.uint8:
            images = images.astype(float)/255.0
        else:
            images = images.astype(float) # this may do nothing if it is already float
            images = images / np.mean(np.abs(images.reshape(-1, images.shape[-1])), axis=0) # normalize by the mean of each channel
        # add leading dimensions and reshape, note offset may be none in dims 1 and 2.
        images = images[None].transpose(-1,0,1,2)
        nI = images.shape[1:]
        x0 = np.arange(nI[0])*dx[0] + ox[0]
        x1 = np.arange(nI[1])*dx[1]
        if ox[1] is None:
            x1 -= np.mean(x1)
        else:
            x1 += ox[1]
        x2 = np.arange(nI[2])*dx[2]
        if ox[2] is None:
            x2 -= np.mean(x2)
        else:
            x2 += ox[2]
        x = [x0,x1,x2]
        title = ''
        names = ['']            
        
        
    else:
        print('Opening with nibabel, note only 3D images supported, sform or quaternion matrix is ignored')
        vol = nibabel.load(fname,**kwargs)
        print(vol.header)
        images = np.array(vol.get_fdata())
        if images.ndim == 3:
            images = images[None]
            
        if 'normalize' in kwargs and kwargs['normalize']:
            images = images / np.mean(np.abs(images)) # normalize
        
        '''
        A = vol.header.get_base_affine()
        # NOTE: february 28, 2023.  the flipping below is causing some trouble
        # I would like to not use the A matrix at all
        
        if not np.allclose(np.diag(np.diag(A[:3,:3])),A[:3,:3]):
            raise Exception('Only support diagonal affine matrix with nibabel')
        x = [ A[i,-1] + np.arange(images.shape[i+1])*A[i,i] for i in range(3)]
        for i in range(3):
            if A[i,i] < 0:
                x[i] = x[i][::-1]
                images = np.array(np.flip(images,axis=i+1))
        '''    
        # instead we do this, whic his simpler
        d = np.array(vol.header['pixdim'][1:4],dtype=float)
        x = [np.arange(n)*d - (n-1)*d/2 for n,d in zip(images.shape[1:],d)]
        
        title = ''
        names = ['']
        
    return x,images,title,names
def write_data(fname,x,out,title,names=None):
    """ 
    Write data

    Raises
    ------
    Exception
        If data is in .nii format, it must be grayscale.
    Exception
        If output is not .vtk or .nii/.nii.gz format.

    Warns
    -----
    If data to be written uses extension .nii or .nii.gz
        Writing image in nii fomat, no title or names saved
    """
    base,ext = os.path.splitext(fname)
    if ext == '.gz':
        base,ext_ = os.path.splitext(base)
        ext = ext_+ext
    #print(f'Found extension {ext}')
    
    if ext == '.vtk':
        write_vtk_data(fname,x,out,title,names)
    elif ext == '.nii' or ext == '.nii.gz':
        if type(out) == torch.Tensor:
            out = out.cpu().numpy()
        if out.ndim == 4 and out.shape[0]==1:
            out = out[0]
        if out.ndim >= 4:
            raise Exception('Only grayscale images supported in nii format')
            
        affine = np.diag((x[0][1]-x[0][0],x[1][1]-x[1][0],x[2][1]-x[2][0],1.0))
        affine[:3,-1] = np.array((x[0][0],x[1][0],x[2][0]))
        img = nibabel.Nifti1Image(out, affine)
        nibabel.save(img, fname)  
        warn('Writing image in nii fomat, no title or names saved')

        
    else:
        raise Exception('Only vtk and .nii/.nii.gz outputs supported')
        
    

def write_matrix_data(fname,A):
    '''
    Write linear transforms as matrix text file.
    Note that in python we use zyx order, 
    but we write outputs in xyz order
    
    Parameter
    ---------
    fname : str
        Filename to write
    A : 2D array
        Matrix data to write. Assumed to be in zyx order.
        
    Returns
    -------
    None
    '''
    # copy the matrix
    A_ = np.zeros((A.shape[0],A.shape[1]))    
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            A_[i,j] = A[i,j]
    
    # swap zyx -> xyz, accounting for affine
    A_[:-1] = A_[:-1][::-1]
    A_[:,:-1] = A_[:,:-1][:,::-1]
    with open(fname,'wt') as f:
        for i in range(A_.shape[0]):
            for j in range(A_.shape[1]):                
                f.write(f'{A_[i,j]}')
                if j < A_.shape[1]-1:
                    f.write(', ')
            f.write('\n')

def read_matrix_data(fname):
    '''
    Read linear transforms as matrix text file.
    Note in python we work in zyx order, but text files are in xyz order
    
    Parameters
    ----------
    fname : str
    
    Returns
    -------
    A : array
        matrix in zyx order
    '''
    A = np.zeros((4,4))
    with open(fname,'rt') as f:
        i = 0
        for line in f:            
            if ',' in line:
                # we expect this to be a csv
                for j,num in enumerate(line.split(',')):
                    A[i,j] = float(num)
            else:
                # it may be separated by spaces
                for j,num in enumerate(line.split(' ')):
                    A[i,j] = float(num)
            i += 1
    
    # if it is 3x3, then i is 3
    A = A[:i,:i]
    # swap xyz -> zyx, accounting for affine
    A[:-1] = A[:-1][::-1]
    A[:,:-1] = A[:,:-1][:,::-1]
    return np.copy(A) # make copy to avoid negative strides



# write vtk point data
def write_vtk_polydata(fname,name,points,connectivity=None,connectivity_type=None):
    '''
    points should by Nx3 in zyx order
    It will be written out in xyz order
    connectivity should be lists of indices or nothing to write only cell data
    
    Parameters
    ----------
    fname : str
        Filename to write
    name : str
        Dataset name
    points : array
        
    connectivity : str
        Array of arrays storing each connectivity element as integers that refer to the points, 
        size number of points by number of dimensions (expected to be 3)
    connectivity_type : str
        Can by VERTICES, or POLYGONS, or LINES
    
    Returns
    -------
    nothing
    
    '''
    # first we'll open the file
    with open(fname,'wt') as f:
        f.write('# vtk DataFile Version 2.0\n')
        f.write(f'{name}\n')
        f.write('ASCII\n')
        f.write('DATASET POLYDATA\n')
        f.write(f'POINTS {points.shape[0]} float\n')
        for i in range(points.shape[0]):
            f.write(f'{points[i][-1]} {points[i][-2]} {points[i][-3]}\n')
                 
        if (connectivity is None) or (connectivity_type.upper() == 'VERTICES'):
            # lets try to add vertices
            # the second number is how many numbers are below
            # there is one extra number per line
            f.write(f'VERTICES {points.shape[0]} {points.shape[0]*2}\n')            
            for i in range(points.shape[0]):
                f.write(f'1 {i}\n')
        elif connectivity_type.upper() == 'POLYGONS':
            nlines = len(connectivity)
            ntot = 0
            for c in connectivity:
                ntot += len(c)
            f.write(f'POLYGONS {nlines} {ntot+nlines}\n')
            for line in connectivity:
                f.write(f'{len(line)} ')
                for l in line:
                    f.write(f'{l} ')
                f.write('\n')
                pass
        elif connectivity_type.upper() == 'LINES':
            nlines = len(connectivity)
            ntot = 0
            for c in connectivity:
                ntot += len(c)
            f.write(f'LINES {nlines} {ntot+nlines}\n')
            for line in connectivity:
                f.write(f'{len(line)} ')
                for l in line:
                    f.write(f'{l} ')
                f.write('\n')
                

def read_vtk_polydata(fname):
    
    '''
    Read ascii vtk polydata from simple legacy files.
    Assume file contains xyz order, they are converted to zyx for python
    
    Parameters
    ----------
    fname : str
        The name of the file to read
        
    Returns
    -------
    points : numpy float array
        nx3 array storing locations of points
    connectivity : list of lists
        list of indices containing connectivity elements
    connectivity_type : str
        VERTICES or LINES or POLYGONS
    name : str
        name of the dataset
        
    
    '''
    
    with open(fname,'rt') as f:
        points = []
        connectivity = []
        connectivity_type = ''
        point_counter = -1
        connectivity_counter = -1
        count = 0
        for line in f:
            #print(line)
            if count == 1:
                # this line has name
                name = line.strip()
            if 'POINTS' in line.upper() and count > 1:
                # we need to make sure we're not detecting the title
                #print(f'found points {line}')
                parts = line.split()
                npoints = int(parts[1])
                dtype = parts[2] # not using
                point_counter = 0
                continue
            
            if point_counter >= 0 and point_counter < npoints:
                points.append(np.array([float(p) for p in reversed(line.split())])) # xyz -> zyx
                point_counter += 1
                if point_counter == npoints:
                    point_counter = -2
                    continue
                    
            if point_counter == -2 and connectivity_counter == -1:                
                # next line should say connectivity type
                parts = line.split()
                connectivity_type = parts[0]
                # next number should say number of connectivity entries
                n_elements = int(parts[1])
                n_indices = int(parts[2])
                connectivity_counter = 0
                continue
            
            if connectivity_counter >= 0 and connectivity_counter < n_elements:
                # the first number specifies how many numbers follow
                connectivity.append([int(i) for i in line.split()[1:]])
            
            count += 1
                
    
    return np.stack(points),connectivity,connectivity_type,name


def fnames(path):
    ''' Get a list of image file names for 2D series, or a single file name for volume image.

    Returns
    -------
    fnames : list of strings
        List of image file names
    '''
    if os.path.splitext(path)[1] == '':
        samples_tsv = os.path.join(path, "samples.tsv")
        fnames = []
        with open(samples_tsv,'rt') as f:
            for count,line in enumerate(f):
                line = line.strip()
                key = '\t' if '\t' in line else '    '
                if count == 0:
                    continue
                fnames.append(os.path.splitext(re.split(key,line)[0])[0])
    else:
        fnames = [path]

    return fnames


def write_transform_outputs_(output_dir, output, I, J):
    '''
    Note this version (whose function name ends in "_" ) is obsolete.
    Write transforms output from emlddmm.  Velocity field, 3D affine transform,
    and 2D affine transforms for each slice if applicable.
    
    Parameters
    ----------
    output_dir : str
        Directory to place output data (will be created of it does not exist)
    output : dict
        Output dictionary from emlddmm algorithm
    A2d_names : list 
        List of file names for A2d transforms
    
    Returns
    -------
    None

    Todo
    ----
    Update parameter list.
    
    '''
    xv = output['xv']
    v = output['v']
    A = output['A']
    slice_outputs = 'A2d' in output
    if slice_outputs:
        A2d = output['A2d']
    
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    series_to_series = I.title == 'slice_dataset' and J.title == 'slice_dataset'
    if series_to_series:
        out = os.path.join(output_dir, f'{I.space}_input/{J.space}_{J.name}_input_to_{I.space}_input/transforms/')
        if not os.path.isdir(out):
            os.makedirs(out)
        A2d_names = []
        for i in range(A2d.shape[0]):
            A2d_names.append(f'{I.space}_input_{I.fnames()[i]}_to_{J.space}_input_{J.fnames()[i]}_matrix.txt')
        for i in range(A2d.shape[0]):
            output_name = os.path.join(out, A2d_names[i])
            write_matrix_data(output_name, A2d[i])

        return
    
    if slice_outputs:
        out3d = os.path.join(output_dir, f'{I.space}/{J.space}_{J.name}_registered_to_{I.space}/transforms/')
        out2d = os.path.join(output_dir, f'{J.space}_registered/{J.space}_input_to_{J.space}_registered/transforms')
        if not os.path.isdir(out2d):
            os.makedirs(out2d)
    else:
        out3d = os.path.join(output_dir, f'{I.space}/{J.space}_{J.name}_to_{I.space}/transforms/')

    if not os.path.isdir(out3d):
        os.makedirs(out3d)
        
    output_name = os.path.join(out3d, 'velocity.vtk')
    title = 'velocity_field'
    write_vtk_data(output_name,xv,v.cpu().numpy(),title)  
    output_name = os.path.join(out3d, 'A.txt')
    write_matrix_data(output_name,A)
    if slice_outputs:
        A2d_names = []
        for i in range(A2d.shape[0]):
            A2d_names.append(f'{J.space}_registered_{J.fnames()[i]}_to_{J.space}_input_{J.fnames()[i]}_matrix.txt')
        for i in range(A2d.shape[0]):
            output_name = os.path.join(out2d, A2d_names[i])
            write_matrix_data(output_name, A2d[i])

    return

def write_transform_outputs(output_dir, output, I, J):
    '''
    Note, daniel is redoing the above slightly. on March 2023

    Write transforms output from emlddmm.  Velocity field, 3D affine transform,
    and 2D affine transforms for each slice if applicable.
    
    Parameters
    ----------
    output_dir : str
        Directory to place output data (will be created of it does not exist)
    output : dict
        Output dictionary from emlddmm algorithm
    A2d_names : list 
        List of file names for A2d transforms
    
    Returns
    -------
    None

    Todo
    ----
    Update parameter list.
    
    '''
    xv = output['xv']
    v = output['v']
    A = output['A']
    slice_outputs = 'A2d' in output
    if slice_outputs:
        A2d = output['A2d']
    
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    series_to_series = I.title == 'slice_dataset' and J.title == 'slice_dataset'
    if series_to_series:
        #out = os.path.join(output_dir, f'{I.space}_input/{J.space}_{J.name}_input_to_{I.space}_input/transforms/')
        # danel thinks the name shouldn't go here
        # it should be clear from the filenames
        #out = os.path.join(output_dir, f'{I.space}_input/{J.space}_input_to_{I.space}_input/transforms/')
        out = os.path.join(output_dir, f'{I.space}/{J.space}_to_{I.space}/transforms/')
        if not os.path.isdir(out):
            os.makedirs(out)
        A2d_names = []
        for i in range(A2d.shape[0]):
            #A2d_names.append(f'{I.space}_input_{I.fnames()[i]}_to_{J.space}_input_{J.fnames()[i]}_matrix.txt')
            A2d_names.append(f'{I.space}_{I.fnames()[i]}_to_{J.space}_{J.fnames()[i]}_matrix.txt')
        for i in range(A2d.shape[0]):
            output_name = os.path.join(out, A2d_names[i])
            write_matrix_data(output_name, A2d[i])

        return
    
    if slice_outputs:        
        #out3d = os.path.join(output_dir, f'{I.space}/{J.space}_{J.name}_registered_to_{I.space}/transforms/')
        # again daniel says no name here
        out3d = os.path.join(output_dir, f'{I.space}/{J.space}_registered_to_{I.space}/transforms/')        
        #out2d = os.path.join(output_dir, f'{J.space}_registered/{J.space}_input_to_{J.space}_registered/transforms')
        out2d = os.path.join(output_dir, f'{J.space}_registered/{J.space}_to_{J.space}_registered/transforms')
        if not os.path.isdir(out2d):
            os.makedirs(out2d)
    else:        
        #out3d = os.path.join(output_dir, f'{I.space}/{J.space}_{J.name}_to_{I.space}/transforms/')
        # again daniel says no name here
        out3d = os.path.join(output_dir, f'{I.space}/{J.space}_to_{I.space}/transforms/')

    if not os.path.isdir(out3d):
        os.makedirs(out3d)
        
    output_name = os.path.join(out3d, 'velocity.vtk')
    title = 'velocity_field'
    write_vtk_data(output_name,xv,v.cpu().numpy(),title)  
    output_name = os.path.join(out3d, 'A.txt')
    write_matrix_data(output_name,A)
    if slice_outputs:
        A2d_names = []
        for i in range(A2d.shape[0]):
            #A2d_names.append(f'{J.space}_registered_{J.fnames()[i]}_to_{J.space}_input_{J.fnames()[i]}_matrix.txt')
            A2d_names.append(f'{J.space}_registered_{J.fnames()[i]}_to_{J.space}_{J.fnames()[i]}_matrix.txt')
        for i in range(A2d.shape[0]):
            output_name = os.path.join(out2d, A2d_names[i])
            write_matrix_data(output_name, A2d[i])

    return    


def write_qc_outputs(output_dir, output, I, J, xS=None, S=None):
    ''' write out registration qc images

    Parameters
    ----------
    output_dir : string
        Path to output parent directory
    output : dict
        Output dictionary from emlddmm algorithm
    I : emlddmm image
        source image
    J : emlddmm image
        target image
    xS : list of arrays, optional
        Label coordinates
    S : array, optional
        Labels
    
    Returns
    -------
    None


    '''

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    #print(f'output dir is {output_dir}')
    
    xv = [x.to('cpu') for x in output['xv']]
    v = output['v'].detach().to('cpu')
    A = output['A'].detach().to('cpu')
    Ai = torch.inverse(A)
    #print(A.device)
    device = A.device
    dtype = A.dtype
    
    # to torch
    Jdata = torch.as_tensor(J.data,dtype=dtype,device=device)
    xJ = [torch.as_tensor(x,dtype=dtype,device=device) for x in J.x]
    Idata = torch.as_tensor(I.data,dtype=dtype,device=device)
    xI = [torch.as_tensor(x,dtype=dtype,device=device) for x in I.x]
    if S is not None: # segmentations go with atlas, they are integers
        S = torch.as_tensor(S,device=device,dtype=dtype) 
        # don't specify dtype here, you had better set it in numpy
        # actually I need it as float in order to apply interp
        if xS is not None:
            xS = [torch.as_tensor(x,dtype=dtype,device=device) for x in xI]
            
    XJ = torch.stack(torch.meshgrid(xJ, indexing='ij'))
    slice_matching = 'A2d' in output
    if slice_matching:
        A2d = output['A2d'].detach().to('cpu')
        A2di = torch.inverse(A2d)
        XJ_ = torch.clone(XJ)           

        XJ_[1:] = ((A2di[:,None,None,:2,:2]@ (XJ[1:].permute(1,2,3,0)[...,None]))[...,0] + A2di[:,None,None,:2,-1]).permute(3,0,1,2)            
        # if registering series to series
        if I.title=='slice_dataset' and J.title=='slice_dataset':
            # first I to J
            AI = interp(xI, Idata, XJ_)
            fig = draw(AI,xJ)
            fig[0].suptitle(f'{I.space}_{I.name}_input_to_{J.space}_input')
            #out = os.path.join(output_dir,f'{J.space}_input/{I.space}_{I.name}_input_to_{J.space}_input/qc/')
            out = os.path.join(output_dir,f'{J.space}/{I.space}_to_{J.space}/qc/')
            if not os.path.isdir(out):
                os.makedirs(out)
            fig[0].savefig(out + f'{I.space}_{I.name}_to_{J.space}.jpg')
            fig = draw(Jdata, xJ)
            fig[0].suptitle(f'{J.space}_{J.name}')
            fig[0].savefig(out + f'{J.space}_{J.name}.jpg')
            # now J to I
            XI = torch.stack(torch.meshgrid(xI, indexing='ij'))
            XI_ = torch.clone(XI)
            XI_[1:] = ((A2d[:,None,None,:2,:2]@ (XI[1:].permute(1,2,3,0)[...,None]))[...,0] + A2d[:,None,None,:2,-1]).permute(3,0,1,2)
            AiJ = interp(xJ,Jdata,XI_)
            fig = draw(AiJ,xI)
            fig[0].suptitle(f'{J.space}_{J.name}_to_{I.space}')
            #out = os.path.join(output_dir,f'{I.space}_input/{J.space}_{J.name}_input_to_{I.space}_input/qc/')
            out = os.path.join(output_dir,f'{I.space}/{J.space}_to_{I.space}/qc/')
            if not os.path.isdir(out):
                os.makedirs(out)
            fig[0].savefig(out + f'{J.space}_{J.name}_to_{I.space}.jpg')
            fig = draw(Idata,xI)
            fig[0].suptitle(f'{I.space}_{I.name}')
            fig[0].savefig(out + f'{I.space}_{I.name}.jpg')

            return
            
    else:
        XJ_ = XJ

    # sample points for affine
    Xs = ((Ai[:3,:3]@XJ_.permute((1,2,3,0))[...,None])[...,0] + Ai[:3,-1]).permute((3,0,1,2))
    # for diffeomorphism
    XV = torch.stack(torch.meshgrid(xv, indexing='ij'))
    phii = v_to_phii(xv,v)
    phiiAi = interp(xv,phii-XV,Xs) + Xs
    # transform image
    AphiI = interp(xI,Idata,phiiAi)
    # target space
    if slice_matching:
        fig = draw(AphiI,xJ)
        fig[0].suptitle(f'{I.space}_{I.name}_to_{J.space}_input')
        #out = os.path.join(output_dir,f'{J.space}_input/{I.space}_{I.name}_to_{J.space}_input/qc/')
        out = os.path.join(output_dir,f'{J.space}/{I.space}_to_{J.space}/qc/')
        if not os.path.isdir(out):
            os.makedirs(out)
        fig[0].savefig(out + f'{I.space}_{I.name}_to_{J.space}.jpg')
        
        fig = draw(Jdata,xJ)
        fig[0].suptitle(f'{J.space}_{J.name}')
        fig[0].savefig(out + f'{J.space}_{J.name}.jpg')
        
        # modify XJ by shifting by mean translation
        mean_translation = torch.mean(A2d[:,:2,-1], dim=0)
        print(f'mean_translation: {mean_translation}')
        XJr = torch.clone(XJ)
        XJr[1:] -= mean_translation[...,None,None,None]
        xJr = [xJ[0], xJ[1] - mean_translation[0], xJ[2] - mean_translation[1]]
        XJr_ = torch.clone(XJr)
        XJr_[1:] = ((A2d[:,None,None,:2,:2]@ (XJr[1:].permute(1,2,3,0)[...,None]))[...,0] + A2d[:,None,None,:2,-1]).permute(3,0,1,2)
        Jr = interp(xJ,Jdata,XJr_)
        fig = draw(Jr,xJr)
        fig[0].suptitle(f'{J.space}_{J.name}_registered')
        #out = os.path.join(output_dir,f'{J.space}_registered/{I.space}_{I.name}_to_{J.space}_registered/qc/')
        out = os.path.join(output_dir,f'{J.space}_registered/{I.space}_to_{J.space}_registered/qc/')
        if not os.path.isdir(out):
            os.makedirs(out)
        fig[0].savefig(out + f'{J.space}_{J.name}_registered.jpg')
        
        # and we need atlas reconstructed in target space

        # sample points for affine
        Xs = ((Ai[:3,:3]@XJr.permute((1,2,3,0))[...,None])[...,0] + Ai[:3,-1]).permute((3,0,1,2))
        # for diffeomorphism
        XV = torch.stack(torch.meshgrid(xv, indexing='ij'))
        phiiAi = interp(xv,phii-XV,Xs) + Xs

        # transform image
        AphiI = interp(xI,Idata,phiiAi)
        fig = draw(AphiI, xJ)
        fig[0].suptitle(f'{I.space}_{I.name}_to_{J.space}_registered')
        fig[0].savefig(out + f'{I.space}_{I.name}_to_{J.space}_registered.jpg')
    else:
        fig = draw(AphiI,xJ)
        fig[0].suptitle(f'{I.space}_{I.name}_to_{J.space}')
        #out = os.path.join(output_dir,f'{J.space}/{I.space}_{I.name}_to_{J.space}/qc/')
        out = os.path.join(output_dir,f'{J.space}/{I.space}_to_{J.space}/qc/')
        if not os.path.isdir(out):
            os.makedirs(out)
        fig[0].savefig(out + f'{I.space}_{I.name}_to_{J.space}.jpg')
        fig = draw(Jdata,xJ)
        fig[0].suptitle(f'{J.space}_{J.name}')
        fig[0].savefig(out + f'{J.space}_{J.name}.jpg')
        Jr = Jdata

    # and source space
    XI = torch.stack(torch.meshgrid(xI, indexing='ij'))
    phi = v_to_phii(xv,-v.flip(0))
    Aphi = ((A[:3,:3]@phi.permute((1,2,3,0))[...,None])[...,0] + A[:3,-1]).permute((3,0,1,2))
    Aphi = interp(xv,Aphi,XI)
    # apply the shift to Aphi since it was subtracted when creating Jr
    if slice_matching:
        Aphi[1:] += mean_translation[...,None,None,None]
    phiiAiJ = interp(xJ,Jr,Aphi)

    fig = draw(phiiAiJ,xI)
    fig[0].suptitle(f'{J.space}_{J.name}_to_{I.space}')
    if slice_matching:
        #out = os.path.join(output_dir,f'{I.space}/{J.space}_{J.name}_registered_to_{I.space}/qc/')
        out = os.path.join(output_dir,f'{I.space}/{J.space}_registered_to_{I.space}/qc/')
        if not os.path.isdir(out):
            os.makedirs(out)
    else:
        #out = os.path.join(output_dir,f'{I.space}/{J.space}_{J.name}_to_{I.space}/qc/')
        out = os.path.join(output_dir,f'{I.space}/{J.space}_to_{I.space}/qc/')
        if not os.path.isdir(out):
            os.makedirs(out)
    fig[0].savefig(out + f'{J.space}_{J.name}_to_{I.space}.jpg' )


    fig = draw(Idata,xI)
    fig[0].suptitle(f'{I.space}_{I.name}')
    fig[0].savefig(out + f'{I.space}_{I.name}.jpg')

    output_slices = slice_matching and ( (xS is not None) and (S is not None))
    if output_slices:
        # transform S
        # note here I had previously converted it to float
        AphiS = interp(xS,torch.tensor(S,device=device,dtype=dtype),phiiAi,mode='nearest').cpu().numpy()[0]

        mods = (7,11,13)
        R = (AphiS%mods[0])/mods[0]
        G = (AphiS%mods[1])/mods[1]
        B = (AphiS%mods[2])/mods[2]
        fig = draw(np.stack((R,G,B)),xJ)

        # also outlines
        M = np.zeros_like(AphiS)
        r = 1
        for i in [0]:#range(-r,r+1): # in the coronal plane my M is "nice"
            for j in range(-r,r+1):
                for k in range(-r,r+1):
                    if (i**2 + j**2 + k**2) > r**2:
                        continue
                    M = np.logical_or(M,np.roll(AphiS,shift=(i,j,k),axis=(0,1,2))!=AphiS)
        #fig = draw(M[None])

        C = np.stack((R,G,B))*M
        q = (0.01,0.99)
        c = np.quantile(J.cpu().numpy(),q)
        Jn = (Jr.cpu().numpy() - c[0])/(c[1]-c[0])
        Jn[Jn<0] = 0
        Jn[Jn>1] = 1
        alpha = 0.5
        show_ = Jn[0][None]*(1-M[None]*alpha) + M[None]*C*alpha
        
        f,ax = plt.subplots()
        for s in range(show_.shape[1]):
            ax.cla()
            ax.imshow(show_[:,s].transpose(1,2,0))
            ax.set_xticks([])
            ax.set_yticks([])

            f.savefig(os.path.join(output_dir,f'slice_{s:04d}.jpg'))
            #f.savefig(join(to_registered_out,f'{dest_space}_{dest_img}_to_{src_space}_REGISTERED_{slice_names[i]}.jpg'))    
            # notte the above line came up in a merge conflict on march 10, 2023.  We'll consider it later.
    return





def write_outputs_for_pair(output_dir,outputs,
                           xI,I,xJ,J,WJ=None,
                           atlas_space_name=None,target_space_name=None,
                           atlas_image_name=None,target_image_name=None):
    # TODO: daniel double check this after bryson name changes (remove input)
    '''
    Write outputs in standard format for a pair of images
    
    Parameters
    ----------
    output_dir : str
        Location to store output data.
    outputs : dict
        Dictionary of outputs from the emlddmm python code
    xI : list of numpy array
        Location of voxels in atlas
    I : numpy array
        Atlas image
    xJ : list of numpy array
        Location of voxels in target
    J : numpy array
        Target image
    atlas_space_name : str
        Name of atlas space (default 'atlas')
    target_space_name : str
        Name of target space (default 'target')
    atlas_image_name : str
        Name of atlas image (default 'atlasimage')
    target_image_name : str
        Name of target image (default 'targetimage')
        
    
    
        
    TODO
    ----
    Implement QC figures.
    
    Check device more carefully, probably better to put everything on cpu.
    
    Check dtype more carefully.
    
    Get determinant of jacobian. (done)
    
    Write velocity out and affine
    
    Get a list of files to name outputs.
    
    
    
        
        
    
    '''    
    if atlas_space_name is None:
        atlas_space_name = 'atlas'
    if target_space_name is None:
        target_space_name = 'target'
    if atlas_image_name is None:
        atlas_image_name = 'atlasimage'
    if target_image_name is None:
        target_image_name = 'targetimage'
           
        
    if 'A2d' in outputs:
        slice_matching = True
    else:
        slice_matching = False
        
    # we will make everything float and cpu
    device = 'cpu'
    dtype = outputs['A'].dtype
    I = torch.tensor(I,device=device,dtype=dtype)
    xI = [torch.tensor(x,device=device,dtype=dtype) for x in xI]
    J = torch.tensor(J,device=device,dtype=dtype)
    if WJ is not None:
        WJ = torch.tensor(WJ,device=device,dtype=dtype)
    xJ = [torch.tensor(x,device=device,dtype=dtype) for x in xJ]
        
    exist_ok=True
    IMAGES = 'images' # define some strings
    TRANSFORMS = 'transforms'
    # make the output directory
    # note this function will do it recursively even if it exists
    os.makedirs(output_dir,exist_ok=exist_ok)
    # to atlas space
    to_space_name = atlas_space_name
    to_space_dir = join(output_dir,to_space_name)
    os.makedirs(to_space_dir, exist_ok=exist_ok)
    # to atlas space from atlas space 
    from_space_name = atlas_space_name
    from_space_dir = join(to_space_dir,f'{from_space_name}_to_{to_space_name}')
    os.makedirs(from_space_dir, exist_ok=exist_ok)
    # to atlas space from atlas space
    images_dir = join(from_space_dir,IMAGES)
    os.makedirs(images_dir, exist_ok=exist_ok)
    # write out the atlas (in single)
    write_data(join(images_dir,f'{atlas_space_name}_{atlas_image_name}_to_{atlas_space_name}.vtk'),
               xI,torch.tensor(I,dtype=dtype),'atlas')
    
    if not slice_matching:        
        print(f'writing NOT slice matching outputs')
        # to atlas space from target space    
        from_space_name = target_space_name
        from_space_dir = join(to_space_dir,f'{from_space_name}_to_{to_space_name}')
        os.makedirs(from_space_dir, exist_ok=exist_ok)
        # to atlas space from target space transforms    
        transforms_dir = join(from_space_dir,TRANSFORMS)
        os.makedirs(transforms_dir,exist_ok=exist_ok)
        # need atlas to target displacement
        # this is A phi - x
        XV = torch.stack(torch.meshgrid(outputs['xv'],indexing='ij'),0)
        phi = v_to_phii(outputs['xv'],-1*torch.flip(outputs['v'],dims=(0,))).cpu()
        XI = torch.stack(torch.meshgrid([torch.tensor(x,device=phi.device,dtype=phi.dtype) for x in xI],indexing='ij'),0)
        phiXI = interp([x.cpu() for x in outputs['xv']],phi.cpu()-XV.cpu(),XI.cpu()) + XI.cpu()
        AphiXI = ((outputs['A'].cpu()[:3,:3]@phiXI.permute(1,2,3,0)[...,None])[...,0] + outputs['A'][:3,-1].cpu()).permute(-1,0,1,2)    
        write_data(join(transforms_dir,f'{to_space_name}_to_{from_space_name}_displacement.vtk'),xI,(AphiXI-XI)[None].to(torch.float32),f'{to_space_name}_to_{from_space_name}_displacement')
        # now detjac        
        dxI = [ (x[1] - x[0]).item() for x in xI]         
        detjac = torch.linalg.det(
            torch.stack(
                torch.gradient(AphiXI,spacing=dxI,dim=(1,2,3))
            ).permute(2,3,4,0,1)
        )
        write_data(join(transforms_dir,f'{to_space_name}_to_{from_space_name}_detjac.vtk'),
                   xI,detjac[None].to(torch.float32),f'{to_space_name}_to_{from_space_name}_detjac')
        # now we need the target to atlas image
        images_dir = join(from_space_dir,IMAGES)
        os.makedirs(images_dir,exist_ok=exist_ok)
        # this is generated form the target with AphiXI (make sure on same device, cpu)
        phiiAiJ = interp(xJ,J,AphiXI.cpu(),padding_mode='zeros')
        write_data(join(images_dir,f'{from_space_name}_{target_image_name}_to_{to_space_name}.vtk'),
                   xI,phiiAiJ.to(torch.float32),f'{from_space_name}_{target_image_name}_to_{to_space_name}')
        
        # qc? TODO
        
        # now to target space
        to_space_name = target_space_name
        to_space_dir = join(output_dir,to_space_name)
        os.makedirs(to_space_dir,exist_ok=exist_ok)
        # from target space (i.e identity)
        from_space_name = target_space_name
        from_space_dir = join(to_space_dir,f'{from_space_name}_to_{to_space_name}')
        os.makedirs(from_space_dir,exist_ok=exist_ok)
        # target image
        images_dir = join(from_space_dir,IMAGES)
        os.makedirs(images_dir, exist_ok=exist_ok)
        # write out the target (in single)
        write_data(join(images_dir,f'{target_space_name}_{target_image_name}_to_{target_space_name}.vtk'),
                   xJ,torch.tensor(J,dtype=torch.float32),'atlas')
        # now from atlas space
        from_space_name = atlas_space_name
        from_space_dir = join(to_space_dir,f'{from_space_name}_to_{to_space_name}')
        os.makedirs(from_space_dir,exist_ok=exist_ok)
        # to target space from atlas space transforms        
        transforms_dir = join(from_space_dir,TRANSFORMS)
        os.makedirs(transforms_dir,exist_ok=exist_ok)
        phii = v_to_phii(outputs['xv'],outputs['v'])
        XJ = torch.stack(torch.meshgrid([torch.tensor(x,device=phii.device,dtype=phii.dtype) for x in xJ],indexing='ij'),0)
        Ai = torch.linalg.inv(outputs['A'])
        AiXJ = ((Ai[:3,:3]@XJ.permute(1,2,3,0)[...,None])[...,0] + Ai[:3,-1]).permute(-1,0,1,2)
        phiiAiXJ = interp([x.cpu() for x in outputs['xv']],(phii-XV).cpu(),AiXJ.cpu()) + AiXJ.cpu()
        write_data(join(transforms_dir,f'{to_space_name}_to_{from_space_name}_displacement.vtk'),xJ,(phiiAiXJ.cpu()-XJ.cpu())[None].to(torch.float32),f'{to_space_name}_to_{from_space_name}_displacement')
        dxJ = [x[1] - x[0] for x in xJ] 
        detjac = torch.linalg.det(
            torch.stack(
                torch.gradient(phiiAiXJ,spacing=dxI,dim=(1,2,3))
            ).permute(2,3,4,0,1)
        )
        write_data(join(transforms_dir,f'{to_space_name}_to_{from_space_name}_detjac.vtk'),
                   xJ,detjac[None].to(torch.float32),f'{to_space_name}_to_{from_space_name}_detjac')
        # write out velocity field
        write_data(join(transforms_dir,f'velocity.vtk'),outputs['xv'],outputs['v'].to(torch.float32), f'{to_space_name}_to_{from_space_name}_velocity')
        # write out affine
        write_matrix_data(join(transforms_dir,f'A.txt'),outputs['A'])
        # atlas image
        images_dir = join(from_space_dir,IMAGES)
        os.makedirs(images_dir,exist_ok=exist_ok)
        AphiI = interp([torch.as_tensor(x).cpu() for x in xI],torch.as_tensor(I).cpu(),phiiAiXJ.cpu(),padding_mode='zeros')
        write_data(join(images_dir,f'{atlas_space_name}_{atlas_image_name}_to_{target_space_name}.vtk'),
                   xJ,torch.tensor(J,dtype=torch.float32),f'{atlas_space_name}_{atlas_image_name}_to_{target_space_name}')
        # TODO qc   
        
    else: # if slice matching
        print(f'writing slice matching transform outputs')
        # WORKING HERE TO MAKE SURE I HAVE THE RIGHT OUTPUTS
        # NOTE: in registered space we may need to sample in a different spot if origin is not in the middle


        # to atlas space from registered target space
        from_space_name = target_space_name + '-registered'
        from_space_dir = join(to_space_dir,f'{from_space_name}_to_{to_space_name}')
        os.makedirs(from_space_dir, exist_ok=exist_ok)
        # to atlas space from registered target space transforms    
        transforms_dir = join(from_space_dir,TRANSFORMS)
        os.makedirs(transforms_dir, exist_ok=exist_ok)
        # we need the atlas to registered displacement
        # this is  A phi - x
        XV = torch.stack(torch.meshgrid(outputs['xv'],indexing='ij'),0)
        phi = v_to_phii(outputs['xv'],-1*torch.flip(outputs['v'],dims=(0,)))
        XI = torch.stack(torch.meshgrid([torch.tensor(x,device=phi.device,dtype=phi.dtype) for x in xI],indexing='ij'),0)
        phiXI = interp(outputs['xv'],phi-XV,XI) + XI            
        AphiXI = ((outputs['A'][:3,:3]@phiXI.permute(1,2,3,0)[...,None])[...,0] + outputs['A'][:3,-1]).permute(-1,0,1,2)                
        write_data(join(transforms_dir,f'{to_space_name}_to_{from_space_name}_displacement.vtk'),xI,(AphiXI-XI)[None].to(torch.float32),f'{to_space_name}_to_{from_space_name}_displacement')
        
        # to atlas space from registered space images
        # nothing here because no images were acquired in this space
        
        
        
        # to atlas space from input target space
        from_space_name = target_space_name
        from_space_dir = join(to_space_dir,f'{from_space_name}_to_{to_space_name}')
        os.makedirs(from_space_dir, exist_ok=exist_ok)
        # to atlas space from input target space transforms    
        # THESE TRANSFORMS DO NOT EXIST
        # to atlas space from input target space images
        images_dir = join(from_space_dir,IMAGES)
        os.makedirs(images_dir, exist_ok=exist_ok)
        # to get these images I first need to map them to registered
        XJ = torch.stack(torch.meshgrid([torch.tensor(x,device=phi.device,dtype=phi.dtype) for x in xJ], indexing='ij'))        
        R = outputs['A2d']
        # TODO: implement a mean shift
        meanshift = torch.mean(R[:,:2,-1],dim=0)
        
        XJshift = XJ.clone()
        XJshift[1:] -= meanshift[...,None,None,None]
        
        xJshift = [xJ[0],xJ[1]-meanshift[0],xJ[2]-meanshift[1]]
        # add mean shift below ( I added but didn't change names)
        RXJ = ((R[:,None,None,:2,:2]@(XJshift[1:].permute(1,2,3,0)[...,None]))[...,0] + R[:,None,None,:2,-1]).permute(-1,0,1,2)
        RXJ = torch.cat((XJ[0][None],RXJ))
        RiJ = interp([torch.tensor(x,device=RXJ.device,dtype=RXJ.dtype) for x in xJ],torch.tensor(J,device=RXJ.device,dtype=RXJ.dtype),RXJ,padding_mode='zeros')
        phiiAiRiJ = interp([torch.tensor(x,device=RXJ.device,dtype=RXJ.dtype) for x in xJshift],RiJ,AphiXI,padding_mode='zeros')
        write_data(join(images_dir,f'{from_space_name}_{target_image_name}_to_{to_space_name}.vtk'),
                   xI,phiiAiRiJ.to(torch.float32),f'{from_space_name}_{target_image_name}_to_{to_space_name}')
        
        # qc in atlas space
        qc_dir = join(to_space_dir,'qc')
        os.makedirs(qc_dir,exist_ok=exist_ok)        
        fig,ax = draw(phiiAiRiJ,xI)
        fig.suptitle('phiiAiRiJ')
        fig.savefig(join(qc_dir,f'{from_space_name}_{target_image_name}_to_{to_space_name}.jpg'))
        fig.canvas.draw()
        fig,ax = draw(I,xI)
        fig.suptitle('I')
        fig.savefig(join(qc_dir,f'{to_space_name}_{atlas_image_name}_to_{to_space_name}.jpg'))
        fig.canvas.draw()
    
        # now we do to registered space
        # TODO: implement proper registered space sample points
        # note that I'm applying transformations to XJ, but I should apply them to XJ with the mean shift
        # 
        
        
        to_space_name = target_space_name + '-registered'
        to_space_dir = join(output_dir,to_space_name)
        os.makedirs(to_space_dir,exist_ok=exist_ok)
        # from input
        from_space_name = target_space_name
        from_space_dir = join(to_space_dir,f'{from_space_name}_to_{to_space_name}')
        os.makedirs(from_space_dir,exist_ok=exist_ok)
        # transforms, registered to input
        transforms_dir = join(from_space_dir,TRANSFORMS)
        os.makedirs(transforms_dir,exist_ok=exist_ok)
        #for i in range(R.shape[0]):
        #    # convert to xyz
        #    Rxyz = torch.tensor(R[i])
        #    Rxyz[:2] = torch.flip(Rxyz[:2],dims=(0,))
        #    Rxyz[:,:2] = torch.flip(Rxyz[:,:2],dims=(1,))
        #    # write out
        #    with open(os.path.join(from_space_dir)):
        write_data(join(transforms_dir,f'{to_space_name}_to_{from_space_name}_displacement.vtk'),
                  xJshift,(RXJ-XJshift)[None].to(torch.float32), f'{to_space_name}_to_{from_space_name}')
        # images
        images_dir = join(from_space_dir,IMAGES)
        os.makedirs(images_dir,exist_ok=exist_ok)
        write_data(join(images_dir,f'{from_space_name}_{target_image_name}_to_{to_space_name}.vtk'),
                  xJshift,(RiJ).to(torch.float32), f'{from_space_name}_{target_image_name}_to_{to_space_name}')
        # from atlas
        from_space_name = atlas_space_name
        from_space_dir = join(to_space_dir,f'{from_space_name}_to_{to_space_name}')
        os.makedirs(from_space_dir,exist_ok=exist_ok)
        # transforms, registered to atlas
        phii = v_to_phii(outputs['xv'],outputs['v'])
        Ai = torch.linalg.inv(outputs['A'])
        AiXJ = ((Ai[:3,:3]@XJshift.permute(1,2,3,0)[...,None])[...,0] + Ai[:3,-1]).permute(-1,0,1,2)
        phiiAiXJ = interp(outputs['xv'],phii-XV,AiXJ) + AiXJ
        transforms_dir = join(from_space_dir,TRANSFORMS)
        os.makedirs(transforms_dir,exist_ok=exist_ok)
        write_data(join(transforms_dir,f'{to_space_name}_to_{from_space_name}_displacement.vtk'),
                  xJshift,(phiiAiXJ)[None].to(torch.float32), f'{to_space_name}_to_{from_space_name}')
        # images
        AphiI = interp(xI,torch.tensor(I,device=phiiAiXJ.device,dtype=phiiAiXJ.dtype),phiiAiXJ,padding_mode='zeros')
        images_dir = join(from_space_dir,IMAGES)
        os.makedirs(images_dir,exist_ok=exist_ok)
        write_data(join(images_dir,f'{from_space_name}_{atlas_image_name}_to_{to_space_name}.vtk'),
                  xJshift,(AphiI).to(torch.float32), f'{from_space_name}_{atlas_image_name}_to_{to_space_name}')
        
        # a qc directory
        qc_dir = join(to_space_dir,'qc')
        os.makedirs(qc_dir,exist_ok=exist_ok)
        fig,ax = draw(RiJ,xJshift)
        fig.suptitle('RiJ')
        fig.savefig(join(qc_dir,
                                 f'input_{target_space_name}_{target_image_name}_to_registered_{target_space_name}.jpg'))
        fig.canvas.draw()
        fig,ax = draw(AphiI,xJshift)
        fig.suptitle('AphiI')
        fig.savefig(join(qc_dir,
                                 f'input_{atlas_space_name}_{atlas_image_name}_to_registered_{target_space_name}.jpg'))
        fig.canvas.draw()
        
        # to input space
        to_space_name = target_space_name
        to_space_dir = join(output_dir,to_space_name)
        os.makedirs(to_space_dir,exist_ok=exist_ok)
        # input to input
        # TODO: i can write the original images here
        
        # registered to input
        from_space_name = target_space_name + '-registered'
        from_space_dir = join(to_space_dir,f'{from_space_name}_to_{to_space_name}')
        os.makedirs(from_space_dir,exist_ok=exist_ok)
        # images, NONE        
        # transforms
        transforms_dir = join(from_space_dir,TRANSFORMS)
        os.makedirs(transforms_dir,exist_ok=exist_ok)
        Ri = torch.linalg.inv(outputs['A2d'])
        RiXJ = ((R[:,None,None,:2,:2]@(XJ[1:].permute(1,2,3,0)[...,None]))[...,0] + Ri[:,None,None,:2,-1]).permute(-1,0,1,2)
        RiXJ = torch.cat((XJ[0][None],RiXJ))
        write_data(join(transforms_dir,f'{to_space_name}_to_{from_space_name}_displacement.vtk'),
                  xJ,(RXJ-XJ)[None].to(torch.float32), f'{to_space_name}_to_{from_space_name}')
        
        
        # atlas to input
        from_space_name = atlas_space_name
        from_space_dir = join(to_space_dir,f'{from_space_name}_to_{to_space_name}')
        os.makedirs(from_space_dir,exist_ok=exist_ok)
        # transforms
        transforms_dir = join(from_space_dir,TRANSFORMS)
        os.makedirs(transforms_dir,exist_ok=exist_ok)
        AiRiXJ = ((Ai[:3,:3]@RiXJ.permute(1,2,3,0)[...,None])[...,0] + Ai[:3,-1]).permute(-1,0,1,2)
        phiiAiRiXJ = interp(outputs['xv'],phii-XV,AiRiXJ) + AiRiXJ
        write_data(join(transforms_dir,f'{to_space_name}_to_{from_space_name}_displacement.vtk'),
                  xJ,(phiiAiRiXJ-XJ)[None].to(torch.float32), f'{to_space_name}_to_{from_space_name}')
        # images
        images_dir = join(from_space_dir,IMAGES)
        os.makedirs(images_dir,exist_ok=exist_ok)
        RAphiI = interp(xI,torch.tensor(I,device=phiiAiRiXJ.device,dtype=phiiAiRiXJ.dtype),phiiAiRiXJ,padding_mode='zeros')
        write_data(join(images_dir,f'{from_space_name}_{atlas_image_name}_to_{to_space_name}.vtk'),
                  xJ,(RAphiI).to(torch.float32), f'{from_space_name}_{atlas_image_name}_to_{to_space_name}')
        
        # input space qc        
        qc_dir = join(to_space_dir,'qc')
        os.makedirs(qc_dir,exist_ok=exist_ok)
        
        fig,ax = draw(J,xJ)
        fig.suptitle('J')
        fig.savefig(join(qc_dir,f'{to_space_name}_{target_image_name}_to_{to_space_name}.jpg'))
        fig.canvas.draw()
        
        fig,ax = draw(RAphiI,xJ)
        fig.suptitle('RAphiI')
        fig.savefig(join(qc_dir,f'{from_space_name}_{atlas_image_name}_to_{to_space_name}.jpg'))
        fig.canvas.draw()
        # TODO: double check this is done
        
