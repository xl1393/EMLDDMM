'''
This file contains a point mapping function that CSH collaborators will use for a variety of purposes.

It will map sets of points between different spaces, using the outputs of the emlddmm library.

This code is intended to work with wrappers so we can have fewer input arguments.
'''



import emlddmm
import numpy as np
import matplotlib.pyplot as plt
from os import sep
from os.path import isdir,join
import json
from glob import glob
from warnings import warn
from difflib import SequenceMatcher


def point_mapper(data,
                 x,y,z=None,
                 from_space=None,to_space=None,
                 from_units='microns',to_units='microns',
                 from_ox=None,from_dx=None,
                 to_ox=None,to_dx=None):
    ''' A script to map points or a set of points using strings.
    
    
    Parameters
    ----------
    data : object
        Can be a string that points to a directory, in which case we assume this is pointing to our output directory
        Can be a string that points to a zip file, in which case we assume the file is a zipped version of the output
        Can be a dictionary, in which case we assume it is json with data.                
    x : numpy array
        Should be a number or a list of numbers showing the x coordinate we want to map.
    y : numpy array
        Should be a number or a list of numbers showing the y coordinate we want to map.
        If a list should be the same size as x.
    z : numpy array
        Should be a number or a list of numbers showing the z coordinate we want to map.
        If a list, should be the same size as x.  If not a list, it can be broadcasted to same size as x.
        It could also be a filename or a list of filenames.
    from_units, to_units : str
        Either 'microns' or 'pixels'.  NOTE: pixels not currently supported
        
    from_ox : tuple
        origin.  Location of first pixel in high res file, in units of microns.
        Should be a triple or a double.  They are in the order x,y or x,y,z
    from_dx : tuple
        pixel size.
    
    
    TODO
    ----
    Registration code ALL uses zyx convention (slice, row, column).
    When we write to disk, we use vtk convention, which is xyz.
    Make sure this is all consistent.
    Maybe we can ask the user to input either slice,row,col, or x,y,z, we'll flip them accordingly.
    Make them keywords arguments so we can do them in any order.  We can do this in matlab too.
    
    '''
    
    # parse the directory
    data_is_directory = False
    data_is_zip = False
    data_is_json = False    
    if isinstance(data,str) and isdir(data):
        data_is_directory = True
        data_directory = data
        print('registration data is a directory')
    elif isinstance(data,str) and data.endswith('.zip'):
        print('registration data is a zip file of an output directory')
        raise Exception(f'Only data directories are supported for now')
    elif data_is_json:
        raise Exception('TODO: load the json file into a json')
        data_is_json = True
        data_is_json_file = False
        data_json = dict()        
    else:
        raise Exception(f'Only output directory is currently supported for registration data, but you input {data}')
    
        
    
    # parse the data points xy
    print('printing x')
    if not isinstance(x,np.ndarray):
        x = np.array(x)
        if x.ndim == 0: x = x[None]
    print(x)
    print('printing y')
    if not isinstance(y,np.ndarray):
        y = np.array(y)
        if y.ndim == 0: y = y[None]
    print(y)
    if not np.all(np.array(x.shape)==np.array(y.shape)):
        raise Exception(f'x and y must be the same size, but are {x.shape} and {y.shape}')
    
    
    
    print('printing z')
    z_is_file = False
    z_is_coordinate = False
    if z is not None:   
        if isinstance(z,str):
            z_is_file = True            
            # assume that z is a string pointing to a filename
            # this will only be necessary when mapping from a 2D space.
            print(f'found a filename for z {z}')
            
        elif not isinstance(z,np.ndarray):
            z_is_coordinate = True
            z = np.array(z)
        print(z)
    else:
        warn(f'z is None, setting to zeros coordinate')
        z = np.zeros_like(x)
        print(z)
    
    # parse the from space
    if from_space is None:
        raise Exception('You must specify a from space')
    print(f'From space is {from_space}')        
    # parse the to space
    if to_space is None:
        raise Exception('You must specify a to space')
    print(f'To space is {to_space}')
    # check if spaces are valid
    if from_space == to_space:
        warn(f'from space equals to space, returning identity')
        return x,y,z
    if data_is_directory:
        files = glob(join(data_directory,'*'))
        space_names = []
        for file in files:
            if isdir(file):                
                space_names.append(file.split(sep)[-1])                                
    elif data_is_json:        
        space_names = []
        raise Exception('json data not implemented yet')    
    print(f'identified spaces {space_names}')
    # TODO, allow for sloppy matching
    if from_space not in space_names:
        raise Exception(f'Your from_space {from_space} was not in the availabile spaces {space_names}')
    if to_space not in space_names:
        raise Exception(f'Your to_space {to_space} was not in the available spaces {space_names}')
    
    
    # if necessary we'll convert
    if from_units == 'pixels':
        # TODO
        warn(f'We need to specify spacing and origin in the from space')        
        # for defaults, we will use 0.46 microns
        # and for origin we will have to go to an image
        if from_ox is None or from_dx is None:
            warn(f'Did not specify from ox or from dx so using defaults (dx=0.46um, origin in center of image)')
            dx = [0.46,0.46]
            # to find the origin, we need to find an image to estimate the size
            raise Exception('This default is not done yet, for now you have to specify the origin')            
            
        #raise Exception('conversion from pixels not yet implemented')
        print('converting from pixels to microns in input space')
        print('before')
        print(x,y,z)
        x = x * from_dx[0] + from_ox[0]
        y = y * from_dx[1] + from_ox[1]
        if len(from_dx) > 2 and len(from_ox) > 2:
            z = z * from_dx[2] + from_ox[2]
        print('after')
        print(x,y,z)
    elif from_units == 'microns':
        pass
    else:
        raise Exception(f'Units must be either pixels or microns, but you specified {units}')
        
        
    # find the appropriate transform
    # we look in the folder from_space/to_space_to_from_space/transforms
    if data_is_directory:
        transform_dir = join(data_directory,from_space,to_space + '_to_' + from_space, 'transforms')
        print(f'Found transform dir {transform_dir}')
        files = glob(join(transform_dir,'*matrix.txt'))
        files.sort()
        if files:
            print(f'This is a 2D transform')
            #print(files)
            # to load the matrix we have to get the right z coordinate or filename
            if z_is_file:
                zbool = [z in f for f in files]
                zind = zbool.index(True)
                print(f'Matched to the transformation file {files[zind]}')
                mat = emlddmm.read_matrix_data(files[zind])
                print(mat)
                # transform the coordinates                
                transformed_coords = (mat[:2,:2]@(np.stack([y,x],-1)[...,None]))[...,0] + mat[:2,-1]
                # these coordinates are in microns
                xtransformed = transformed_coords[...,1]
                ytransformed = transformed_coords[...,0]
                # what if we want to convert to pixels
                if to_units == 'pixels':
                    if to_ox is None or to_dx is None:
                        raise Exception('if output space is "pixel" then you must specify both to_ox and to_dx')
                    # inverse of previous function
                    xtransformed = (xtransformed - to_ox[0])/to_dx[0]
                    ytransformed = (ytransformed - to_ox[1])/to_dx[1]
                    
                return xtransformed, ytransformed
            elif z_is_coordinate:
                raise Exception(f'TODO: we have to look up all the files and match one to this coordinate')
                pass                        
            else:
                raise Exception('Something is wrong, z should have been either a file or a coordinate')
        else:
            
            print(f'This is a 3D transform')
            print(f'We load  a vtk file and apply it')
            files = glob(join(transform_dir,'*displacement.vtk'))
            files.sort()            
            print(files)            
            if z_is_file:
                zbool = [z in f for f in files]
                zind = zbool.index(True)
                print(f'Matched to the transformation file {files[zind]}')
                xtform,tform,_,_ = emlddmm.read_data(files[zind])                                
                tform = tform.squeeze()                
                print(tform.shape)
                zcoord = xtform[0]                
                zcoord = np.ones_like(x)*zcoord
                print(zcoord)
                # todo, implement this here instead of import
                from scipy.interpolate import interpn
                zyx = np.stack((zcoord,y,x),-1)
                #print(zyx)
                yx = np.stack((y,x),-1)
                #print(tform.shape)                
                transformed_coords = interpn(xtform[1:],tform.transpose(1,2,0),yx,bounds_error=False,fill_value=0) + zyx
                return transformed_coords
            
            elif z_is_coordinate:
                pass
            else:
                raise Exception('Something is wrong, z should have been either a file or a coordinate')
    else:
        raise Exception('currently only works when data is a directory')