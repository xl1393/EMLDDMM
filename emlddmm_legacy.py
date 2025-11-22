
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn.functional import grid_sample
import glob
import os
from os.path import join,split,splitext
from os import makedirs
import nibabel
import json
import re
import argparse
from warnings import warn
#from skimage import measure, filters
#from mayavi import mlab
# import nrrd
# from scipy.spatial.distance import directed_hausdorff, dice
# from medpy.metric import binary
import tifffile as tf # for 16 bit tiff
from scipy.stats import mode
from scipy.interpolate import interpn
import PIL # only required for one format conversion function
try:    
    PIL.Image.MAX_IMAGE_PIXELS = None # prevent decompression bomb error for large files
except:
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None # prevent decompression bomb error for large files
    
# for interactive display with widget
from IPython.display import display
    
# display
def extent_from_x(xJ):
    ''' Given a set of pixel locations, returns an extent 4-tuple for use with imshow.
    
    Note
    ----
    Note inputs are locations of pixels along each axis, i.e. row column not xy.
    
    Parameters
    ----------
    xJ : list of torch tensors
        Location of pixels along each axis
    
    Returns
    -------
    extent : tuple
        (xmin, xmax, ymin, ymax) tuple
    
    Example
    -------
    Draw a 2D image stored in J, with pixel locations of rows stored in xJ[0] and pixel locations
    of columns stored in xJ[1].
    
    >>> import matplotlib.pyplot as plt
    >>> extent_from_x(xJ)
    >>> fig,ax = plt.subplots()
    >>> ax.imshow(J,extent=extentJ)
    
    '''
    dJ = [x[1]-x[0] for x in xJ]
    extentJ = ( (xJ[1][0] - dJ[1]/2.0).item(),
               (xJ[1][-1] + dJ[1]/2.0).item(),
               (xJ[0][-1] + dJ[0]/2.0).item(),
               (xJ[0][0] - dJ[0]/2.0).item())
    return extentJ


def labels_to_rgb(S,seed=0,black_label=0,white_label=255):
    ''' Convert an integer valued label image into a randomly colored image 
    for visualization with the draw function
    
    Parameters
    ----------
    S : numpy array
        An array storing integer labels.  Expected to be 4D (1 x slices x rows x columns), 
        but can be 3D (slices x rows x columns).
    seed : int
        Random seed for reproducibility
    black_label : int
        Color to assign black.  Usually for background.
    
    '''
    if isinstance(S,torch.Tensor):        
        Scopy = S.clone().detach().cpu().numpy()
    else:
        Scopy = S
    np.random.seed(seed)
    labels,ind = np.unique(Scopy,return_inverse=True)
    colors = np.random.rand(len(labels),3)
    colors[labels==black_label] = 0.0
    colors[labels==white_label] = 1.0
    
    SRGB = colors[ind].T # move colors to first axis
    SRGB = SRGB.reshape((3,S.shape[1],S.shape[2],S.shape[3]))
    
    return SRGB
    
    

def draw(J,xJ=None,fig=None,n_slices=5,vmin=None,vmax=None,disp=True,cbar=False,slices_start_end=[None,None,None],**kwargs):    
    """ Draw 3D imaging data.
    
    Images are shown by sampling slices along 3 orthogonal axes.
    Color or grayscale data can be shown.
    
    Parameters
    ----------
    J : array like (torch tensor or numpy array)
        A 3D image with C channels should be size (C x nslice x nrow x ncol)
        Note grayscale images should have C=1, but still be a 4D array.
    xJ : list
        A list of 3 numpy arrays.  xJ[i] contains the positions of voxels
        along axis i.  Note these are assumed to be uniformly spaced. The default
        is voxels of size 1.0.
    fig : matplotlib figure
        A figure in which to draw pictures. Contents of the figure will be cleared.
        Default is None, which creates a new figure.
    n_slices : int
        An integer denoting how many slices to draw along each axis. Default 5.
    vmin
        A minimum value for windowing imaging data. Can also be a list of size C for
        windowing each channel separately. Defaults to None, which corresponds 
        to tha 0.001 quantile on each channel.
    vmax
        A maximum value for windowing imaging data. Can also be a list of size C for
        windowing each channel separately. Defaults to None, which corresponds 
        to tha 0.999 quantile on each channel.
    disp : bool
        Figure display toggle
    kwargs : dict
        Other keywords will be passed on to the matplotlib imshow function. For example
        include cmap='gray' for a gray colormap

    Returns
    -------
    fig : matplotlib figure
        The matplotlib figure variable with data.
    axs : array of matplotlib axes
        An array of matplotlib subplot axes containing each image.


    Example
    -------
    Here is an example::

       >>> example test
   
    TODO
    ----
    Put interpolation='none' in keywords


    """
    if type(J) == torch.Tensor:
        J = J.detach().clone().cpu()
    J = np.array(J)
    if xJ is None:
        nJ = J.shape[-3:]
        xJ = [np.arange(n) - (n-1)/2.0 for n in nJ] 
    if type(xJ[0]) == torch.Tensor:
        xJ = [np.array(x.detach().clone().cpu()) for x in xJ]
    xJ = [np.array(x) for x in xJ]
    
    if fig is None:
        fig = plt.figure()
    fig.clf()    
    if vmin is None:
        vmin = np.quantile(J,0.001,axis=(-1,-2,-3))
    if vmax is None:
        vmax = np.quantile(J,0.999,axis=(-1,-2,-3))
    vmin = np.array(vmin)
    vmax = np.array(vmax)    
    # I will normalize data with vmin, and display in 0,1
    if vmin.ndim == 0:
        vmin = np.repeat(vmin,J.shape[0])
    if vmax.ndim == 0:
        vmax = np.repeat(vmax,J.shape[0])
    if len(vmax) >= 2 and len(vmin) >= 2:
        # for rgb I'll scale it, otherwise I won't, so I can use colorbars
        J -= vmin[:,None,None,None]
        J /= (vmax[:,None,None,None] - vmin[:,None,None,None])
        J[J<0] = 0
        J[J>1] = 1
        vmin = 0.0
        vmax = 1.0
    # I will only show the first 3 channels
    if J.shape[0]>3:
        J = J[:3]
    if J.shape[0]==2:
        J = np.stack((J[0],J[1],J[0]))
    
    
    axs = []
    axsi = []
    # ax0
    slices = np.round(np.linspace(0,J.shape[1]-1,n_slices+2)[1:-1]).astype(int)     
    if slices_start_end[0] is not None:
        slices = np.round(np.linspace(slices_start_end[0][0],slices_start_end[0][1],n_slices+2)[1:-1]).astype(int)     
        
    # for origin upper (default), extent is x (small to big), then y reversed (big to small)
    extent = (xJ[2][0],xJ[2][-1],xJ[1][-1],xJ[1][0])
    for i in range(n_slices):
        ax = fig.add_subplot(3,n_slices,i+1)
        toshow = J[:,slices[i]].transpose(1,2,0)
        if toshow.shape[-1] == 1:
            toshow = toshow.squeeze(-1)
        ax.imshow(toshow,vmin=vmin,vmax=vmax,aspect='equal',extent=extent,**kwargs)
        if i>0: ax.set_yticks([])
        axsi.append(ax)
    axs.append(axsi)
    axsi = []
    # ax1
    slices = np.round(np.linspace(0,J.shape[2]-1,n_slices+2)[1:-1]).astype(int)    
    if slices_start_end[1] is not None:
        slices = np.round(np.linspace(slices_start_end[1][0],slices_start_end[1][1],n_slices+2)[1:-1]).astype(int)         
    extent = (xJ[2][0],xJ[2][-1],xJ[0][-1],xJ[0][0])
    for i in range(n_slices):
        ax = fig.add_subplot(3,n_slices,i+1+n_slices)      
        toshow = J[:,:,slices[i]].transpose(1,2,0)
        if toshow.shape[-1] == 1:
            toshow = toshow.squeeze(-1)
        ax.imshow(toshow,vmin=vmin,vmax=vmax,aspect='equal',extent=extent,**kwargs)
        if i>0: ax.set_yticks([])
        axsi.append(ax)
    axs.append(axsi)
    axsi = []
    # ax2
    slices = np.round(np.linspace(0,J.shape[3]-1,n_slices+2)[1:-1]).astype(int)        
    if slices_start_end[2] is not None:
        slices = np.round(np.linspace(slices_start_end[2][0],slices_start_end[2][1],n_slices+2)[1:-1]).astype(int)     
    
    extent = (xJ[1][0],xJ[1][-1],xJ[0][-1],xJ[0][0])
    for i in range(n_slices):        
        ax = fig.add_subplot(3,n_slices,i+1+n_slices*2)
        toshow = J[:,:,:,slices[i]].transpose(1,2,0)
        if toshow.shape[-1] == 1:
            toshow = toshow.squeeze(-1)
        ax.imshow(toshow,vmin=vmin,vmax=vmax,aspect='equal',extent=extent,**kwargs)
        if i>0: ax.set_yticks([])
        axsi.append(ax)
    axs.append(axsi)
    
    fig.subplots_adjust(wspace=0,hspace=0)
    if not disp:
        plt.close(fig)
    axs = np.array(axs)
    
    if cbar and disp:
        plt.colorbar(mappable=[h for h in axs[0][0].get_children() if 'Image' in str(h)][0],ax=np.array(axs).ravel())
    return fig,axs
    
    
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
def sinc_resample_numpy(I,n):
    ''' Perform sinc resampling of an image in numpy.
    
    This function does sinc resampling using numpy rfft
    torch does not let us control behavior of fft well enough
    This is intended to be used to resample velocity fields if necessary
    Only intending it to be used for upsampling.
    
    Parameters
    ----------
    I : numpy array
        An image to be resampled. Can be an arbitrary number of dimensions.
    n : list of ints
        Desired dimension of output data.
    
    
    Returns
    -------
    Id : numpy array
        A resampled image of size n.
    '''
    Id = np.array(I)
    for i in range(len(n)):        
        if I.shape[i] == n[i]:
            continue
        Id = np.fft.irfft(np.fft.rfft(Id,axis=i),axis=i,n=n[i])
    # output with correct normalization
    Id = Id*np.prod(Id.shape)/np.prod(I.shape) 
    return Id
    

def downsample_ax(I,down,ax,W=None):
    '''
    Downsample imaging data along one of the first 5 axes.
    
    Imaging data is downsampled by averaging nearest pixels.
    Note that data will be lost from the end of images instead of padding.
    This function is generally called repeatedly on each axis.
    
    Parameters
    ----------
    I : array like (numpy or torch)
        Image to be downsampled on one axis.
    down : int
        Downsampling factor.  2 means average pairs of nearest pixels 
        into one new downsampled pixel
    ax : int
        Which axis to downsample along.
    W : np array
        A mask the same size as I, but without a "channel" dimension
    
    Returns
    -------
    Id : array like
        The downsampled image.
    
    Raises
    ------
    Exception
        If a mask (W) is included and ax == 0. 
    '''
    nd = list(I.shape)        
    nd[ax] = nd[ax]//down
    if type(I) == torch.Tensor:
        Id = torch.zeros(nd,device=I.device,dtype=I.dtype)
    else:
        Id = np.zeros(nd,dtype=I.dtype)
    if W is not None:
        if type(W) == torch.Tensor:
            Wd = torch.zeros(nd[1:],device=W.device,dtype=W.dtype)
        else:
            Wd = np.zeros(nd[1:],dtype=W.dtype)            
    if W is None:
        for d in range(down):
            if ax==0:        
                Id += I[d:down*nd[ax]:down]
            elif ax==1:        
                Id += I[:,d:down*nd[ax]:down]
            elif ax==2:
                Id += I[:,:,d:down*nd[ax]:down]
            elif ax==3:
                Id += I[:,:,:,d:down*nd[ax]:down]
            elif ax==4:
                Id += I[:,:,:,:,d:down*nd[ax]:down]
            elif ax==5:
                Id += I[:,:,:,:,:,d:down*nd[ax]:down]
            # ... should be enough but there really has to be a better way to do this        
            # note I could use "take"
        Id = Id/down
        return Id
    else:
        # if W is not none
        for d in range(down):
            if ax==0:        
                Id += I[d:down*nd[ax]:down]*W[d:down*nd[ax]:down]
                raise Exception('W not supported with ax=0')
                
            elif ax==1:        
                Id += I[:,d:down*nd[ax]:down]*W[d:down*nd[ax]:down]
                Wd += W[d:down*nd[ax]:down]
            elif ax==2:
                Id += I[:,:,d:down*nd[ax]:down]*W[:,d:down*nd[ax]:down]
                Wd += W[:,d:down*nd[ax]:down]
            elif ax==3:
                Id += I[:,:,:,d:down*nd[ax]:down]*W[:,:,d:down*nd[ax]:down]
                Wd += W[:,:,d:down*nd[ax]:down]
            elif ax==4:
                Id += I[:,:,:,:,d:down*nd[ax]:down]*W[:,:,:,d:down*nd[ax]:down]
                Wd += W[:,:,:,d:down*nd[ax]:down]
            elif ax==5:
                Id += I[:,:,:,:,:,d:down*nd[ax]:down]*W[:,:,:,:,d:down*nd[ax]:down]
                Wd += W[:,:,:,:,d:down*nd[ax]:down]
        Id = Id / (Wd + Wd.max()*1e-6)
        
        
        Wd = Wd / down
        return Id,Wd
        
def downsample(I,down,W=None):
    '''
    Downsample an image by an integer factor along each axis. Note extra data at 
    the end will be truncated if necessary.
    
    If the first axis is for image channels, downsampling factor should be 1 on this.
    
    Parameters
    ----------
    I : array (numpy or torch)
        Imaging data to downsample
    down : list of int
        List of downsampling factors for each axis.
    W : array (numpy or torch)
        A weight of the same size as I but without the "channel" dimension
    
    Returns
    -------
    Id : array (numpy or torch as input)
        Downsampled imaging data.
    '''
    down = list(down)
    while len(down) < len(I.shape):
        down.insert(0,1)    
    if type(I) == torch.Tensor:
        Id = torch.clone(I)
    else:
        Id = np.copy(I)
    if W is not None:
        if type(W) == torch.Tensor:
            Wd = torch.clone(W)
        else:
            Wd = np.copy(W)
    for i,d in enumerate(down):
        if d==1:
            continue
        if W is None:
            Id = downsample_ax(Id,d,i)
        else:
            Id,Wd = downsample_ax(Id,d,i,W=Wd)
    if W is None:
        return Id
    else:
        return Id,Wd


def downsample_image_domain(xI,I,down,W=None): 
    '''
    Downsample an image as well as pixel locations
    
    Parameters
    ----------
    xI : list of numpy arrays
        xI[i] is a numpy array storing the locations of each voxel
        along the i-th axis.
    I : array like
        Image to be downsampled
    down : list of ints
        Factor by which to downsample along each dimension
    W : array like
        Weights the same size as I, but without a "channel" dimension
        
    Returns
    -------
    xId : list of numpy arrays
        New voxel locations in the same format as xI
    Id : numpy array
        Downsampled image.
    
    Raises
    ------
    Exception
        If the length of down and xI are not equal.
    '''
    if len(xI) != len(down):
        raise Exception('Length of down and xI must be equal')
    if W is None:
        Id = downsample(I,down)    
    else:
        Id,Wd = downsample(I,down,W=W)
    xId = []
    for i,d in enumerate(down):
        xId.append(downsample_ax(xI[i],d,0))
    if W is None:
        return xId,Id
    else:
        return xId,Id,Wd
    
    
def downmode(xI,S_,down):
    ''' Downsamples a 3D image by taking the mode among rectangular neighborhoods.
    This is appropriate for label images, where averaging pixel values is not meaningful.
    
    Note
    ----
    2D images can be hanled by adding a singleton dimension
    no leading batch dimensions
    
    Parameters
    ----------
    xI : list of 3 numpy arrays
        Locations of image pixels along each axis
    S_ : numpy array
        Numpy array storing imaging data.  Note there should not be
        a leading dimension for channels.
    down : list of 3 ints
        downsample by this factor along each axis.
                
    
    Returns
    -------
    xd : list of 3 numpy arrays
        Locations of image pixels along each axis after downsampling.
    Sd : numpy array
        The downsampled image.
    
    
    '''
    
    # crop it off the right side so its size is a multiple of down
    nS = np.array(S_.shape)
    nSd = nS//down
    nSu = nSd*down
    S_ = np.copy(S_)[:nSu[0],:nSu[1],:nSu[2]]
    # now reshape
    S_ = np.reshape(S_,(nSd[0],down[0],nSd[1],down[1],nSd[2],down[2]))
    S_ = S_.transpose(1,3,5,0,2,4)
    S_ = S_.reshape(down[0]*down[1]*down[2],nSd[0],nSd[1],nSd[2])

    S_ = mode(S_,axis=0)[0][0]
    # now same for xR
    xI_ = [np.copy(x) for x in xI]
    xI_[0] = xI_[0][:nSu[0]]
    xI_[0] = np.mean(xI_[0].reshape(nSd[0],down[0]),1)

    xI_[1] = xI_[1][:nSu[1]]
    xI_[1] = np.mean(xI_[1].reshape(nSd[1],down[1]),1)

    xI_[2] = xI_[2][:nSu[2]]
    xI_[2] = np.mean(xI_[2].reshape(nSd[2],down[2]),1)
    
    return xI_,S_

def downmedian(xI,S_,down):
    ''' Downsamples a 3D image by taking the median among rectangular neighborhoods.
    This is often appropriate when image pixels has a small number of outliers, or when
    pixel values are assumed to be ordered, but don't otherwise belong to a vector space.
    
    Note
    ----
    2D images can be hanled by adding a singleton dimension
    no leading batch dimensions
    
    Parameters
    ----------
    xI : list of 3 numpy arrays
        Locations of image pixels along each axis
    S_ : numpy array
        Numpy array storing imaging data.  Note there should not be
        a leading dimension for channels.
    down : list of 3 ints
        downsample by this factor along each axis.
                
    
    Returns
    -------
    xd : list of 3 numpy arrays
        Locations of image pixels along each axis after downsampling.
    Sd : numpy array
        The downsampled image.
    
    
    '''
    
    # crop it off the right side so its size is a multiple of down
    nS = np.array(S_.shape)
    nSd = nS//down
    nSu = nSd*down
    S_ = np.copy(S_)[:nSu[0],:nSu[1],:nSu[2]]
    # now reshape
    S_ = np.reshape(S_,(nSd[0],down[0],nSd[1],down[1],nSd[2],down[2]))
    S_ = S_.transpose(1,3,5,0,2,4)
    S_ = S_.reshape(down[0]*down[1]*down[2],nSd[0],nSd[1],nSd[2])

    S_ = np.median(S_,axis=0)[0][0]
    # now same for xR
    xI_ = [np.copy(x) for x in xI]
    xI_[0] = xI_[0][:nSu[0]]
    xI_[0] = np.mean(xI_[0].reshape(nSd[0],down[0]),1)

    xI_[1] = xI_[1][:nSu[1]]
    xI_[1] = np.mean(xI_[1].reshape(nSd[1],down[1]),1)

    xI_[2] = xI_[2][:nSu[2]]
    xI_[2] = np.mean(xI_[2].reshape(nSd[2],down[2]),1)
    
    return xI_,S_

# build an interp function from grid sample
def interp(x, I, phii, interp2d=False, **kwargs):
    '''
    Interpolate an image with specified regular voxel locations at specified sample points.
    
    Interpolate the image I, with regular grid positions stored in x (1d arrays),
    at the positions stored in phii (3D or 4D arrays with first channel storing component)
    
    Parameters
    ----------
    x : list of numpy arrays
        x[i] is a numpy array storing the pixel locations of imaging data along the i-th axis.
        Note that this MUST be regularly spaced, only the first and last values are queried.
    I : array
        Numpy array or torch tensor storing 2D or 3D imaging data.  In the 3D case, I is a 4D array with 
        channels along the first axis and spatial dimensions along the last 3. For 2D, I is a 3D array with
        spatial dimensions along the last 2.
    phii : array
        Numpy array or torch tensor storing positions of the sample points. phii is a 3D or 4D array
        with components along the first axis (e.g. x0,x1,x1) and spatial dimensions 
        along the last axes.
    interp2d : bool, optional
        If True, interpolates a 2D image, otherwise 3D. Default is False (expects a 3D image).
    kwargs : dict
        keword arguments to be passed to the grid sample function. For example
        to specify interpolation type like nearest.  See pytorch grid_sample documentation.
    
    Returns
    -------
    out : torch tensor
        Array storing an image with channels stored along the first axis. 
        This is the input image resampled at the points stored in phii.


    '''
    # first we have to normalize phii to the range -1,1    
    I = torch.as_tensor(I)
    phii = torch.as_tensor(phii)
    phii = torch.clone(phii)
    ndim = 2 if interp2d==True else 3
    for i in range(ndim):
        phii[i] -= x[i][0]
        phii[i] /= x[i][-1] - x[i][0]
    # note the above maps to 0,1
    phii *= 2.0
    # to 0 2
    phii -= 1.0
    # done

    # NOTE I should check that I can reproduce identity
    # note that phii must now store x,y,z along last axis
    # is this the right order?
    # I need to put batch (none) along first axis
    # what order do the other 3 need to be in?    
    # feb 2022
    if 'padding_mode' not in kwargs:
        kwargs['padding_mode'] = 'border' # note that default is zero, but we switchthe default to border
    if interp2d==True:
        phii = phii.flip(0).permute((1,2,0))[None]
    else:
        phii = phii.flip(0).permute((1,2,3,0))[None]
    out = grid_sample(I[None], phii, align_corners=True, **kwargs)

    # note align corners true means square voxels with points at their centers
    # post processing, get rid of batch dimension
    out = out[0]
    return out
    
# now we need to create a flow
# timesteps will be along the first axis
def v_to_phii(xv,v,**kwargs):
    '''
    Use Euler's method to construct a position field from a velocity field
    by integrating over time.
    
    This method uses interpolation and subtracts and adds identity for better
    behavior outside boundaries. This method is sometimes refered to as the
    method of characteristics.
    
    Parameters
    ----------
    xv : list of 1D tensors
        xv[i] is a tensor storing the location of the sample points along
        the i-th dimension of v
    v : 5D tensor
        5D tensor where first axis corresponds to time, second corresponds to 
        component, and 3rd to 5th correspond to spatial dimensions.
    
    Returns
    -------    
    phii : 4D tensor
        Inverse transformation is output with component on the first dimension
        and space on the last 3. Note that the whole timeseries is not output.
    
    '''
    XV = torch.stack(torch.meshgrid(xv,indexing='ij'))
    phii = torch.clone(XV)
    dt = 1.0/v.shape[0]
    for t in range(v.shape[0]):
        Xs = XV - v[t]*dt
        phii = interp(xv,phii-XV,Xs,**kwargs)+Xs
    return phii
        
    
def reshape_for_local(J,local_contrast):
    '''
    Reshapes an image into blocks for simple local contrast estimation.
    
    Parameters
    ----------
    J : tensor
        3D image data where first index stores the channel information (i.e. 4D array)
    local_contrast : tensor
        1D tensor storing the block size on each dimension
    
    Returns
    -------
    Jv : tensor
        Reshaped imaging data to be used for contrast estimation
    
    '''
    # get shapes and pad
    Jshape = torch.as_tensor(J.shape[1:],device=J.device)
    topad = Jshape%local_contrast
    topad = (local_contrast-topad)%local_contrast    
    Jpad = torch.nn.functional.pad(J,(0,topad[2].item(),0,topad[1].item(),0,topad[0].item()))   
    '''
    # let's do symmetric padding instead
    # note, we need cropping at the end to match this
    # if its even, they'll both be the same
    # if its odd, right will be one more
    leftpad = torch.floor(topad/2.0).int()
    rightpad = torch.ceil(topad/2.0).int()
    Jpad = torch.nn.functional.pad(J,(leftpad[2].item(),rightpad[2].item(),leftpad[1].item(),rightpad[1].item(),leftpad[0].item(),rightpad[0].item()))   
    '''
    
    # now reshape it
    Jpad_ = Jpad.reshape( (Jpad.shape[0],
                           Jpad.shape[1]//local_contrast[0].item(),local_contrast[0].item(),
                           Jpad.shape[2]//local_contrast[1].item(),local_contrast[1].item(),
                           Jpad.shape[3]//local_contrast[2].item(),local_contrast[2].item()))
    Jpad__ = Jpad_.permute(1,3,5,2,4,6,0)
    Jpadv = Jpad__.reshape(Jpad__.shape[0],Jpad__.shape[1],Jpad__.shape[2],
                           torch.prod(local_contrast).item(),Jpad__.shape[-1])

    return Jpadv
                           
def reshape_from_local(Jv,local_contrast=None):
    '''
    After changing contrast, transform back
    TODO: this did not get used
    '''
    pass
    
    
def project_affine_to_rigid(A,XJ):
    ''' This function finds the closest rigid transform to the given affine transform.
    
    Close is defined in terms of the action of A^{-1} on XJ
    
    That is, we find R to minimize || A^{-1}XJ - R^{-1}XJ||^2_F = || R (A^{-1}XJ) - XJ ||^2_F.
    
    We use a standard procurstes method.
    
    Parameters
    ----------
    A : torch tensor
        A 4x4 affine transformation matrix
    XJ : torch tensor
        A 3 x slice x row x col array of coordinates in the space of the target image
    
    Returns
    -------
    R : torch tensor
        A 4x43 rigid affine transformation matrix.
    
    '''
    Ai = torch.linalg.inv(A)
    AiXJ = ( (Ai[:3,:3]@XJ.permute(1,2,3,0)[...,None])[...,0] + Ai[:3,-1] ).permute(-1,0,1,2)
    YJ = AiXJ
    YJbar = torch.mean(YJ,(1,2,3),keepdims=True)
    XJbar = torch.mean(XJ,(1,2,3),keepdims=True)
    YJ0 = YJ - YJbar
    XJ0 = XJ - XJbar
    Sigma = YJ0.reshape(3,-1)@XJ0.reshape(3,-1).T 
    u,s,vh = torch.linalg.svd(Sigma)    
    R = (u@vh).T
    T = XJbar.squeeze() - R@YJbar.squeeze()
    A.data[:3,:3] = R
    A.data[:3,-1] = T
    return A



def emlddmm(**kwargs):
    '''
    Run the EMLDDMM algorithm for deformable registration between two
    different imaging modalities with possible missing data in one of them
    
    Details of this algorithm can be found in
    
    * [1] Tward, Daniel, et al. "Diffeomorphic registration with intensity transformation and missing data: Application to 3D digital pathology of Alzheimer's disease." Frontiers in neuroscience 14 (2020): 52.
    * [2] Tward, Daniel, et al. "3d mapping of serial histology sections with anomalies using a novel robust deformable registration algorithm." Multimodal Brain Image Analysis and Mathematical Foundations of Computational Anatomy. Springer, Cham, 2019. 162-173.
    * [3] Tward, Daniel, et al. "Solving the where problem in neuroanatomy: a generative framework with learned mappings to register multimodal, incomplete data into a reference brain." bioRxiv (2020).
    * [4] Tward DJ. An optical flow based left-invariant metric for natural gradient descent in affine image registration. Frontiers in Applied Mathematics and Statistics. 2021 Aug 24;7:718607.
    
    
    Note all parameters are keyword arguments, but the first four are required.    
    
    
    Parameters
    ----------
    
    xI : list of arrays
        xI[i] stores the location of voxels on the i-th axis of the atlas image I (REQUIRED)
    I : 4D array (numpy or torch)
        4D array storing atlas imaging data.  Channels (e.g. RGB are stored on the 
        first axis, and the last three are spatial dimensions. (REQUIRED)
    xJ : list of arrays
        xJ[i] stores the location of voxels on the i-th axis of the target image J (REQUIRED)
    J : 4D array (numpy or torch)
        4D array storing target imaging data.  Channels (e.g. RGB are stored on the 
        first axis, and the last three are spatial dimensions. (REQUIRED)
    nt : int
        Number of timesteps for integrating a velocity field to yeild a position field (default 5).
    eA : float
        Gradient descent step size for affine component (default 1e-5).  It is strongly suggested
        that you test this value and not rely on defaults. Note linear and translation components
        are combined following [4] so only one stepsize is required.
    ev : float
        Gradient descent step size for affine component (default 1e-5).  It is strongly suggested
        that you test this value and not rely on defaults.
    order : int
        Order of the polynomial used for contrast mapping. If using local contranst,
        only order 1 is supported.
    n_draw : int
        Draw a picture every n_draw iterations. 0 for do not draw.
    sigmaR : float
        Amount of regularization of the velocity field used for diffeomorphic transformation,
        of the form 1/sigmaR^2 * (integral over time of norm velocity squared ).
    n_iter : int
        How many iterations of optimization to run.
    n_e_step : int
        How many iterations of M step to run before another E step is ran in
        expectation maximization algorithm for detecting outliers.
    v_start : int
        What iteration to start optimizing velocity field.  One may want to compute an affine
        transformation before beginning to compute a deformation (for example).
    n_reduce_step : int
        Simple stepsize reducer for gradient descent optimization. Every this number of steps,
        we check if objective function is oscillating. If so we reduce the step size.
    v_expand_factor : float
        How much bigger than the atlas image should the domain of the velocity field be? This
        is helpful to avoid wraparound effects caused by discrete Fourier domain calculations.
        0.2 means 20% larger.
    v_res_factor : float
        How much lower resolution should the velocity field be sampled at than the atlas image.
        This is overrided if you specify dv.
    dv : None or float or list of 3 floats
        Explicitly state the resolution of the sampling grid for the velocity field.
    a : float
        Constant with units of length.  In velocity regularization, its square is multiplied against the Laplacian.
        Regularization is of the form 1/2/sigmaR^2 int |(id - a^2 Delta)^p v_t|^2_{L2} dt.
    p : float
        Power of the Laplacian operator in regularization of the velocity field.  
        Regularization is of the form 1/2/sigmaR^2 int |(id - a^2 Delta)^p v_t|^2_{L2} dt.
        
    
    
        
    
        
    
    
    
    Returns
    -------
    out : dict
        Returns a dictionary of outputs storing computing transforms. if full_outputs==True, 
        then more data is output including figures.
    
    Raises
    ------
    Exception
        If the initial velocity does not have three components.
    Exception
        Local contrast transform requires either order = 1, or order > 1 and 1D atlas.
    Exception
        If order > 1. Local contrast transform not implemented yet except for linear.
    Exception
        Amode must be 0 (normal), 1 (rigid), or 2 (rigid+scale), or 3 (rigid using XJ for projection).
    
    
    '''
    # required arguments are
    # I - atlas image, size C x slice x row x column
    # xI - list of pixels locations in I, corresponding to each axis other than channels
    # J - target image, size C x slice x row x column
    # xJ - list of pixel locations in J
    # other parameters are specified in a dictionary with defaults listed below
    # if you provide an input for PARAMETER it will be used as an initial guess, 
    # unless you specify update_PARAMETER=False, in which case it will be fixed
    
    
    
    # I should move them to torch and put them on the right device
    I = kwargs['I']
    J = kwargs['J']
    xI = kwargs['xI']
    xJ = kwargs['xJ']
    
    ##########################################################################################################
    # everything else is optional, defaults are below
    defaults = {'nt':5,
                'eA':1e5,
                'ev':2e3,
                'order':1, # order of polynomial
                'n_draw':10,
                'sigmaR':1e6,
                'n_iter':2000,
                'n_e_step':5,
                'v_start':200,
                'n_reduce_step':10,
                'v_expand_factor':0.2,
                'v_res_factor':2.0, # gets ignored if dv is specified
                'dv':None,
                'a':None, # note default below dv[0]*2.0
                'p':2.0,    
                'aprefactor':0.1, # in terms of voxels in the downsampled atlas
                'device':None, # cuda:0 if available otherwise cpu
                'dtype':torch.double,
                'downI':[1,1,1],
                'downJ':[1,1,1],      
                'W0':None,
                'priors':None,
                'update_priors':True,
                'full_outputs':False,
                'muB':None,
                'update_muB':False,
                'muA':None,
                'update_muA':False,
                'sigmaA':None,                
                'sigmaB':None,                
                'sigmaM':None,                
                'A':None,
                'Amode':0, # 0 for standard, 1 for rigid, 2 for rigid+scale, 3 for rigid using XJ for projection
                'v':None,
                'A2d':None,     
                'eA2d':1e-3, # removed eL and eT using metric, need to double check 2d case works well
                'slice_matching':False, # if true include rigid motions and contrast on each slice
                'slice_matching_start':0,
                'slice_matching_isotropic':False, # if true 3D affine is isotropic scale
                'slice_matching_initialize':False, # if True, AND no A2d specified, we will run atlas free as an initializer
                'local_contrast':None, # simple local contrast estimation mode, should be a list of ints
                'reduce_factor':0.9,
                'auto_stepsize_v':0, # 0 for no auto stepsize, or a number n for updating every n iterations
                'auto_stepsize_A':0, # 0 for no auto stepsize, or a number n for updating every n iterations
                'up_vector':None, # the up vector in the atlas, which should remain up (pointing in the -y direction) in the target
                'slice_deformation':False, # if slice_matching is also true, we will add 2d deformation. mostly use same parameters
                'ev2d':1e-6, #TODO
                'v2d':None, #TODO
                'slice_deformation_start':250, #TODO
                'slice_to_neighbor_sigma':None, # add a loss function for aligning slices to neighbors by simple least squres
                'slice_to_average_a': None,
                'small':1e-7, # for matrix inverse
                'out_of_plane':True, # if False, will project the velocity to be in plane only
                'rigid_procrustes':False, # for 2D project onto rigid using procrustes, else use svd
               }
    defaults.update(kwargs)
    kwargs = defaults
    device = kwargs['device']
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
    dtype = kwargs['dtype']
    if dtype is None:
        dtype = torch.float
    if isinstance(dtype,str):
        if dtype == 'float':
            dtype = torch.float
        elif dtype == 'float32':
            dtype = torch.float32
        elif dtype == 'float64':
            dtype = torch.float64
    
    # move the above to the right device
    I = torch.as_tensor(I,device=device,dtype=dtype)
    J = torch.as_tensor(J,device=device,dtype=dtype)
    xI = [torch.as_tensor(x,device=device,dtype=dtype) for x in xI]
    xJ = [torch.as_tensor(x,device=device,dtype=dtype) for x in xJ]
    
    
    ##########################################################################################################
    nt = kwargs['nt']
    eA = kwargs['eA']
    ev = kwargs['ev']
    eA2d = kwargs['eA2d']
    
    reduce_factor = kwargs['reduce_factor']
    auto_stepsize_v = kwargs['auto_stepsize_v']
    if auto_stepsize_v is None:
        auto_stepsize_v = 0
    auto_stepsize_A = kwargs['auto_stepsize_A']
    if auto_stepsize_A is None:
        auto_stepsize_A = 0
    
    order = kwargs['order'] 
    
    sigmaR = kwargs['sigmaR']
    n_iter = kwargs['n_iter']
    v_start = kwargs['v_start']
    n_draw = kwargs['n_draw']
    n_e_step = kwargs['n_e_step']
    n_reduce_step = kwargs['n_reduce_step']
    small = kwargs['small']
    
    v_expand_factor = kwargs['v_expand_factor']
    v_res_factor = kwargs['v_res_factor']
    dv = kwargs['dv']
    out_of_plane = kwargs['out_of_plane']
    
    a = kwargs['a']
    p = kwargs['p']
    aprefactor = kwargs['aprefactor']
    
        
    
    downI = kwargs['downI']
    downJ = kwargs['downJ']
    W0 = kwargs['W0']
    if W0 is None:
        W0 = torch.ones_like(J[0]) # this has only one channel and should not have extra dimension in front
    else:
        W0 = torch.as_tensor(W0,device=device,dtype=dtype)
    N = torch.sum(W0) 
    # missing slices
    # slice to neighbor
    slice_to_neighbor_sigma = kwargs['slice_to_neighbor_sigma']
    # this will be either a float, or None for skip it
    if slice_to_neighbor_sigma is not None:
        # we need to identify missing slices, and find the next slice
        # if we did contrast matching we would also want to provide the previous slice
        # we want to align by maching Jhat[:,this_slice] to Jhat[:,next_slice]        
        this_slice_ = []        
        for i in range(W0.shape[0]):
            if not torch.all(W0[i]==0):
                this_slice_.append(i)
        this_slice = torch.tensor(this_slice_[:-1],device=device)
        next_slice = torch.tensor(this_slice_[1:],device=device)
    slice_to_average_a = kwargs['slice_to_average_a']
    if (slice_to_average_a is not None) and (slice_to_neighbor_sigma is not None):
        raise Exception('You can specify slice to neighbor or slice to average approaches but not both')
        
        
            
    priors = kwargs['priors']
    update_priors = kwargs['update_priors']
    if priors is None and not update_priors:        
        priors = [1.0,1.0,1.0]
    full_outputs = kwargs['full_outputs']
    muA = kwargs['muA']
    update_muA = kwargs['update_muA']
    muB = kwargs['muB']
    update_muB = kwargs['update_muB']
    sigmaA = kwargs['sigmaA']    
    sigmaB = kwargs['sigmaB']    
    sigmaM = kwargs['sigmaM']    
    
    
    A = kwargs['A']
    Amode = kwargs['Amode']
    v = kwargs['v']
    
    # slice matching
    slice_matching = kwargs['slice_matching']
    if slice_matching is None:
        slice_matching = False
    if slice_matching:
        A2d = kwargs['A2d']
    slice_matching_start = kwargs['slice_matching_start']
    slice_matching_isotropic = kwargs['slice_matching_isotropic']
    rigid_procrustes = kwargs['rigid_procrustes']
    
    
    
    
    # slice deformation
    slice_deformation = kwargs['slice_deformation']
    if slice_deformation is None:
        slice_deformation = False
    if slice_deformation and not slice_matching:
        raise Exception('slice deformation is only available if slice matching is True')
    if slice_deformation:
        v2d = kwargs['v2d']
    slice_deformation_start = kwargs['slice_deformation_start']
    
    
    # local contrast, a list of ints, if empty don't do it
    local_contrast = kwargs['local_contrast']
    if local_contrast is None:
        local_contrast = []
    if local_contrast:
        local_contrast = torch.as_tensor(local_contrast, device=device)
    # up vector
    up_vector = kwargs['up_vector']
    if up_vector is not None:
        up_vector = torch.tensor(up_vector,device=device,dtype=dtype)
        if len(up_vector) != 3: raise Exception('problem with up vector')
    
    
    
    
    ##########################################################################################################    
    # domain
    dI = torch.tensor([xi[1]-xi[0] for xi in xI],device=device,dtype=dtype)
    dJ = torch.tensor([xi[1]-xi[0] for xi in xJ],device=device,dtype=dtype)    
    nI = torch.tensor(I.shape,dtype=dtype,device=device)
    nJ = torch.tensor(J.shape,dtype=dtype,device=device)
    
    # set up a domain for xv    
    # I'll put it a bit bigger than xi
    if dv is None:
        dv = dI*torch.tensor(v_res_factor,dtype=dtype,device=device) # we want this to be independent of the I downsampling, 
    else:        
        if isinstance(dv,float):
            dv = torch.tensor([dv,dv,dv],dtype=dtype,device=device)
        elif isinstance(dv,list):
            if len(dv) == 3:
                dv = torch.tensor(dv,dtype=dtype,device=device)
            else:
                raise Exception(f'dv must be a scalar or a 3 element list, but was {dv}')
        else:
            # just try it
            dv = torch.tensor(dv,dtype=dtype,device=device)
            if len(dv) != 3:
                raise Exception(f'dv must be a scalar or a 3 element list, but was {dv}')
                
                                                    
        
    # feb 4, 2022, I want it to be isotropic though
    #print(f'dv {dv}')
    if a is None:
        a = dv[0]*2.0 # so this is also independent of the I downsampling amount
    #print(f'a scale is {a}')
    x0v = [x[0] - (x[-1]-x[0])*v_expand_factor for x in xI]
    x1v = [x[-1] + (x[-1]-x[0])*v_expand_factor for x in xI]
    xv = [torch.arange(x0,x1,d,device=device,dtype=dtype) for x0,x1,d in zip(x0v,x1v,dv)]
    nv = torch.tensor([len(x) for x in xv],device=device,dtype=dtype)
    XV = torch.stack(torch.meshgrid(xv,indexing='ij'))
    #print(f'velocity size is {nv}')
    
    
    # downample    
    xI,I = downsample_image_domain(xI,I,downI)
    xJ,J,W0 = downsample_image_domain(xJ,J,downJ,W=W0)
    dI *= torch.prod(torch.tensor(downI,device=device,dtype=dtype))
    dJ *= torch.prod(torch.tensor(downJ,device=device,dtype=dtype))
    # I think the above two lines are wrong, let's just repeat
    dI = torch.tensor([xi[1]-xi[0] for xi in xI],device=device,dtype=dtype)
    dJ = torch.tensor([xi[1]-xi[0] for xi in xJ],device=device,dtype=dtype)    
    nI = torch.tensor(I.shape,dtype=dtype,device=device)
    nJ = torch.tensor(J.shape,dtype=dtype,device=device)
    
    vminI = [np.quantile(J_.cpu().numpy(),0.001) for J_ in I]
    vmaxI = [np.quantile(J_.cpu().numpy(),0.999) for J_ in I]
    vminJ = [np.quantile(J_.cpu().numpy(),0.001) for J_ in J]
    vmaxJ = [np.quantile(J_.cpu().numpy(),0.999) for J_ in J]
    
        
    XI = torch.stack(torch.meshgrid(xI,indexing='ij'))
    XJ = torch.stack(torch.meshgrid(xJ,indexing='ij'))
    
    
    # build an affine metric for 3D affine
    # this one is based on pullback metric from action on voxel locations (not image)
    # build a basis in lexicographic order and push forward using the voxel locations
    XI_ = XI.permute((1,2,3,0))[...,None]    
    E = []
    for i in range(3):
        for j in range(4):
            E.append(   ((torch.arange(4,dtype=dtype,device=device)[None,:] == j)*(torch.arange(4,dtype=dtype,device=device)[:,None] == i))*torch.ones(1,device=device,dtype=dtype) )
    # it would be nice to define scaling so that the norm of a perturbation had units of microns
    # e.g. root mean square displacement
    g = torch.zeros((12,12), dtype=torch.double, device=device)
    for i in range(len(E)):
        EiX = (E[i][:3,:3]@XI_)[...,0] + E[i][:3,-1]
        for j in range(len(E)):
            EjX = (E[j][:3,:3]@XI_)[...,0] + E[j][:3,-1]
            # matrix multiplication            
            g[i,j] = torch.sum(EiX*EjX) * torch.prod(dI) # because gradient has a factor of N in it, I think its a good idea to do sum
    # note, on july 21 I add factor of voxel size, so it can cancel with factor in cost function

    # feb 2, 2022, use double precision.  TODO: implement this as a solve when it is applied instead of inverse
    gi = torch.inverse(g.double()).to(dtype)
    

    # TODO affine metric for 2D affine
    # I'll use a quick hack for now
    # this is again based on pullback metric for voxel locations
    # need to verify that this is correct given possibly moving coordinate
    # maybe better 
    E = []
    
    for i in range(2):
        for j in range(3):
            E.append(   ((torch.arange(3,dtype=dtype,device=device)[None,:] == j)*(torch.arange(3,dtype=dtype,device=device)[:,None] == i))*torch.ones(1,device=device,dtype=dtype) )
    g2d = torch.zeros((6,6),dtype=dtype,device=device)

    
    for i in range(len(E)):
        EiX = (E[i][:2,:2]@XI_[0,...,1:,:])[...,0] + E[i][:2,-1]
        for j in range(len(E)):
            EjX = (E[j][:2,:2]@XI_[0,...,1:,:])[...,0] + E[j][:2,-1]
            g2d[i,j] = torch.sum(EiX*EjX) * torch.prod(dI[1:])

    # feb 2, 2022, use double precision.  TODO: implement this as a solve when it is applied instead of inverse
    g2di = torch.inverse(g2d.double()).to(dtype)

            
    # build energy and smoothing operator for velocity
    fv = [torch.arange(n,device=device,dtype=dtype)/d/n for n,d in zip(nv,dv)]
    FV = torch.stack(torch.meshgrid(fv,indexing='ij'))

    LL = (1.0 - 2.0*a**2 * 
              ( (torch.cos(2.0*np.pi*FV[0]*dv[0]) - 1)/dv[0]**2  
            + (torch.cos(2.0*np.pi*FV[1]*dv[1]) - 1)/dv[1]**2  
            + (torch.cos(2.0*np.pi*FV[2]*dv[2]) - 1)/dv[2]**2   ) )**(p*2)
    K = 1.0/LL

    LLpre = (1.0 - 2.0*(aprefactor*torch.max(dI))**2 * 
             ( (torch.cos(2.0*np.pi*FV[0]*dv[0]) - 1)/dv[0]**2  
             + (torch.cos(2.0*np.pi*FV[1]*dv[1]) - 1)/dv[1]**2  
             + (torch.cos(2.0*np.pi*FV[2]*dv[2]) - 1)/dv[2]**2   ) )**(p*2)
    Kpre = 1.0/LLpre
    KK = K*Kpre

    # build energy and smoothing operator for 2d velocity
    if slice_deformation:
        x0v2d = [x[0] - (x[-1]-x[0])*v_expand_factor for x in xJ[1:]]
        x1v2d = [x[-1] + (x[-1]-x[0])*v_expand_factor for x in xJ[1:]]
        xv2d = [torch.arange(x0,x1,d,device=device,dtype=dtype) for x0,x1,d in zip(x0v2d,x1v2d,dv)]
        nv2d = torch.tensor([len(x) for x in xv2d],device=device,dtype=dtype)
        XV2d = torch.stack(torch.meshgrid(xv2d,indexing='ij'))
    
    
    # now initialize variables and optimizers    
    vsize = (nt,3,int(nv[0]),int(nv[1]),int(nv[2]))
    
    if v is None:
        v = torch.zeros(vsize,dtype=dtype,device=device,requires_grad=True)
    else:
        # check the size
        if torch.all(torch.as_tensor(v.shape,device=device,dtype=dtype)==torch.as_tensor(vsize,device=device,dtype=dtype)):
            # note as_tensor will not do a copy if it is the same dtype and device
            # torch.tensor will always copy
            if isinstance(v,torch.Tensor):
                v = torch.tensor(v.detach().clone(),device=device,dtype=dtype)
            else:
                v = torch.tensor(v,device=device,dtype=dtype) 
            v.requires_grad = True
        else:
            if v.shape[1] != vsize[1]:
                raise Exception('Initial velocity must have 3 components')
            # resample it
            v = sinc_resample_numpy(v.cpu(),vsize)
            v = torch.as_tensor(v,device=device,dtype=dtype)
            v.requires_grad = True
            
    if slice_deformation:
        v2dsize = (nt,2,int(nv[0]),int(nv2d[0]),int(inv2d[1]))
        if v2d is None:
            v2d = torch.zeros(v2dsize,dtype=dtype,device=device,requires_grad=True)
        else:
            # check the size
            if torch.all(torch.as_tensor(v2d.shape,device=device,dtype=dtype)==torch.as_tensor(v2dsize,device=device,dtype=dtype)):
                if isinstance(v2d,torch.Tensor):
                    v2d = torch.tensor(v2d.detach().clone(),device=device,dtype=dtype)
                else:
                    v2d = torch.tensor(v2d,device=device,dtype=dtype)
                v2d.requires_grad = True
            else:
                if v2d.shape[1] != v2dsize[1]:
                    raise EXception('Initial 2d velocity must have 2 components')
                # resample it
                v2d = sinc_resample_numpy(v2d.cpu(),vsize)
                v2d = torch.as_tensor(v2d,device=device,dtype=dtype)
                v2d.requires_grad = True
            
            
            
    # TODO: what if A is a strong because it came from a json file?
    # should be okay, the json loader will convert it to lists of lists 
    # and that will initialize correctly
    if A is None:
        A = torch.eye(4,requires_grad=True, device=device, dtype=dtype)
    else:
        # use tensor, not as_tensor, to make a copy
        # A = torch.tensor(A,device=device,dtype=dtype)
        # This is only to bypass the warning message. Gray, Sep. 2022
        if type(A) == torch.Tensor:
            A = torch.tensor(A.detach().clone(),device=device,dtype=dtype)
        else:
            A = torch.tensor(A,device=device,dtype=dtype)
        A.requires_grad = True

        
    if slice_matching:
        if A2d is None:
            A2d = torch.eye(3,device=device,dtype=dtype)[None].repeat(J.shape[1],1,1)
            A2d.requires_grad = True
            
            
            # TODO, here if  slice_matching_initialize is true
            slice_matching_initialize = kwargs['slice_matching_initialize']
            if slice_matching_initialize:
                # do the atlas free reconstruction to start at lowest resolution
                # TODO
                pass
                
            
        else:
            # use tensor not as tensor to make a copy
            # A2d = torch.tensor(A2d, device=device, dtype=dtype)
            # This is only to bypass the warning message. Gray, Sep. 2022
            if type(A2d) == torch.Tensor:
                A2d = torch.tensor(A2d.detach().clone(),device=device, dtype=dtype)
            else:
                A2d = torch.tensor(A2d, device=device, dtype=dtype)
            A2d.requires_grad = True
        # if slice matching is on we want to add xy translation in A to A2d
        with torch.no_grad():
            A2di = torch.inverse(A2d)
            vec = A[1:3,-1]
            A2d[:,:2,-1] += (A2di[:,:2,:2]@vec[...,None])[...,0]
            A[1:3,-1] = 0
    
    WM = torch.ones(J.shape[1:], device=device, dtype=dtype)*0.8
    WA = torch.ones(J.shape[1:], device=device, dtype=dtype)*0.1
    WB = torch.ones(J.shape[1:], device=device, dtype=dtype)*0.1

    if muA is None:         
        #muA = torch.tensor([torch.max(J_[W0>0]) for J_ in J],dtype=dtype,device=device)      
        muA = torch.tensor([torch.tensor(np.quantile( (J_[W0>0]).cpu().numpy(), 0.999 ),dtype=dtype,device=device) for J_ in J],dtype=dtype,device=device)
    else: # if we input some value, we'll just use that
        muA = torch.tensor(muA,dtype=dtype,device=device)    
    if muB is None:
        #muB = torch.tensor([torch.min(J_[W0>0]) for J_ in J],dtype=dtype,device=device)  
        muB = torch.tensor([torch.tensor(np.quantile( (J_[W0>0]).cpu().numpy(), 0.001 ),dtype=dtype,device=device) for J_ in J],dtype=dtype,device=device)
    else: # if we input a value we'll just use that
        muB = torch.tensor(muB,dtype=dtype,device=device)    
    
    # TODO update to covariance, for now just diagonal
    DJ = torch.prod(dJ)
    if sigmaM is None:
        sigmaM = torch.std(J,dim=(1,2,3))*1.0#*DJ
    else:
        sigmaM = torch.as_tensor(sigmaM,device=device,dtype=dtype)
    if sigmaA is None:
        sigmaA = torch.std(J,dim=(1,2,3))*5.0#*DJ
    else:
        sigmaA = torch.as_tensor(sigmaA,device=device,dtype=dtype)
    if sigmaB is None:
        sigmaB = torch.std(J,dim=(1,2,3))*2.0#*DJ
    else:
        sigmaB = torch.as_tensor(sigmaB,device=device,dtype=dtype)
        
    if n_draw: # if n_draw is not 0, we create figures
        figE,axE = plt.subplots(1,3)
        hfigE = display(figE,display_id=True)
        figA,axA = plt.subplots(2,2)
        hfigA = display(figA,display_id=True)
        axA = axA.ravel()
        if slice_matching:
            figA2d,axA2d = plt.subplots(2,2)
            hfigA2d = display(figA2d,display_id=True)
            axA2d = axA2d.ravel()
        figI = plt.figure()
        hfigI = display(figI,display_id=True)
        figfI = plt.figure()
        hfigfI = display(figfI,display_id=True)
        figErr = plt.figure()
        hfigErr = display(figErr,display_id=True)
        figJ = plt.figure()
        hfigJ = display(figJ,display_id=True)
        figV = plt.figure()
        hfigV = display(figV,display_id=True)
        figW = plt.figure()
        hfigW = display(figW,display_id=True)
        


    Esave = []
    Lsave = []
    Tsave = []
    
    if slice_matching:
        T2dsave = []
        L2dsave = []
    maxvsave = []
    sigmaMsave = []
    sigmaAsave = []
    sigmaBsave = []

    
    ################################################################################
    # end of setup, start optimization loop
    for it in range(n_iter):
        # get the transforms        
        phii = v_to_phii(xv,v) # on the second iteration I was getting an error here 
        Ai = torch.inverse(A)
        # 2D transforms
        if slice_matching:
            A2di = torch.inverse(A2d)            
            XJ_ = torch.clone(XJ)
            # leave z component the same (x0) and transform others                   
            XJ_[1:] = ((A2di[:,None,None,:2,:2]@ (XJ[1:].permute(1,2,3,0)[...,None]))[...,0] + A2di[:,None,None,:2,-1]).permute(3,0,1,2)            
        else:
            XJ_ = XJ

        # sample points for affine
        Xs = ((Ai[:3,:3]@XJ_.permute((1,2,3,0))[...,None])[...,0] + Ai[:3,-1]).permute((3,0,1,2))
        # for diffeomorphism
        phiiAi = interp(xv,phii-XV,Xs) + Xs
        
        # transform image
        AphiI = interp(xI,I,phiiAi)

        # transform contrast
        # I'd like to support two cases, order 1 and arbitrary dim
        # or order > 1 and 1 dim
        # first step is to set up the basis functions
        Nvoxels = AphiI.shape[1]*AphiI.shape[2]*AphiI.shape[3] # why did I write 0, well its equal to 1 here        
        if type(local_contrast) == list: # an empty list means global contrast
            if I.shape[0] == 1:
                B_ = torch.ones((Nvoxels,order+1),dtype=dtype,device=device)
                for i in range(order):
                    B_[:,i+1] = AphiI.reshape(-1)**(i+1) # this assumes atlas is only dim 1, okay for now
            elif I.shape[0] > 1 and order == 1:
                # in this case, I still need a column of ones
                B_ = torch.ones((Nvoxels,AphiI.shape[0]+1),dtype=dtype,device=device)
                B_[:,1:] = AphiI.reshape(AphiI.shape[0],-1).T
            elif order == 0:
                # should be ok, skip contrast estimation
                pass
            else:
                raise Exception('Require either order = 1 or order>1 and 1D atlas')
            # note B was size N voxels by N channels
        else:
            # simple approach to local contrast
            # we will pad and refactor
            # pad AphiI
            # permute
            # reshape
            # find out how much to pad
            # this weight variable will have zeros at the end
            Wlocal = (WM*W0)[None]           
            
            # test
            Jpadv = reshape_for_local(J,local_contrast)
            Wlocalpadv = reshape_for_local(Wlocal,local_contrast)
            AphiIpadv = reshape_for_local(AphiI,local_contrast)
            
            # now basis function
            if order>1:                
                raise Exception('local not implemented yet except for linear')
            elif order == 1:
                B_ = torch.cat((torch.ones_like(AphiIpadv[...,0])[...,None],AphiIpadv),-1)
            else:
                raise Exception('Require either order = 1 or order>1 and 1D atlas')
            
        

        if not slice_matching or (slice_matching and type(local_contrast)!=list and local_contrast[0]==1):

            if type(local_contrast)==list:
                if order == 0:
                    fAphiI = AphiI
                else:
                    # global contrast mapping
                    with torch.no_grad():                
                        # multiply by weight
                        B__ = B_*(WM*W0).reshape(-1)[...,None]
                        # feb 2, 2022 converted from inv to solve and used double
                        # august 2022 add id a small constnat times identity, but I'll set it to zero for now so no change
                        #small = 1e-2*0
                        # september 2023, set to 1e-4, because I was getting nan
                        
                        BB = (B__.T@B_).double() 
                        BB = BB + torch.eye(B__.shape[1],device=device,dtype=torch.float64)*(torch.amax(BB)+1)*small
                        coeffs = torch.linalg.solve(BB, 
                                                    (B__.T@(J.reshape(J.shape[0],-1).T)).double() ).to(dtype)
                    fAphiI = ((B_@coeffs).T).reshape(J.shape) # there are unnecessary transposes here, probably slowing down, to fix later
            else:
                # local contrast estimation using refactoring
                with torch.no_grad():
                    BB = B_.transpose(-1,-2)@(B_*Wlocalpadv)
                    BJ = B_.transpose(-1,-2)@(Jpadv*Wlocalpadv)
                    
                    # convert to double here
                    coeffs = torch.linalg.solve( BB.double() + torch.eye(BB.shape[-1],device=BB.device,dtype=torch.float64)*(torch.amax(BB,(1,2),keepdims=True)+1).double()*small,BJ.double()).to(dtype)
                fAphiIpadv = (B_@coeffs).reshape(Jpadv.shape[0],Jpadv.shape[1],Jpadv.shape[2],
                                                 local_contrast[0].item(),local_contrast[1].item(),local_contrast[2].item(), 
                                                 Jpadv.shape[-1])
                # reverse this permutation (1,3,5,2,4,6,0)
                fAphiIpad_ = fAphiIpadv.permute(6,0,3,1,4,2,5)        
                fAphiIpad = fAphiIpad_.reshape(Jpadv.shape[-1], 
                                              Jpadv.shape[0]*local_contrast[0].item(), 
                                              Jpadv.shape[1]*local_contrast[1].item(), 
                                              Jpadv.shape[2]*local_contrast[2].item())
                fAphiI = fAphiIpad[:,:J.shape[1],:J.shape[2],:J.shape[3]]
                # todo, symmetric cropping and padding


        else: # with slice matching I need to solve these equation for every slice                        
            # recall B_ is size nvoxels by nchannels
            if order == 0:
                fAphiI = AphiI
            else:
                B_ = B_.reshape(J.shape[1],-1,B_.shape[-1])




                # now be is nslices x npixels x nchannels B
                with torch.no_grad():
                    # multiply by weight                
                    B__ = B_*(WM*W0).reshape(WM.shape[0],-1)[...,None]     
                    # B__ is shape nslices x npixels x nchannelsB
                    BB = (B__.transpose(-1,-2)@B_).double()
                    # BB is shape nslices x nchannelsb x nchannels b
                    # add a bit to the diagonal
                    
                    BB = BB + torch.eye(BB.shape[-1],device=BB.device,dtype=BB.dtype)[None].repeat((BB.shape[0],1,1)).double()*(torch.amax(BB,(1,2),keepdims=True)+1)*small

                    J_ = (J.permute(1,2,3,0).reshape(J.shape[1],-1,J.shape[0]))        
                    # J_ is shape nslices x npixels x nchannelsJ
                    # B__.T is shape nslices x nchannelsB x npixels
                    BJ = (B__.transpose(-1,-2))@ J_
                    # BJ is shape nslices x nchannels B x nchannels J                    
                    
                    #coeffs = torch.inverse(BB) @ BJ
                    coeffs = torch.linalg.solve(BB,BJ.double()).to(dtype)
                    # coeffs is shape nslices x nchannelsB x nchannelsJ
                    if torch.any(torch.isnan(coeffs)) or torch.any(torch.isinf(coeffs)):
                        print(coeffs)
                        print(BB)
                        asdf


                fAphiI = (B_[...,None,:]@coeffs[:,None]).reshape(J.shape[1],J.shape[2],J.shape[3],J.shape[0]).permute(-1,0,1,2)
        

        
        err = (fAphiI - J)
        err2 = (err**2*(WM*W0))
        # most of my updates are below (at the very end), but I'll update this here because it is in my cost function
        # note that in my derivation, the sigmaM should have this factor of DJ
        sseM = torch.sum( err2,(-1,-2,-3))        
                
        JmmuA2 = (J - muA[...,None,None,None])**2
        JmmuB2 = (J - muB[...,None,None,None])**2
                
        if not it%n_e_step:
            with torch.no_grad():                                
                if update_priors:
                    priors = [torch.sum(WM*W0)/N, torch.sum(WA*W0)/N, torch.sum(WB*W0)/N]                         
                
                WM = (1.0/2.0/np.pi)**(J.shape[0]/2)/torch.prod(sigmaM) * torch.exp(  -torch.sum(err**2/2.0/sigmaM[...,None,None,None]**2,0)  )*priors[0]
                WA = (1.0/2.0/np.pi)**(J.shape[0]/2)/torch.prod(sigmaA) * torch.exp(  -torch.sum(JmmuA2/2.0/sigmaA[...,None,None,None]**2,0)  )*priors[1]
                WB = (1.0/2.0/np.pi)**(J.shape[0]/2)/torch.prod(sigmaB) * torch.exp(  -torch.sum(JmmuB2/2.0/sigmaB[...,None,None,None]**2,0)  )*priors[2]
                WS = WM+WA+WB                
                WS += torch.max(WS)*1e-12 # for numerical stability, but this may be leading to issues
                WM /= WS
                WA /= WS
                WB /= WS
                # todo think gabout MAP EM instead of ML EM, some dirichlet prior                
                # note, I seem to be getting black in my mask, why would it be black?
                



        # matching cost        
        # sum of squares when it is known (note sseM includes weights)
        EM = torch.sum(sseM/sigmaM**2)*DJ/2.0
        # here I have a DJ
        # and for slice to neighbor
        if slice_to_neighbor_sigma is not None:
            A2di = torch.linalg.inv(A2d)
            meanshift = torch.mean(A2di[:,0:2,-1],dim=0)
            xJshift = [x.clone() for x in xJ]
            xJshift[1] += meanshift[0]
            xJshift[2] += meanshift[1]
            XJshift = torch.stack(torch.meshgrid(xJshift,indexing='ij'))
            XJ_ = torch.clone(XJshift)
            # leave z component the same (x0) and transform others                   
            XJ_[1:] = ((A2d[:,None,None,:2,:2]@ (XJshift[1:].permute(1,2,3,0)[...,None]))[...,0] + A2d[:,None,None,:2,-1]).permute(3,0,1,2)      
            RiJ = interp(xJ,J,XJ_)
            RiW = interp(xJ,W0[None],XJ_)

            # there is an issue here with missing data
            # i.e.the boundaries of the image
            # we don't want to line up the boundaries, that would be bad
            # so we should probably have some kind of weight
            # the problem with these weights is that generally we can decrease the cost by just moving it off screen
            EN = torch.sum((RiJ[:,this_slice] - RiJ[:,next_slice])**2*RiW[:,this_slice]*RiW[:,next_slice])/2.0/slice_to_neighbor_sigma**2*DJ

        if slice_to_average_a is not None:
            # on the first iteration we'll do some setup
            with torch.no_grad():

                meanshift = torch.mean(A2di[:,0:2,-1],dim=0)
                xJshift = [x.clone() for x in xJ]
                xJshift[1] += meanshift[0]
                xJshift[2] += meanshift[1]
                XJshift = torch.stack(torch.meshgrid(xJshift,indexing='ij'))
                XJ__ = torch.clone(XJshift)
                # leave z component the same (x0) and transform others                   
                XJ__[1:] = ((A2d[:,None,None,:2,:2]@ (XJshift[1:].permute(1,2,3,0)[...,None]))[...,0] + A2d[:,None,None,:2,-1]).permute(3,0,1,2)      
                RiJ = interp(xJ,J,XJ__)
                RiW0 = (interp(xJ,W0[None],XJ__)==1) # I want this to be a hard weight
                
                # sigmaM is either a 1D tensor, or might be a 0D tensor
                if (sigmaM.ndim > 0) and  (len(sigmaM) != 1):
                    sigmaM_ = sigmaM[...,None,None,None]
                else:
                    sigmaM_ = sigmaM
                RiW = RiW0*interp(xJ,WM[None],XJ__)/sigmaM_**2

                # TODO: padding
                npadblur = 5
                fz = torch.arange(nJ[1]+2*npadblur,device=device,dtype=dtype)/dJ[0]/(nJ[1]+2*npadblur)
                # discrete laplacian in the Fourier domain
                # since it is laplacian it is negative definite
                Lblur = slice_to_average_a**2*2*(torch.cos(2.0*np.pi*fz*dJ[0]) - 1.0)/dJ[0]**2                         


                if 'Jblur' not in locals():
                    Jblur = RiJ.clone().detach() # initialize (*0?)
                niter_Jblur = 1
                for it_Jblur in range(niter_Jblur):
                    # normalize the weights so max is 1
                    # TODO add padding here
                    Wnorm = torch.max(RiW)                        
                    RiWnorm = RiW / Wnorm
                    Jblur = RiJ*RiWnorm + Jblur*(1-RiWnorm)
                    # pad it
                    Jblur = torch.nn.functional.pad(Jblur,(0,0,0,0,npadblur,npadblur),mode='reflect')
                    Jblurhat = torch.fft.fftn(Jblur,dim=1)/( (1.0 + Lblur**2/Wnorm)[None,:,None,None] )
                    Jblur = torch.fft.ifftn(Jblurhat,dim=1)[:,npadblur:-npadblur].real

                # calculate the energy of this J
                # not sure this is right due to padding
                EN = torch.sum(    torch.abs(Jblurhat)**2*Lblur[None,:,None,None]**2  )/(J.shape[1]+2*npadblur)/2.0 * DJ
                
            # apply A2d to Jblur, and match to AphiI
            # we need to sample at the points XJ_ above   , this is A2di@XJ                 
            RJblur = interp(xJshift,Jblur,XJ_)
            if (sigmaM.ndim > 0) and  (len(sigmaM) != 1):
                    sigmaM_ = sigmaM[...,None,None,None]
            else:
                sigmaM_ = sigmaM
            EN = EN + torch.sum((J - RJblur)**2/sigmaM_**2*WM[None]*W0[None])/2.0*DJ
            

        
        
        # reg cost (note that with no complex, there are two elements on the last axis)
        #version_num = int(torch.__version__.split('.')[1])
        #version_num = float('.'.join( torch.__version__.split('.')[:2] ))
        version_num = [int(x) for x in torch.__version__.split('.')[:2]]
        if version_num[0] <= 1 and version_num[1]< 7:
            vhat = torch.rfft(v,3,onesided=False)
        else:
            #vhat = torch.view_as_real(torch.fft.fftn(v,dim=3,norm="backward"))
            # dec 14, I don't think the above is correct, need to tform over dim 2,3,4 (zyx)
            # note that "as real" gives real and imaginary as a last index
            vhat = torch.view_as_real(torch.fft.fftn(v,dim=(2,3,4)))
        
        ER = torch.sum(torch.sum(vhat**2,(0,1,-1))*LL)/torch.prod(nv)*torch.prod(dv)/nt/2.0/sigmaR**2
        

        # total cost 
        E = EM + ER
        if (slice_to_neighbor_sigma is not None) or (slice_to_average_a is not None):            
            E = E + EN
        

        # gradient
        E.backward()

        # covector to vector
        if version_num[0] <= 1 and version_num[1]< 7:
            vgrad = torch.irfft(torch.rfft(v.grad,3,onesided=False)*(KK)[None,None,...,None],3,onesided=False)
        else:

            #vgrad = torch.view_as_real(torch.fft.ifftn(torch.fft.fftn(v.grad,dim=3,norm="backward")*(KK),
            #    dim=3,norm="backward"))
            #vgrad = vgrad[...,0]
            # dec 14, 2021 I don't think the above is correct, re dim
            vgrad = torch.fft.ifftn(torch.fft.fftn(v.grad,dim=(2,3,4))*(KK),dim=(2,3,4)).real
            
        # Agrad = (gi@(A.grad[:3,:4].reshape(-1).to(dtype=torch.double))).reshape(3,4)
        Agrad = torch.linalg.solve(g.to(dtype=torch.double), A.grad[:3,:4].reshape(-1).to(dtype=torch.double)).reshape(3,4).to(dtype=dtype)
        if slice_matching:
            # A2dgrad = (g2di@(A2d.grad[:,:2,:3].reshape(A2d.shape[0],6,1).to(dtype=torch.double))).reshape(A2d.shape[0],2,3)
            A2dgrad = torch.linalg.solve(g2d.to(dtype=torch.double), A2d.grad[:,:2,:3].reshape(A2d.shape[0],6,1).to(dtype=torch.double)).reshape(A2d.shape[0],2,3).to(dtype=dtype)
            

        # plotting
        if (slice_to_neighbor_sigma is None) and (slice_to_average_a is None):
            Esave.append([E.detach().cpu(),EM.detach().cpu(),ER.detach().cpu()])        
        else:
            Esave.append([E.detach().cpu(),EM.detach().cpu(),ER.detach().cpu(),EN.detach().cpu()])
        Tsave.append(A[:3,-1].detach().clone().squeeze().cpu().numpy())
        Lsave.append(A[:3,:3].detach().clone().squeeze().reshape(-1).cpu().numpy())
        maxvsave.append(torch.max(torch.abs(v.detach())).clone().cpu().numpy())
        if slice_matching:
            T2dsave.append(A2d[:,:2,-1].detach().clone().squeeze().reshape(-1).cpu().numpy())
            L2dsave.append(A2d[:,:2,:2].detach().clone().squeeze().reshape(-1).cpu().numpy())
        # a nice check on step size would be to see if these are oscilating or monotonic
        if n_reduce_step and (it > 10 and not it%n_reduce_step):
            
            checksign0 = np.sign(maxvsave[-1] - maxvsave[-2])
            checksign1 = np.sign(maxvsave[-2] - maxvsave[-3])
            checksign2 = np.sign(maxvsave[-3] - maxvsave[-4])
            if np.any((checksign0 != checksign1)*(checksign1 != checksign2) ):
                ev *= reduce_factor
                print(f'Iteration {it} reducing ev to {ev}')

            checksign0 = np.sign(Tsave[-1] - Tsave[-2])
            checksign1 = np.sign(Tsave[-2] - Tsave[-3])
            checksign2 = np.sign(Tsave[-3] - Tsave[-4])
            reducedA = False
            if np.any((checksign0 != checksign1)*(checksign1 != checksign2)):
                eA *= reduce_factor
                print(f'Iteration {it}, translation oscilating, reducing eA to {eA}')
                reducedA = True
            checksign0 = np.sign(Lsave[-1] - Lsave[-2])
            checksign1 = np.sign(Lsave[-2] - Lsave[-3])
            checksign2 = np.sign(Lsave[-3] - Lsave[-4])
            if np.any( (checksign0 != checksign1)*(checksign1 != checksign2) ) and not reducedA:
                eA *= reduce_factor
                print(f'Iteration {it}, linear oscilating, reducing eA to {eA}')
                
            # to do, check sign for a2d

        
        if n_draw and (not it%n_draw or it==n_iter-1):
            #print(A)
            axE[0].cla()
            axE[0].plot(np.array(Esave)[:,0])
            axE[0].plot(np.array(Esave)[:,1])
            if slice_to_neighbor_sigma is not None:
                axE[0].plot(np.array(Esave)[:,3])
                
            axE[0].set_title('Energy')
            axE[1].cla()
            axE[1].plot(np.array(Esave)[:,1])
            if (slice_to_neighbor_sigma is not None) or (slice_to_average_a is not None):
                axE[1].plot(np.array(Esave)[:,3])
            axE[1].set_title('Matching')
            axE[2].cla()
            axE[2].plot(np.array(Esave)[:,2])
            axE[2].set_title('Reg')


            
            
            # if slice matching, it would be better to see the reconstructed version
            if not slice_matching:
                _ = draw(AphiI.detach().cpu(),xJ,fig=figI,vmin=vminI,vmax=vmaxI)
                figI.suptitle('AphiI')                
                _ = draw(fAphiI.detach().cpu(),xJ,fig=figfI,vmin=vminJ,vmax=vmaxJ)
                figfI.suptitle('fAphiI')     
                
                
                _ = draw(fAphiI.detach().cpu() - J.cpu(),xJ,fig=figErr)
                figErr.suptitle('Err')
                _ = draw(J.cpu(),xJ,fig=figJ,vmin=vminJ,vmax=vmaxJ)
                figJ.suptitle('J')

            else:                                     
                # find a sampling grid
                A2di = torch.linalg.inv(A2d.clone().detach())
                meanshift = torch.mean(A2di[:,0:2,-1],dim=0)
                #print(meanshift)
                xJshift = [x.clone() for x in xJ]
                xJshift[1] += meanshift[0]
                xJshift[2] += meanshift[1]
                XJshift = torch.stack(torch.meshgrid(xJshift,indexing='ij'))
                XJ_ = torch.clone(XJshift)
                # leave z component the same (x0) and transform others                   
                XJ_[1:] = ((A2d[:,None,None,:2,:2]@ (XJshift[1:].permute(1,2,3,0)[...,None]))[...,0] + A2d[:,None,None,:2,-1]).permute(3,0,1,2)      
                RiJ = interp(xJ,J,XJ_)
                
                Xs_ = ((Ai[:3,:3]@XJshift.permute((1,2,3,0))[...,None])[...,0] + Ai[:3,-1]).permute((3,0,1,2))
                phiiAi_ = interp(xv,phii-XV,Xs_) + Xs_
        
                # transform image
                AphiI_ = interp(xI,I,phiiAi_)
                if np.any([np.any(np.isnan(x.numpy())) for x in xJshift]):
                    print(A2d)
                    print(A2di)
                    print(meanshift)
                    print(xJshift)
                    asdf
                _ = draw(AphiI_.detach().cpu(),xJshift,fig=figI,vmin=vminI,vmax=vmaxI)
                figI.suptitle('AphiI')
                
                fAphiI_ = interp(xJ,fAphiI,XJ_)
                _ = draw(fAphiI_.detach().cpu(),xJshift,fig=figfI,vmin=vminJ,vmax=vmaxJ)
                figfI.suptitle('fAphiI')  
                
                
                _ = draw(fAphiI_.detach().cpu() - RiJ.cpu(),xJshift,fig=figErr)
                figErr.suptitle('Err')
                _ = draw(RiJ.cpu(),xJ,fig=figJ,vmin=vminJ,vmax=vmaxJ)
                figJ.suptitle('RiJ')
                
                
                if slice_to_average_a is not None:
                    if 'figN' not in locals():
                        figN = plt.figure()
                        hfigN = display(figN,display_id=True)
                    _ = draw(Jblur.cpu(),xJshift,fig=figN,vmin=vminJ,vmax=vmaxJ)
                    figN.suptitle('Jblur')
                    # save them for debugging!
                    #figN.savefig(f'Jblurtmp/Jblur_{it:06d}.jpg')
                    
                    # TODO during debugging                
                    #figJ.savefig(f'Jblurtmp/Jrecon_{it:06d}.jpg')
                    
                    #if 'figN_' not in locals():
                    #    figN_ = plt.figure()
                    #_ = draw(RJblur.cpu(),xJ,fig=figN_,vmin=vminJ,vmax=vmaxJ)
                    #figN_.suptitle('RJblur')
                    #figN_.canvas.draw()
                    
                    
            
            

            axA[0].cla()
            axA[0].plot(np.array(Tsave))
            axA[0].set_title('T')
            axA[1].cla()
            axA[1].plot(np.array(Lsave))
            axA[1].set_title('L')
            axA[2].cla()
            axA[2].plot(np.array(maxvsave))
            axA[2].set_title('maxv')

                 

            if slice_matching:
                axA2d[0].cla()
                axA2d[0].plot(np.array(T2dsave))
                axA2d[0].set_title('T2d')
                axA2d[1].cla()
                axA2d[1].plot(np.array(L2dsave))
                axA2d[1].set_title('L2d')


            figV.clf()
            draw(v[0].detach(),xv,fig=figV)
            figV.suptitle('velocity')

            figW.clf()
            draw(torch.stack((WM*W0,WA*W0,WB*W0)),xJ,fig=figW,vmin=0.0,vmax=1.0)
            figW.suptitle('Weights')

            # TODO check if widget backend, and only then call update
            figE.canvas.draw()
            hfigE.update(figE)
            figI.canvas.draw()
            hfigI.update(figI)
            figfI.canvas.draw()
            hfigfI.update(figfI)
            figErr.canvas.draw()
            hfigErr.update(figErr)
            figA.canvas.draw()
            hfigA.update(figA)
            figV.canvas.draw()
            hfigV.update(figV)
            figW.canvas.draw()
            hfigW.update(figW)
            figJ.canvas.draw()
            hfigJ.update(figJ)
            if slice_matching:
                figA2d.canvas.draw()
                hfigA2d.update(figA2d)
            if slice_to_average_a is not None:
                figN.canvas.draw()
                hfigN.update(figN)

        ################################################################################
        # update parameters        
        with torch.no_grad():
            # here we'll do an automatic step size estimation for e based on a quadratic/Gauss-newton approximation
            # e1 = -int  err*DI*grad  dx/sigmaM**2  + int (Lv)*(Lgrad)dx/sigmaR**2
            # e2 = int  |DI*grad|^2  dx/sigmaM**2  + int (Lgrad)*(Lgrad)dx/sigmaR**2
            # e = e1/e2
            # what do I need to do to implement this?
            # the only tricky thing is to resample grad onto the same grid as J            
            if auto_stepsize_v and not it%auto_stepsize_v: # this will happen at the first time
                # WORKING
                # first we need to deal with the affine
                # we will transform J and fAphiI and W back to the space of I                
                Xs = ((A[:3,:3]@XI.permute(1,2,3,0)[...,None])[...,0] + A[:3,-1]).permute(-1,0,1,2)
                AiJ = interp(xJ,J,Xs)
                fphiI = interp(xJ,fAphiI,Xs)
                AiW = interp(xJ,(WM*W0)[None],Xs)
                
                
                # for the data attachment term
                # now we need DI
                dxI = [(x[1] - x[0]).item() for x in xI]
                DfphiI = torch.stack(torch.gradient(fphiI,spacing=dxI,dim=(1,2,3)),0)
                # I will just use v0 (t=0)
                # we need to compute DIg
                # we need to resample vgrad
                vgradI = interp(xv,vgrad[0],XI)
                DfphiIg = torch.sum(DfphiI*vgradI,0) # row times column gives scalar
                errDfphiIg = torch.sum( (fphiI - AiJ)*DfphiIg , 0)
                DfphiIgDfphiiG = torch.sum( DfphiIg*DfphiIg , 0)
                
                
                # for the regularization term we need these quantities
                # we will sum over them and we don't need any resampling 
                
                LLvgrad = torch.fft.ifftn(torch.fft.fftn(vgrad[0],dim=(-1,-2,-3)),dim=(-1,-2,-3)).real
                vLLvgrad = torch.sum(v[0]*LLvgrad,0)
                vgradLLvgrad = torch.sum(vgrad[0]*LLvgrad,0)
                
                # now we have to compute the sums
                e1 = - torch.sum( errDfphiIg*AiW )/sigmaM**2*np.prod(dxI)*torch.linalg.det(A) + torch.sum( vLLvgrad )/sigmaR**2*torch.prod(dv)
                e2 = torch.sum(DfphiIgDfphiiG*AiW )/sigmaM**2*np.prod(dxI)*torch.linalg.det(A) + torch.sum( vgradLLvgrad )/sigmaR**2*torch.prod(dv)
                
                ev_auto = e1/e2
                #print(e1,e2,ev_auto)
                # when using auto step size we expect ev to be something of order 1
            elif not auto_stepsize_v: 
                ev_auto = 1.0
            
            if it >= v_start:
                v -= vgrad*ev*ev_auto                
            v.grad.zero_()
            
            A[:3] -= Agrad*eA
            if Amode==1: # 1 means rigid
                # TODO update this with center of mass
                U,S,VH = torch.linalg.svd(A[:3,:3])
                A[:3,:3] = U@VH
            elif Amode==2: # 2 means rigid + scale
                U,S,VH = torch.linalg.svd(A[:3,:3])
                A[:3,:3] = (U@VH)*torch.exp(torch.mean(torch.log(S)))
            elif Amode==0: # 0 means nothing
                pass
            elif Amode == 3:
                # here we use the sample points to do the projection
                # that is, we want to find the R which is closets to A
                # in the sense that it maps voxels to similar locations
                # so which voxels to use? We could either use the target voxels
                # or the template voxels
                # it makes more sense to me to use the target voxels
                # argmin_R |R^{-1}XJ - A^{-1}XJ|^2_W
                # where W are my weights from above
                # for now I'll skip the weights
                # here XJ is a 4xN vector where N is the number of voxels
                # then we can use a procrustes method to find the solution for R^{-1}
                # 
                A = project_affine_to_rigid(A,XJ)                
            else:
                raise Exception('Amode must be 0 (normal), 1 (rigid), or 2 (rigid+scale), or 3 (rigid using XJ for projection)')
            A.grad.zero_()
            
            if not out_of_plane:
                # added dec 22, 2024
                # need to project onto the plane
                # how do we do that?
                # get the out of plane vector
                # we act on them with Ai
                # we normalize them
                Ai = torch.linalg.inv(A)
                e0 = torch.tensor([1.0,0.0,0.0],dtype=A.dtype,device=A.device)                
                e0 = Ai[:3,:3]@e0
                e0 = e0/torch.sqrt(torch.sum(e0**2))
                # make a projection matrix
                projection = torch.eye(3) - e0[:,None]*e0[None,:]
                # use the projection to zero out the v
                v = (projection@v.permute(0,2,3,4,1)[...,None]).permute(0,-1,1,2,3)
            
            if slice_matching:
                
                # project A to isotropic and normal up
                # isotropic (done, really where is normal up?)
                # TODO normal                
                # what I really should do is parameterize this group and work out a metric                
                if slice_matching_isotropic:
                    u,s,vh = torch.linalg.svd(A[:3,:3])
                    s = torch.exp(torch.mean(torch.log(s)))*torch.eye(3,device=device,dtype=dtype)
                    A[:3,:3] = u@s@vh  # why transpose? torch svd does not return the transpose, but this will be deprecated (fixed)
                
                if it > slice_matching_start:
                    A2d[:,:2,:3] -= A2dgrad*eA2d # already scaled
                
                
                A2d.grad.zero_()
                
                
                # project onto rigid
                if not rigid_procrustes:
                    u,s,v_ = torch.svd(A2d[:,:2,:2])
                    A2d[:,:2,:2] = u@v_.transpose(1,2)
                    A2d.grad.zero_()
                else:
                
                    # TODO, when not centered at origin, I may need a different projection (see procrustes)
                    # Let X be the voxel locations in the target image J
                    # then let A2dX = Y be the transformed voxel locations with a nonrigid transform
                    # then we want to find R to minimize |RX - Y|^2
                    # this is done in 2 steps, first we center
                    # then we svd

                    # these are the untransformed points
                    X = torch.clone(XJ[1:])

                    # these are the transformed points Y
                    # we will need to update this, otherwise I'm just projecting onto the same one as last time
                    A2di = torch.linalg.inv(A2d) 
                    Y = ((A2di[:,None,None,:2,:2]@ (X.permute(1,2,3,0)[...,None]))[...,0] + A2di[:,None,None,:2,-1]).permute(3,0,1,2)                            
                    # for linear algebra, vector components at end
                    X = X.permute(1,2,3,0)
                    Y = Y.permute(1,2,3,0)
                    # we want to find a rigid transform of X to match Y (sum over row and column)
                    Xbar = torch.mean(X,dim=(1,2),keepdims=True)
                    Ybar = torch.mean(Y,dim=(1,2),keepdims=True)
                    X = X - Xbar
                    Y = Y - Ybar
                    # now we want to find the best rotation that matches X to Y
                    # we want
                    # min |RX - Y|^2
                    # min tr[(RX-Y)(RX-Y)^T]
                    # min -tr[RXY^T]
                    # max tr[R(XY^T)]
                    # let XY^T = USV^T
                    # then
                    # max tr[R (USV^T)]
                    # max tr[(V^T R U) S]
                    # this is maximized when the bracketted term is identity
                    # so V^T R U = id
                    # R = V U^T
                    # note xyz is first dimension, I will want it to be last
                    # for tanslation
                    # now I need the translation part
                    # but this should just be
                    # R(X-Xbar) = (Y-Ybar)
                    # RX - RXbar = Y-Ybar
                    # RX + [-RXbar + Ybar] = Y
                    S = X.reshape(X.shape[0],-1,2).transpose(-1,-2) @ Y.reshape(X.shape[0],-1,2)
                    U,_,Vh = torch.linalg.svd(S)
                    #R = U.transpose(-1,-2)@Vh.transpose(-1,-2)
                    #R = R.transpose(-1,-2) # ? this seems to work
                    R = Vh.transpose(-1,-2)@U.transpose(-1,-2)
                    T = ((-R[:,None,None,]@Xbar[...,None])[...,0] + Ybar)[:,0,0,:]
                    # now we need to stack them together
                    A2di_ = torch.zeros(T.shape[0],3,3,dtype=dtype,device=device)
                    A2di_[:,:2,:2] = R
                    A2di_[:,:2,-1] = T
                    A2di_[:,-1,-1] = 1
                    A2d.data = torch.linalg.inv(A2di_)
                
            
            
                
                
                
                # move any xy translation into 2d
                # to do this I will have to account for any linear transformation
                vec = A[1:3,-1]
                A2d[:,:2,-1] += (A2di[:,:2,:2]@vec[...,None])[...,0]
                A[1:3,-1] = 0
                
                if up_vector is not None:
                    #print(up_vector)
                    # what direction does the up vector point now?
                    new_up = A[:3,:3]@up_vector
                    #print(new_up)
                    # now we ignore the z component
                    new_up_2d = new_up[1:]
                    #print(new_up_2d)
                    # we'll find the rotation angle, with respect to the -y axis
                    # use standard approach to find angle from x axis, and add 90 deg
                    angle = torch.atan2(new_up_2d[0],new_up_2d[1]) + np.pi/2 # of the form x y (TODO double check)
                    #print(angle*180/np.pi)
                    # form a rotation matrix by the negative of this angle
                    rot = torch.tensor([[1.0,0.0,0.0,0.0],[0.0,torch.cos(angle),-torch.sin(angle),0.0],[0.0,torch.sin(angle),torch.cos(angle),0.0],[0.0,0.0,0.0,1.0]],dtype=dtype,device=device)
                    #print(rot)
                    # now we have to apply this matrix to the 3D (on the left)
                    # and its inverse to the 2D (on the right)
                    # so they will cancel out
                    # BUT, what do I do about the translation?
                    # of course, I can add a translation to rot
                    # but do to the above issue (no xy translation on A), I think it should just be zero
                    
                    #print('A before',A)
                    A[:,:] = rot@A
                    #print('A after',A)
                    #print(torch.linalg.inv(rot)[1:,1:])
                    A2d[:,:,:] = A2d@torch.linalg.inv(rot)[1:,1:]
                    #double check
                    #print('test up',A[:3,:3]@up_vector)
                    # I expect this should now point exactly up
                    # but it seems to not
                    
                    
            

            # other terms in M step (M-maximization verus E-expectation ), these don't actually matter until I update
            WAW0 = (WA*W0)[None]
            WAW0s = torch.sum(WAW0)
            if update_muA:
                muA = torch.sum(J*WAW0,dim=(-1,-2,-3))/WAW0s
            
            

            WBW0 = (WB*W0)[None]
            WBW0s = torch.sum(WBW0)
            if update_muB:
                muB = torch.sum(J*WBW0,dim=(-1,-2,-3))/WBW0s
            
            
        if not it%10:
            # todo print other info
            #print(f'Finished iteration {it}')
            pass
            
    # outputs
    out = {'A':A.detach().clone().cpu(),
           'v':v.detach().clone().cpu(),
           'xv':[x.detach().clone() for x in xv]}
    if slice_matching:
        out['A2d'] = A2d.detach().clone().cpu()
    if full_outputs:
        # other data I may need
        out['WM'] = WM.detach().clone().cpu()
        out['WA'] = WA.detach().clone().cpu()
        out['WB'] = WB.detach().clone().cpu()
        out['W0'] = W0.detach().clone().cpu()
        out['muB'] = muB.detach().clone().cpu()
        out['muA'] = muA.detach().clone().cpu()
        out['sigmaB'] = sigmaB.detach().clone().cpu()
        out['sigmaA'] = sigmaA.detach().clone().cpu()
        out['sigmaM'] = sigmaM.detach().clone().cpu()
        if order>0:
            out['coeffs'] = coeffs.detach().clone().cpu()
        # return figures
        if n_draw:
            out['figA'] = figA
            out['figE'] = figE
            out['figI'] = figI        
            out['figfI'] = figfI        
            out['figErr'] = figErr        
            out['figJ'] = figJ        
            out['figW'] = figW        
            out['figV'] = figV        
        # others ...
    return out




    
    
    
# everything in the config will be either a list of the same length as downI
# or a list of length 1
# or a scalar 
def emlddmm_multiscale(**kwargs):
    '''
    Run the emlddmm algorithm multiple times, restarting 
    with the results of the previous iteration. This is intended
    to be used to register data from coarse to fine resolution.
    
    Parameters
    ----------
    emlddmm parameters either as a list of length 1 (to use the same value
    at each iteration) or a list of length N (to use different values at 
    each of the N iterations).
    
    Returns
    -------
    A list of emlddmm outputs (see documentation for emlddmm)
    
    '''

    # how many levels?
    # note I expect downI to be either a list, or a list of lists, not numpy array
    if 'downI' in kwargs:
        downI = kwargs['downI']        
        if type(downI[0]) == list:
            nscales = len(downI)
        else:
            nscales = 1
        print(f'Found {nscales} scales')
        
        
    
    outputs = []
    for d in range(nscales):
        # now we have to convert the kwargs to a new dictionary with only one value
        params = {}
        for key in kwargs:
            test = kwargs[key]
                        
            
            # general cases
            if type(test) == list:
                if len(test) > 1:
                    params[key] = test[d]
                else:
                    params[key] = test[0]
            else: # not a list, e.g. inputing v as a numpy array
                params[key] = test            
        
        if 'sigmaM' not in params:
            params['sigmaM'] = np.ones(kwargs['J'].shape[0])
        if 'sigmaB' not in params:
            params['sigmaB'] = np.ones(kwargs['J'].shape[0])*2.0
        if 'sigmaA' not in params:
            params['sigmaA'] = np.ones(kwargs['J'].shape[0])*5.0
        #print(f'starting emlddmm with params')
        #print(params)
        output = emlddmm(**params)
        # I should save an output at each iteration
        outputs.append(output)

        A = output['A']
        v = output['v']
        
        kwargs['A'] = A
        kwargs['v'] = v
        if 'slice_matching' in params and params['slice_matching']:
            A2d = output['A2d']
            kwargs['A2d'] = A2d
        
    return outputs # should I return the whole list outputs?

    
# we need to output the transformations as vtk, with their companion jsons (TODO)
# note Data with implicit topology (structured data such as vtkImageData and vtkStructuredGrid) 
# are ordered with x increasing fastest, then y,thenz .
# this is notation, it means they expect first index fastest in terms of their notation
# I am ignoring the names xyz, and just using fastest to slowest
# note dataTypeis  one  of  the  types
# bit,unsigned_char,char,unsigned_short,short,unsigned_int,int,unsigned_long,long,float,ordouble.
# I should only need the latter 2
dtypes = {    
        np.dtype('float32'):'float',
        np.dtype('float64'):'double',
        np.dtype('uint8'):'unsigned_char',
        np.dtype('uint16'):'unsigned_short',
        np.dtype('uint32'):'unsigned_int',
        np.dtype('uint64'):'unsigned_long',
        np.dtype('int8'):'char',
        np.dtype('int16'):'short',
        np.dtype('int32'):'int',
        np.dtype('int64'):'long',
    }


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


class Image:
    '''

    Attributes
    ----------
    space : string
        name of the image space
    name : string
        image name. This is provided when instantiating an Image object.
    x : list of numpy arrays
        image voxel coordinates
    data : numpy array
        image data
    title : string
        image title passed by the read_data function
    names : list of strings
        information about image data dimensions
    mask : array
        image mask
    path : string
        path to image file or directory containing the 2D image series
    
    Methods
    -------
    normalize(norm='mean', q=0.99)
        normalize image
    downsample(down)
        downsample image, image coordinated, and mask
    fnames()
        get filenames of 2D images in a series, or return the single filename of an image volume
    
    '''
    def __init__(self, space, name, fpath, mask=None, x=None):
        '''
        Parameters
        ----------
        space : string
            Name for image space
        name : string
            Name for image
        fpath : string
            Path to image file or directory containing the 2D image series
        mask : numpy array, optional
        x : list of numpy arrays
            Space coordinates for 2D series

        '''
        self.space = space
        self.name = name
        
        self.x, self.data, self.title, self.names = read_data(fpath, x=x, normalize=True)
                
        # I think we should check the title to see if we need to normalize
        if 'label' in self.title.lower() or 'annotation' in self.title.lower() or 'segmentation' in self.title.lower():
            print('Found an annotation image, not converting to float, and reloading with normalize false')
            self.annotation = True
            # read it again with no normalization
            self.x, self.data, self.title, self.names = read_data(fpath, x=x, normalize=False)  
        else:
            self.data = self.data.astype(float) 
            self.annotation = False
                

        self.mask = mask
        self.path = fpath
        if 'mask' in self.names:
            maskind = self.names.index('mask')
            self.mask = self.data[maskind]
            self.data = self.data[np.arange(self.data.shape[0])!=maskind]
        elif mask == True: # only initialize mask array if mask arg is True
            self.mask = np.ones_like(self.data[0])
    
    # normalize is now performed during data loading
    def _normalize(self, norm='mean', q=0.99):
        ''' Normalize image

        Parameters
        ----------
        norm : string
            Takes the values 'mean', or 'quantile'. Default is 'mean'
        q : float
            Quantile used for normalization if norm is set to 'quantile'

        Return
        ------
        numpy array
            Normalized image
        '''
        if norm == 'mean':
            return self.data / np.mean(np.abs(self.data))
        if norm == 'quantile':
            return self.data / np.quantile(self.data, q)
        else:
            warn(f'{norm} is not a valid option for the norm keyword argument.')
    
    def downsample(self, down):
        ''' Downsample image

        Parameters
        ----------
        down : list of ints
            Factor by which to downsample along each dimension 

        Returns
        -------
        x : list of numpy arrays
            Pixel locations where each element of the list identifies pixel
            locations in corresponding axis.
        data : numpy array
            image data
        mask : numpy array
            binary mask array
        '''
        x, data = downsample_image_domain(self.x, self.data, down)
        mask = self.mask
        if mask is not None:
            mask = downsample(mask,down)
        # TODO account for weights in mask when downsampling image
        return x, data, mask

    def fnames(self):
        ''' Get a list of image file names for 2D series, or a single file name for volume image.

        Returns
        -------
        fnames : list of strings
            List of image file names
        '''
        if os.path.splitext(self.path)[1] == '':
            samples_tsv = os.path.join(self.path, "samples.tsv")
            fnames = []
            with open(samples_tsv,'rt') as f:
                for count,line in enumerate(f):
                    line = line.strip()
                    key = '\t' if '\t' in line else '    '
                    if count == 0:
                        continue
                    fnames.append(os.path.splitext(re.split(key,line)[0])[0])
        else:
            fnames = [self.path]

        return fnames
        

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


def registered_domain(x,A2d):
    '''Construct a new domain that fits all rigidly aligned slices.

    Parameters
    ----------
    x : list of arrays
        list of numpy arrays containing voxel positions along each axis.
    A2d : numpy array
        Nx3x3 array of affine transformations
    
    Returns
    -------
    xr : list of arrays
        new list of numpy arrays containing voxel positions along each axis

    '''
    X = torch.stack(torch.meshgrid(x, indexing='ij'), -1)
    A2di = torch.inverse(A2d)
    points = (A2di[:, None, None, :2, :2] @ X[..., 1:, None])[..., 0]
    m0 = torch.min(points[..., 0])
    M0 = torch.max(points[..., 0])
    m1 = torch.min(points[..., 1])
    M1 = torch.max(points[..., 1])
    # construct a recon domain
    dJ = [xi[1] - xi[0] for xi in x]
    # print('dJ shape: ', [x.shape for x in dJ])
    xr0 = torch.arange(float(m0), float(M0), dJ[1], device=m0.device, dtype=m0.dtype)
    xr1 = torch.arange(float(m1), float(M1), dJ[2], device=m0.device, dtype=m0.dtype)
    xr = x[0], xr0, xr1

    return xr


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





class Transform():    
    '''
    A simple class for storing and applying transforms
    
    

    Note that the types of transforms we can support are
    
    #. Deformations stored as a displacement field loaded from a vtk file.  These should be a 1x3xrowxcolxslice array.
    #. Deformations stored as a position field in a python variable. These are a 3xrowxcolxslice array.
    #. Velocity fields stored as a python variable. These are ntx3xrowxcolxslice arrays.
    #. a 4x4 affine transform loaded from a text file.
    #. a 4x4 affine transform stored in a python variable
    #. a nslices x 3 x 3 sequence of affine transforms stored in an array. *** new and special case ***

    Note the data stores position fields or matrices.  If it is a vtk file it will store a position field.
    
    

    Raise
    -----
    Exception
        If transform is not a txt or vtk file or valid python variable.
    Exception
        if direction is not 'f' or 'b'.
    Exception
        When inputting a velocity field, if the domain is not included.
    Exception
        When inputting a matrix, if its shape is not 3x3 or 4x4.
    Exception
        When specifying a mapping, if the direction is 'b' (backward).

    '''
    def __init__(self, data, direction='f', domain=None, dtype=torch.float, device='cpu', verbose=False, **kwargs):
        if isinstance(data,str):
            if verbose: print(f'Your data is a string, attempting to load files')
            # If the data is a string, we assume it is a filename and load it
            prefix,extension = os.path.splitext(data)
            if extension == '.txt':
                '''
                data = np.genfromtxt(data,delimiter=',')                
                # note that there are nans at the end if I have commas at the end
                if np.isnan(data[0,-1]):
                    data = data[:,:data.shape[1]-1]
                    #print(data)
                '''
                # note on March 29, daniel adds the following and commented out the above
                # Here we load a matrix from a text file.  We will expect this to be a 4x4 matrix.
                data = read_matrix_data(data)
                if verbose: print(f'Found txt extension, loaded matrix data')
            elif extension == '.vtk':
                # if it is a vtk file we will assume this is a displacement field
                x,images,title,names = read_vtk_data(data)
                domain = x
                data = images
                if verbose: print(f'Found vtk extension, loaded vtk file with size {data.shape}')
            elif extension == '':
                # if there is no extension, we assume this is a directory containing a sequence of transformation matrices
                # TODO: there are a couple issues here
                # first we need to make sure we can sort the files in the right way.  right now its sorting based on the last 4 characters
                # this works for csh, but does not work in general.
                # second we may have two datasets mixed in, so we'll need a glob pattern.
                transforms_ls = sorted(os.listdir(data), key=lambda x: x.split('_matrix.txt')[0][-4:])
                data = []
                for t in transforms_ls:
                    A2d = read_matrix_data(t)
                    data.append(A2d)
                    # converting to tensor will deal with the stacking
                if verbose: print(f'Found directory, loaded a series of {len(data)} matrix files with size {data[-1].shape}')
            else:
                raise Exception(f'Only txt and vtk files supported but your transform is {data}')
        
        # convert the data to a torch tensor        
        self.data = torch.as_tensor(data,dtype=dtype,device=device)
        if domain is not None:
            domain = [torch.as_tensor(d,dtype=dtype,device=device) for d in domain]            
        self.domain = domain
        if not direction in ['f','b']:
            raise Exception(f'Direction must be \'f\' or \'b\' but it was \'{direction}\'')
        self.direction = direction
        
        # if it is a velocity field we need to integrate it
        # if it is a displacement field, then we need to add identity to it
        if self.data.ndim == 5:
            if verbose: print(f'Found 5D dataset, this is either a displacement field or a velocity field')
            if self.data.shape[0] == 1:
                # assume this is a displacement field and add identity
                # if it is a displacement field we cannot invert it, so we should throw an error if you use the wrong f,b
                raise Exception('Displacement field not supported yet')
                pass
            else:
                # assume this is a velocity field and integrate it
                if self.domain is None:
                    raise Exception('Domain is required when inputting velocity field')
                if self.direction == 'b':
                    self.data = v_to_phii(self.domain,self.data,**kwargs)
                    if verbose: print(f'Integrated inverse from velocity field')
                else:
                    self.data = v_to_phii(self.domain,-torch.flip(self.data,(0,)),**kwargs)
                    if verbose: print(f'Integrated velocity field')            
        elif self.data.ndim == 2:# if it is a matrix check size
            if verbose: print(f'Found 2D dataset, this is an affine matrix.')
            if self.data.shape[0] == 3 and self.data.shape[1]==3:
                if verbose: print(f'converting 2D to 3D with identity')
                tmp = torch.eye(4,device=device,dtype=dtype)
                tmp[1:,1:] = self.data
                self.data = tmp            
            elif not (self.data.shape[0] == 4 and self.data.shape[1]==4):
                raise Exception(f'Only 3x3 or 4x4 matrices supported now but this is {self.data.shape}')
                
            if self.direction == 'b':
                self.data = torch.inverse(self.data)
        elif self.data.ndim == 3: # if it is a series of 2d affines, we leave them as 2d.
            if verbose: print(f'Found a series of 2D affines.')
            if self.direction == 'b':
                self.data = torch.inverse(self.data)
        elif self.data.ndim == 4: # if it is a mapping
            if self.direction == 'b':
                raise Exception(f'When specifying a mapping, backwards is not supported')
                        
                
                
    def apply(self,X):
        X = torch.as_tensor(X,dtype=self.data.dtype,device=self.data.device)
        if self.data.ndim == 2:
            # then it is a matrix
            A = self.data
            return ((A[:3,:3]@X.permute(1,2,3,0)[...,None])[...,0] + A[:3,-1]).permute(3,0,1,2)
        elif self.data.ndim == 3:
            # if it is 3D then it is a stack of 2D 3x3 affine matrices
            # this will only work if the input data is the right shape
            # ideally, I should attach a domain to it.
            A2d = self.data
            X[1:] = ((A2d[:,None,None,:2,:2]@ (X[1:].permute(1,2,3,0)[...,None]))[...,0] + A2d[:,None,None,:2,-1]).permute(3,0,1,2)
            return X
        elif self.data.ndim == 4:
            # then it is a displacement field, we need interp
            # recall all components are stored on the first axis,
            # but for sampling they need to be on the last axis
            ID = torch.stack(torch.meshgrid(self.domain,indexing='ij'))
            # print(f'ID shape {ID.shape}')
            # print(f'X shape {X.shape}')
            # print(f'data shape {self.data.shape}')

            return interp(self.domain,(self.data-ID),X) + X
            
                    
    def __repr__(self):
        return f'Transform with data size {self.data.shape}, direction {self.direction}, and domain {type(self.domain)}'
    def __call__(self,X):
        return self.apply(X)
            
# now wrap this into a function
def compose_sequence(transforms,Xin,direction='f',verbose=False):
    ''' Compose a set of transformations, to produce a single position field, 
    suitable for use with :func:`emlddmm.apply_transform_float` for example.
    
    Parameters
    ----------
    transforms : 
        Several types of inputs are supported.

        #. A list of transforms class.
        #. A list of filenames (single direction in argument)
        #. A list of a list of 2 tuples that specify direction (f,b)
        #. An output directory
        
    Xin : 3 x slice x row x col array
        The points we want to transform (e.g. sample points in atlas).  Also supports input as a list of voxel locations,
        along each axis which will be reshaped as above using meshgrid.
    direction : char
        Can be 'f' for foward or 'b' for bakward.  f is default which maps points from atlas to target, 
        or images from target to atlas.
        
    Returns
    -------
    Xout :  3 x slicex rowxcol array
        Points from Xin that have had a sequence of transformations applied to them.
    
        
        
    Note
    ----
    Note, if the input is a string, we assume it is an output directory and get A and V. In this case we use the direction argument.
    If the input is a tuple of length 2, we assume it is an output directory and a direction
    
    Otherwise, the input must be a list.  It can be a list of strings, or transforms, or string-direction tuples.  
    
    We check that it is an instace of a list, so it should not be a tuple.
    
    

    Raises
    ------
    Exception
        Transforms must be either output directory,
        or list of objects, or list of filenames,
        or list of tuples storing filename/direction.
    
    
    
    Todo
    ----
    #. use os path join
    #. support direction as a list, right now direction only is used for a single direction

    '''
    
    #print(f'starting to compose sequence with transforms {transforms}')    
    
    
    # check special case for a list of length 1 but the input is a directory
    if (type(transforms) == list and len(transforms)==1 
        and type(transforms[0]) == str and os.path.splitext(transforms[0])[1]==''):
        transforms = transforms[0]
    # check special case for a list of length one but the input is a tuple
    if (type(transforms) == list and len(transforms)==1 
        and type(transforms[0]) == tuple and type(transforms[0][0]) == str 
        and os.path.splitext(transforms[0][0])[1]=='' and transforms[0][1].lower() in ['b','f']):
        direction = transforms[0][1]
        transforms = transforms[0][0] # note variable is redefined                
    
    
    if type(transforms) == str or ( type(transforms) == list and len(transforms)==1 and type(transforms[0]) == str ):
        if type(transforms) == list: transforms = transforms[0]            
        # assume output directory
        # print('printing transforms input')
        # print(transforms)
        if direction == 'b':
            # backward, first affine then deform
            transforms = [Transform(join(transforms,'transforms','A.txt'),direction=direction),
                          Transform(join(transforms,'transforms','velocity.vtk'),direction=direction)]
        elif direction == 'f':
            # forward, first deform then affine
            transforms = [Transform(join(transforms,'transforms','velocity.vtk'),direction=direction),
                          Transform(join(transforms,'transforms','A.txt'),direction=direction)]    
        #print('printing modified transforms')
        #print(transforms)    
    elif type(transforms) == list:
        # there is an issue here:
        # when I call this from outside, the type is emlddmm.Transform, not Transform. The test fails
        # print(type(transforms[0]))
        # print(isinstance(transforms[0],Transform))
        #if type(transforms[0]) == Transform:
        if 'Transform' in str(type(transforms[0])): 
            # this approach may fix the issue but I don't like it
            # I am having trouble reproducing this error on simpler examples
            # don't do anything here
            pass
        elif type(transforms[0]) == str:
            # list of strings
            transforms = [Transform(t,direction=direction) for t in transforms]
        elif type(transforms[0]) == tuple:
            transforms = [Transform(t[0],direction=t[1]) for t in transforms]
        else:
            raise Exception('Transforms must be either output directory, \
        or list of objects, or list of filenames, \
        or list of tuples storing filename/direction')
    else:
        raise Exception('Transforms must be either output directory, \
        or list of objects, or list of filenames, \
        or list of tuples storing filename/direction')
    # oct 2023, support input as a list of zyx coords
    if isinstance(Xin,list):
        Xin = [torch.as_tensor(x,device=transforms[0].data.device,dtype=transforms[0].data.dtype) for x in Xin]
        Xin = torch.stack(torch.meshgrid(*Xin,indexing='ij'))
    Xin = torch.as_tensor(Xin,device=transforms[0].data.device,dtype=transforms[0].data.dtype)    
    Xout = torch.clone(Xin)
    for t in transforms:
        Xout = t(Xout)
    return Xout
    
def apply_transform_float(x,I,Xout,**kwargs):
    '''Apply transform to image
    Image points stored in x, data stored in I
    transform stored in Xout
    
    There is an issue with numpy integer arrays, I'll have two functions
    '''
    if type(I) == np.array:
        isnumpy = True
    else:
        isnumpy = False
    
    AphiI = interp(x,torch.as_tensor(I,dtype=Xout.dtype,device=Xout.device),Xout,**kwargs).cpu()
    if isnumpy:
        AphiI = AphiI.numpy()
    return AphiI

def apply_transform_int(x,I,Xout,double=True,**kwargs):
    '''Apply transform to image
    Image points stored in x, data stored in I
    transform stored in Xout
    
    There is an issue with numpy integer arrays, I'll have two functions
    
    Note that we often require double precision when converting to floats and back

    Raises
    ------
    Exception
        If mode is not 'nearest' for ints.
    '''
    if type(I) == np.array:
        isnumpy = True
    else:
        isnumpy = False
    Itype = I.dtype
    if 'mode' not in kwargs:
        kwargs['mode'] = 'nearest'
    else:
        if kwargs['mode'] != 'nearest':
            raise Exception('mode must be nearest for ints')
    try:
        device = Xout.device
    except:
        device = 'cpu'
    
    # for int, I need to convert to float for interpolation    
    if not double:
        AphiI = interp(x,torch.as_tensor(I.astype(float),dtype=Xout.dtype,device=device),Xout,**kwargs).cpu()
    else:
        AphiI = interp(x,torch.as_tensor(I.astype(np.float64),dtype=torch.float64,device=device),torch.as_tensor(Xout,dtype=torch.float64),**kwargs).cpu()
        
    if isnumpy:
        AphiI = AphiI.numpy().astype(Itype)
    else:
        AphiI = AphiI.int()
    return AphiI
    
            

        
def rigid2D(xI,I,xJ,J,**kwargs):
    ''' 
    Rigid transformation between 2D slices.
    
    
    '''
    pass
        
    
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
        
def pad(xI,I,n,**kwargs):
    '''
    Pad an image and its domain.    
    
    Perhaps include here
    
    Parameters
    ----------
    xI : list of arrays
        Location of pixels in I
    I : array
        Image
    n : list of ints or list of pairs of ints
        Pad on front and back
        
    '''
    raise Exception('Not Implemented')
    if isinstance(I,torch.Tensor):
        raise Exception('only implemented for numpy')
    
    pass
    
def map_image(emlddmm_path, root_dir, from_space_name, to_space_name,
              input_image_fname, output_image_directory=None,
              from_slice_name=None, to_slice_name=None,use_detjac=False,
              verbose=False,**kwargs):
    '''
    This function will map imaging data from one space to another. 
        
    
    
    There are four cases:
    
    #. 3D to 3D mapping: A single displacement field is used to map data
    
    #. 3D to 2D mapping: A single displacement field is used to map data, a slice filename is needed in addition to a space
    
    #. 2D to 2D mapping: A single matrix is used to map data.
    
    #. 2D to 3D mapping: Currently not supported. Ideally this will output data, and weights for a single slice, so it can be averaged with other slices.
    
    Warning
    -------
    This function was built for a particular use case at Cold Spring Harbor, and is generally not used.  
    It may be removed in the future.
    
    
    Parameters
    ----------
    emlddmm_path : str
        Path to the emlddmm python library, used for io
    root_dir : str
        The root directory of the output structure
    from_space_name : str
        The name of the space we are mapping data from
    to_space_name : str
        The name of the space we are mapping data to
    input_image_fname : str
        Filename of the input image to be transformed
    output_image_fname : str
        Filename of the output image after transformation. If None (default), it will be returned as a python variable but not written to disk.
    from_slice_name : str
        When transforming slice based image data only, we also need to know the filename of the slice the data came from.
    to_slice_name : str
        When transforming slice based image data only, we also need to know the filename of the slice the data came from.
    use_detjac : bool
        If the image represents a density, it should be transformed and multiplied by the Jacobian of the transformation
    
    Keyword Arguments
    -----------------
    **kwargs : dict
        Arguments passed to torch interpolation (grid_resample), e.g. padding_mode,
    
    Returns
    -------
    phiI : array
        Transformed image
    
    Raises
    ------
    Exception
        If use_detjac set to True for 3D to 2D mapping. Detjac not currently supported for 3D to 2D.
    Exception
        2D to 3D not implemented yet, may not get implemented.
    Exception
        Jacobian not implemented yet for 2D slices.

    Warns
    -----
    DetJac is ignored if mapping is 2D to 2D.

    '''
    
    from os.path import split,join,splitext
    from glob import glob     
    from warnings import warn
    warn(f'This function is experimental')
    
    # first load the image to be transformed
    xI,I,title,names = read_data(input_image_fname)    
    
    # find the transformation we need, in each case I will load the appropriate data, and return a phi
    transform_dir = join(root_dir,to_space_name,from_space_name + '_to_' + to_space_name,'transforms')
    if from_slice_name is None and to_slice_name is None:
        # This is case 1, 3D to 3D
        files = glob(join(transform_dir,to_space_name +'_to_' + from_space_name + '_displacement.vtk'))
        xD,D,title,names = read_data(files[0])        
        D = D[0]
        phi = D + np.stack(np.meshgrid(*xD,indexing='ij'))
        if use_detjac:
            detjac = np.linalg.det(np.stack(np.gradient(phi,xD[0][1]-xD[0][0], xD[1][1]-xD[1][0], xD[2][1]-xD[2][0], axis=(1,2,3))).permute(2,3,4,0,1))
    elif from_slice_name is None and to_slice_name is not None:
        # This is case 2, 3D to 2D
        # we have a displacement field  
        to_slice_name_ = splitext(split(to_slice_name)[-1])[0]
        files = glob(join(transform_dir,to_space_name + '_' + to_slice_name_ + '_to_' + from_space_name + '_displacement.vtk'))
        xD,D,title,names = read_data(files[0])        
        D = D[0]
        phi = D + np.stack(np.meshgrid(*xD,indexing='ij'))        
        if use_detjac:
            raise Exception('Detjac not currently supported for 3D to 2D')
    elif from_slice_name is not None and to_slice_name is not None:        
        # This is case 3, 2D to 2D, we have an xyz matrix        
        to_slice_name_ = splitext(split(to_slice_name)[-1])[0]
        from_slice_name_ = splitext(split(from_slice_name)[-1])[0]
        files = glob(join(transform_dir,to_space_name + '_' + to_slice_name_ + '_to_' + from_space_name + '_' + from_slice_name_ + '_matrix.txt'))
        if use_detjac:
            warn('DetJac is 1 for 2D to 2D, so it is ignored')
        data = read_matrix_data(files[0])
        if verbose:
            print('matrix data:')
            print(data)
        # we need to convert this to a displacement field
        # to do this we need some image to get sample points
        # we use the "to" slice image name
        image_dir = join(root_dir,to_space_name,from_space_name + '_to_' + to_space_name,'images')
        
        testfile = glob(join(image_dir,'*' + splitext(from_slice_name)[0] + '*_to_*' + splitext(to_slice_name)[0] + '*.vtk'))
        xS,J,_,_ = read_data(testfile[0])
        Xs = np.stack(np.meshgrid(*xS,indexing='ij'))
        # now we can build a phi
        A = data
        phi = ((A[:2,:2]@Xs[1:].transpose(1,2,3,0)[...,None])[...,0] + A[:2,-1]).transpose(-1,0,1,2)        
        phi = np.concatenate((Xs[0][None],phi) )         
        xD = xS
        
    elif from_slice_name is not None and to_slice_name is None:
        # this is 2D to 3D we have a displacement field
        raise Exception('2D to 3D not implemented yet, may not get implemented')
    
    # apply the transform to the image
    # if desired calculate jacobian
    if use_detjac:
        raise Exception('Jacobian not implemented yet for 2D slices')
        
    # todo, implement when I is int
    phiI = interp(xI,I,phi,**kwargs)    
    
    if output_image_directory is not None:
        # we need to go back to the four cases here
        if from_slice_name is None and to_slice_name is None:
            # case 1, 3d to 3d
            outfname = join(output_image_directory,f'{from_space_name}_{splitext(split(input_image_fname)[-1])[0]}_to_{to_space_name}.vtk')
            outtitle = split(splitext(outfname)[0])[-1]
        elif from_slice_name is None and to_slice_name is not None:
            # case 2, 3d to 2d
            outfname = join(output_image_directory,f'{from_space_name}_{splitext(split(input_image_fname)[-1])[0]}_to_{to_space_name}_{split(splitext(to_slice_name)[0])[-1]}.vtk')
            outtitle = split(splitext(outfname)[0])[-1]
        elif from_slice_name is not None and to_slice_name is not None:
            # case 3, 2d to 2d
            outfname = join(output_image_directory,f'{from_space_name}_{split(splitext(from_slice_name)[0])[-1]}_{splitext(split(input_image_fname)[-1])[0]}_to_{to_space_name}_{split(splitext(to_slice_name)[0])[-1]}.vtk')
            outtitle = split(splitext(outfname)[0])[-1]            
        else:
            # case 4, 2d to 3d
            raise Exception('2D to 3D not supported')
            
        # a hack for slice thickness
        if len(xD[0]) == 1:
            xD[0] = np.array([xD[0][0],xD[0][0]+20.0])    
        if verbose:
            print(f'writing file with name {outfname} and title {outtitle}')
        write_data(outfname,xD,phiI,outtitle)
        
        
    return xD,phiI
    
    
        
def map_points(emlddmm_path, root_dir, from_space_name, to_space_name,
              input_points_fname, output_points_directory=None,
              from_slice_name=None, to_slice_name=None,use_detjac=False,
              verbose=False,**kwargs):
    '''    
    For points we need to get the transforms in the opposite folder to images.
    
    This function will map imaging data from one space to another. 
    There are four cases:
    
    #. 3D to 3D mapping: A single displacement field is used to map data
    
    #. 3D to 2D mapping: Currently not supported.
    
    #. 2D to 2D mapping: A single matrix is used to map data.
    
    #. 2D to 3D mapping: A single displacement field is used to map data, a slice filename is needed in addition to a space
    
    
    Warning
    -------
    This function was built for a particular use case at Cold Spring Harbor, and is generally not used.  
    It may be removed in the future.
    
    
    Parameters
    ----------
    emlddmm_path : str
        Path to the emlddmm python library, used for io
    root_dir : str
        The root directory of the output structure
    from_space_name : str
        The name of the space we are mapping data from
    to_space_name : str
        The name of the space we are mapping data to
    input_points_fname : str
        Filename of the input image to be transformed
    output_directory_fname : str
        Filename of the output image after transformation. If None (default), it will be returned as a python variable but not written to disk.
    from_slice_name : str
        When transforming slice based image data only, we also need to know the filename of the slice the data came from.
    to_slice_name : str
        When transforming slice based image data only, we also need to know the filename of the slice the data came from.
    use_detjac : bool
        If the image represents a density, it should be transformed and multiplied by the Jacobian of the transformation
    **kwargs : dict
        Arguments passed to torch interpolation (grid_resample), e.g. padding_mode,
    
    Returns
    -------
    phiP : array
        Transformed points
    connectivity : list of lists
        Same connectivity entries as loaded data
    connectivity_type : str
        Same connectivity type as loaded data
    
    Raises
    ------
    Exception
        3D to 2D mapping is not implemented for points.
    Exception
        If use_detjac set to True for 2D to 3D mapping. Detjac not currently supported for 2D to 3D
    Exception
        If use_detjac set to True. Jacobian is not implemented yet.
    Exception
        If attempting to map points from 3D to 2D. 
    Warns
    -----
    DetJac is ignored if mapping is 2D to 2D.    
    '''
    
    from os.path import split,join,splitext
    from glob import glob    
    from warnings import warn
    warn(f'This function is experimental')
    
    # first load the points to be transformed
    points,connectivity,connectivity_type,name = read_vtk_polydata(input_points_fname)    
    
    # find the transformation we need, in each case I will load the appropriate data, and return a phi
    # note that for points we use the inverse to images
    # transform_dir = join(root_dir,to_space_name,from_space_name + '_to_' + to_space_name,'transforms')
    # above was for images
    transform_dir = join(root_dir,from_space_name,to_space_name + '_to_' + from_space_name,'transforms')
    if from_slice_name is None and to_slice_name is None:
        # This is case 1, 3D to 3D
        files = glob(join(transform_dir,from_space_name +'_to_' + to_space_name + '_displacement.vtk'))
        xD,D,title,names = read_data(files[0])        
        D = D[0]
        phi = D + np.stack(np.meshgrid(*xD,indexing='ij'))
        if use_detjac:
            detjac = np.linalg.det(np.stack(np.gradient(phi,xD[0][1]-xD[0][0], xD[1][1]-xD[1][0], xD[2][1]-xD[2][0], axis=(1,2,3))).permute(2,3,4,0,1))
    elif from_slice_name is None and to_slice_name is not None:
        # This is case 2, 3D to 2D
        # this one is not implemented
        raise Exception('3D to 2D not implemented for points')
        
    elif from_slice_name is not None and to_slice_name is not None:        
        # This is case 3, 2D to 2D, we have an xyz matrix        
        to_slice_name_ = splitext(split(to_slice_name)[-1])[0]
        from_slice_name_ = splitext(split(from_slice_name)[-1])[0]
        file_search = join(transform_dir,from_space_name + '_' + from_slice_name_ + '_to_' + to_space_name + '_' + to_slice_name_ + '_matrix.txt')
        files = glob(file_search)
        if verbose:
            print(file_search)
            print(files)
        if use_detjac:
            warn('DetJac is 1 for 2D to 2D, so it is ignored')
            
        data = read_matrix_data(files[0])
        if verbose:
            print('matrix data:')
            print(data)
        # we do not need to convert this to a displacement field
        phi = None
        
    elif from_slice_name is not None and to_slice_name is None:
        # this is 2D to 3D we have a displacement field
        from_slice_name_ = splitext(split(from_slice_name)[-1])[0]
        file_search = join(transform_dir,from_space_name + '_' + from_slice_name_ + '_to_' + to_space_name + '_displacement.vtk')
        if verbose:
            print(file_search)
        files = glob(file_search)
        xD,D,title,names = read_data(files[0])        
        D = D[0]
        phi = D + np.stack(np.meshgrid(*xD,indexing='ij'))        
        if use_detjac:
            raise Exception('Detjac not currently supported for 2D to 3D')
            
    
    # apply the transform to the image
    # if desired calculate jacobian
    if use_detjac:
        raise Exception('Jacobian not implemented yet')
            
    if phi is not None:
        # points is size N x 3
        # we want 3 x 1 x 1 x N
        points_ = points.transpose()[:,None,None,:]                
        phiP = interp(xD,phi,points_,**kwargs)[:,0,0,:].T
        
    else:
        phiP = data[:2,:2]@points[...,1:] + data[:2,-1]
        phiP = np.concatenate((points[...,0][...,None],phiP),-1)
        
    
    if output_points_directory is not None:        
        # we need to go back to the four cases here
        if from_slice_name is None and to_slice_name is None:
            # case 1, 3d to 3d
            outfname = join(output_points_directory,f'{from_space_name}_{splitext(split(input_points_fname)[-1])[0]}_to_{to_space_name}.vtk')
            
        elif from_slice_name is None and to_slice_name is not None:
            # case 2, 3d to 2d
            raise Exception('3D to 2D not supported')
            
        elif from_slice_name is not None and to_slice_name is not None:
            # case 3, 2d to 2d
            outfname = join(output_points_directory,f'{from_space_name}_{split(splitext(from_slice_name)[0])[-1]}_{splitext(split(input_points_fname)[-1])[0]}_to_{to_space_name}_{split(splitext(to_slice_name)[0])[-1]}.vtk')
                    
        else:
            # case 4, 2d to 3d
            outfname = join(output_points_directory,f'{from_space_name}_{split(splitext(from_slice_name)[0])[-1]}_{splitext(split(input_points_fname)[-1])[0]}_to_{to_space_name}.vtk')
            
            
         
        if verbose:
            print(f'writing file with name {outfname}')
        write_vtk_polydata(outfname,name,phiP,connectivity=connectivity,connectivity_type=connectivity_type)
        
        
    return phiP, connectivity, connectivity_type, name
    



def convert_points_from_json(points, d_high, n_high=None, sidecar=None, z=None, verbose=False):
    '''
    We load points from a json produced by Samik at cold spring harbor.
    
    These are indexed to pixels in a high res image, rather than any physical units.
    
    To convert to proper points in 3D for transforming, we need information about pixel size and origin.
    
    Pixel size of the high res image is a required input.
    
    If we have a json sidecar file that was prepared for the registration dataset, we can get all the info from this.
    
    If not, we get it elsewhere.
    
    Origin information can be determined from knowing the number of pixels in the high res image. 
    
    Z coordinate information is not required if we are only applying 2D transforms, 
    but for 3D it will have to be input manually if we do not have a sidecar file.
    
    
    Parameters
    ----------
    points : str or numpy array
        either a geojson filename, or a Nx2 numpy array with coordinates loaded from such a file.
    d_high : float
        pixel size of high resolution image where cells were detected
    n_high : str or numpy array
        WIDTH x HEIGHT of high res image.  Or the filename of the high res image.
    sidecar : str
        Filename of sidecar file to get z and origin info
    z : float
        z coordinate
    
    Returns
    -------
    q : numpy array
        A Nx3 array of points in physical units using our coordinate system convention (origin in center).
    
    Notes
    -----
    If we are applying 3D transforms, we need a z coordinate.  This can be determined either by specifying it
    or by using a sidecar file.  If we have neither, it ill be set to 0.
    
    TODO
    ----
    Consider setting z to nan instead of zero.
    
    '''
    
    # deal with the input points
    if isinstance(points,str):
        if verbose: print(f'points was a string, loading from json file')
        # If we specified a string, load the points
        with open(points,'rt') as f:
            data = json.load(f)
        points = data['features'][0]['geometry']['coordinates']
        points = [p for p in points if p]
        points = np.array(points)
    else:
        if verbose: print(f'Points was not a string, using as is')
    
    # start processing points
    q = np.array(points[:,::-1])
    # flip the sign of the x0 component
    q[:,0] = q[:,0]*(-1)
    # multiply by the pixel size
    q = q * d_high
    if verbose: print(f'flipped the first and second column, multiplied the first column of the result by -1')
    
    # now check if there is a sidecar
    if sidecar is not None:
        if verbose: print(f'sidecar specified, loading origin and z information')
        with open(sidecar,'rt') as f:
            data = json.load(f) 
        # I don't think this is quite right
        # this would be right if the downsampling factor was 1 and the voxel centers matched
        # if the downsampling factor was 2, we'd have to move a quarter (big) voxel to the left
        # | o | _ |
        # if the downsampling factor was 4 we'd have to move 3 eights
        # | o | _ | _ | _ |
        # what's the pattern? we move half a big voxel to the left, then half a small voxel to the right
        # 
        #downsampling_factor = (np.diag(np.array(data['SpaceDirections'][1:]))/(d_high))[0]
        d_low = np.diag(np.array(data['SpaceDirections'][1:]))
        if verbose: print(f'd low {d_low}')
        q[:,0] += data['SpaceOrigin'][1] - d_low[1]/2.0 + d_high/2.0
        q[:,1] += data['SpaceOrigin'][0] - d_low[0]/2.0 + d_high/2.0
        q = np.concatenate((np.zeros_like(q[:,0])[:,None]+data['SpaceOrigin'][-1],q   ), -1)
        
    
    else:
        if verbose: print(f'no sidecar specified, loading origin from n_high')
        # we have to get origin and z information elsewhere
        if n_high is None:
            raise Exception(f'If not specifying sidecar, you must specify n_high')
        elif isinstance(n_high,str):
            if verbose: print(f'loading n_high from jp2 file')
            # this sould be a jp2 file
            image = PIL.Image.open(n_high)
            n_high = np.array(image.size) # this should be xy
            image.close()
        # in this case, the point 0,0 (first pixel) should have a coordinate -n/2            
        q[:,0] -= (n_high[1]-1)/2*d_high
        q[:,1] -= (n_high[0]-1)/2*d_high
        if verbose: print(f'added origin')
            
        if z is None:
            # if we have no z information we will just pad zeros
            if verbose: print(f'no z coordinate, appending 0')
            q = np.concatenate((np.zeros_like(q[:,0])[:,None],q   ), -1)
        else:
            if verbose: print(f'appending input z coordinate')
            q = np.concatenate((np.ones_like(q[:,0])[:,None]*z,q   ), -1)
            
        
    return q
        

def apply_transform_from_file_to_points(q,tform_file):
    '''
    To transform points from spacei to spacej (example from fluoro to nissl_registerd)
    We look for the output folder called
    outputs/spacei/spacej_to_spacei/transforms (example outputs/fluoro/nissl_registered_to_fluoro/transforms)
    Note "spacej to spacei" is not a typo 
    (even though it looks backwards, point data uses inverse of transform as compared to images).
    In the transforms folder, there are transforms of the form
    "spacei to spacej".  
    If applying transforms to slice data, you will have to find the appropriate slice.
    
    Parameters
    ----------
    q : numpy array
        A Nx3 numpy array of coordinates in slice,row,col order
    tform_file : str
        A string pointing to a transform file
        
    Returns
    -----
    Tq : numpy array
        The transformed set of points.
    '''
    if tform_file.endswith('.txt'):
        # this is a matrix
        # we do matrix multiplication to the xy components, and leave the z component unchanged
        R = read_matrix_data(tform_file)                
        Tq = np.copy(q)
        Tq[:,1:] = (R[:2,:2]@q[:,1:].T).T + R[:2,-1]
        return Tq
    elif tform_file.endswith('displacement.vtk'):
        # this is a vtk displacement field
        x,d,title,names = read_data(tform_file)
        if d.ndim == 5:
            d = d[0]
        identity = np.stack(np.meshgrid(*x,indexing='ij'))
        phi = d + identity # add identity to convert "displacement" to "position"
        # evaluate the position field at the location of these points.
        Tq = interpn(x,phi.transpose(1,2,3,0),q,bounds_error=False,fill_value=None,method='nearest')
        return Tq
    


def orientation_to_RAS(orientation,verbose=False):
    ''' Compute a linear transform from a given orientation to RAS.
    
    Orientations are specified using 3 letters, by selecting one of each 
    pair for each image axis: R/L, A/P, S/I
    
    Parameters
    ----------
    orientation : 3-tuple
        orientation can be any iterable with 3 components. 
        Each component should be one of R/L, A/P, S/I. There should be no duplicates
    
    Returns
    -------
    Ao : 3x3 numpy array
        A linear transformation to transform your image to RAS        
    
    '''
    orientation_ = [o for o in orientation]
    Ao = np.eye(3)
    # first step, flip if necessary, so we only use symbols R A and S
    for i in range(3):
        if orientation_[i] == 'L':
            Ao[i,i] *= -1
            orientation_[i] = 'R'
        if orientation_[i] == 'P':
            Ao[i,i] *= -1
            orientation_[i] = 'A'
        if orientation_[i] == 'I':
            Ao[i,i] *= -1
            orientation_[i] = 'S'

    # now we need to handle permutations
    # there are 6 cases
    if orientation_ == ['R','A','S']:
        pass
    elif orientation_ == ['R','S','A']:
        # flip the last two axes to change to RAS
        Ao = np.eye(3)[[0,2,1]]@Ao # elementary matrix is identity with rows flipped
        orientation_ = [orientation_[0],orientation_[2],orientation_[1]]
    elif orientation_ == ['A','R','S']:
        # flip the first two axes
        Ao = np.eye(3)[[1,0,2]]@Ao 
        orientation_ = [orientation_[1],orientation_[0],orientation_[2]]
    elif orientation_ == ['A','S','R']:
        # we need a 2,0,1 permutation
        Ao = np.eye(3)[[0,2,1]]@np.eye(3)[[2,1,0]]@Ao 
        orientation_ = [orientation_[2],orientation_[0],orientation_[1]]
    elif orientation_ == ['S','R','A']:
        # flip the first two, then the second two
        Ao = np.eye(3)[[0,2,1]]@np.eye(3)[[1,0,2]]@Ao 
        orientation_ = [orientation_[1],orientation_[2],orientation_[0]]        
    elif orientation_ == ['S','A','R']:
        # flip the first and last
        Ao = np.eye(3)[[2,1,0]]@Ao         
        orientation_ = [orientation_[2],orientation_[1],orientation_[0]]    
    else:
        raise Exception('Something is wrong with your orientation')
    return Ao

def orientation_to_orientation(orientation0,orientation1,verbose=False):
    ''' Compute a linear transform from one given orientation to another.
    
    Orientations are specified using 3 letters, by selecting one of each 
    pair for each image axis: R/L, A/P, S/I
    
    This is done by computing transforms to and from RAS.
    
    Parameters
    ----------
    orientation : 3-tuple
        orientation can be any iterable with 3 components. 
        Each component should be one of R/L, A/P, S/I. There should be no duplicates
    
    Returns
    -------
    Ao : 3x3 numpy array
        A linear transformation to transform your image from orientation0 to orientation1        
    
    '''
    Ao = np.linalg.inv(orientation_to_RAS(orientation1,verbose))@orientation_to_RAS(orientation0,verbose)
    return Ao    
        
def affine_from_figure(x,shift=1000.0,angle=5.0):
    ''' 
    Build small affine transforms by looking at figures generated in draw.
    
    Parameters
    ----------
    x : str
        Two letters. "t","m","b" for top row middle row bottom row.
    
        Then 'w','e','n','s' for left right up down
    
        Or 'r','l' for turn right turn left
    
        or 'id' for identity.
    shift : float
        How far to shift.
    angle : float
        How far to rotate (in degrees)
        
    Returns
    -------
    A0 : numpy array
        4x4 numpy array affine transform matrix
      
    
    '''
    x = x.lower()
    
    theta = angle*np.pi/180
    R = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])    
    A0 = np.eye(4)
    if x == 'ts' or x == 'be':
        A0[1,-1] = shift
    elif x == 'tn' or x == 'bw':
        A0[1,-1] = -shift
    elif x == 'te' or x == 'me':
        A0[2,-1] = shift
    elif x == 'tw' or x == 'mw':
        A0[2,-1] = -shift
    elif x == 'mn' or x == 'bn':
        A0[0,-1] = -shift
    elif x == 'ms' or x == 'bs':
        A0[0,-1] = shift        
    elif x == 'bl':        
        A0[((0,0,1,1),(0,1,0,1))]  = R.ravel()
    elif x == 'br':
        A0[((0,0,1,1),(0,1,0,1))]  = R.T.ravel()
    elif x == 'ml':        
        A0[((0,0,2,2),(0,2,0,2))]  = R.ravel()    
    elif x == 'mr':        
        A0[((0,0,2,2),(0,2,0,2))]  = R.T.ravel()
    elif x == 'tl':
        A0[((1,1,2,2),(1,2,1,2))]  = R.ravel()    
    elif x == 'tr':
        A0[((1,1,2,2),(1,2,1,2))]  = R.T.ravel()    
    elif x == 'id':
        pass
    else:
        raise Exception('input string not supported.')
    return A0



def compute_atlas_from_slices(J,W,ooop,niter=10,I=None,draw=False):
    ''' 
    Construct an atlas image by averaging between slices.
    
    This uses an MM (also could be interpreted as EM) algorithm for 
    converting a weighted least squares problem to an ordinary least squares problem.
    The latter can be solved as a stationary problem, and updated iteratively.
    
    Parameters
    ----------
    J : array
        an C x slice x row x col image array.
    W : array
        An slice x row x col array of weights
    ooop : array
        a 1D array with "slice" elements.  This is a frequency domain
        one over operator for doing smoothing.
    niter : int
        Number of iterations to update in MM algorithm. (default to 10)
    I : array
        An C x slice x row x col image array representing an initial guess.
    draw : int
        Draw the image every draw iterations.  Do not draw if 0 (default).
    
    Note
    ----
    This code uses numpy, not torch, since it is not gradient based.
        
    Todo
    ----
    Better boundary conditions
    '''
    if draw:
        fig = plt.figure()
    if I is None:
        I = np.zeros_like(J)

    for it in range(niter):
        I = np.fft.ifft(np.fft.fft(W*J + (1-W)*I,axis=1)*ooop[None,:,None,None],axis=1).real
        # question, am I missing something? like divide by the weight?
        # like, what if the weight is 0.5 everywhere? This will not give the right answer
        if draw and ( not it%draw or it == niter-1):
            draw(I,xJ,fig=fig,interpolation='none')
            fig.suptitle(f'it {it}')
            fig.canvas.draw()
    return I
    
    
    
def atlas_free_reconstruction(**kwargs):
    ''' Atlas free slice alignment
    
    Uses an MM algorithm to align slices.  Minimizes a Sobolev norm over rigid transformations of each slice.
    
    All arguments are keword arguments
    
    Keword Arguments
    ----------------
    xJ : list of arrays
        Lists of pixel locations in slice row col axes
    J : array
        A C x slice x row x col set of 2D image.
    W : array
        A slice x row x col set of weights of 2D images.  1 for good quality, to 0 for bad quality or out of bounds.
    a : float
        length scale (multiplied by laplacian) for smootheness averaging between slices. Defaults to twice the slice separation.
    p : float
        Power of laplacian in somothing. Defaults to 2.    
    draw : int
        Draw figure (default false)
    n_steps : int      
        Number of iterations of MM algorithm.
    **kwargs : dict
        All other keword arguments are passed along to slice matching in the emlddmm algorithm.
        
    Returns
    -------
    out : dictionary
        All outputs from emlddmm algorithm, plus the new image I, and reconstructed image Jr
    out['I'] : numpy array
        A C x slice x row x col set of 2D images (same size as J), averaged between slices.
    out['Jr'] : numpy array
        A C x slice x row x col set of 2D images (same size as J)
        
    Todo
    ----
    Think about how to do this with some slices fixed.
    Think about normalizing slice to slice contrast.
        
    '''
    if 'J' not in kwargs:
        raise Exception('J is a required keyword argument')
    J = np.array(kwargs.pop('J'))
    
    if 'xJ' not in kwargs:
        raise Exception('xJ is a required keyword argument')
    
    xJ = [np.array(x) for x in kwargs.pop('xJ')]
    
    dJ = np.array([x[1] - x[0] for x in xJ])
    nJ = np.array(J.shape[1:])
    if 'W' not in kwargs:
        W = np.ones_like(J[0])
    else:
        W = kwargs.pop('W')
        
    if 'n_steps' not in kwargs:
        n_steps = 10
    else:
        n_steps = kwargs.pop('n_steps')
    # create smoothing operators
    # define an operator L, it can be a gaussian derivative or whatever
    #a = dJ[0]*2
    if 'a' not in kwargs:
        a = dJ[0]*3
    else:            
        a = float(kwargs.pop('a'))    
    
    #p = 2.0
    if 'p' not in kwargs:
        p = 2.0
    else:
        p = float(kwargs.pop('p'))

    fJ = [np.arange(n)/n/d for n,d in zip(nJ,dJ)]
    # note this operator must have an identity in it to be a solution to our problem.
    # the scale is then set entirely by a
    # note it shouldn't really be identity, it should be sigma^2M.
    # but in our equations, we can just multiply everything through by 1/2/sigmaM**2
    # (1 + a^2 \Delta )^(2p) (here p = 2, \Delta is the discrete Laplacian) 
    
    if 'lddmm_operator' in kwargs:
        lddmm_operator = kwargs['lddmm_operator']
    else:
        lddmm_operator = False
    if lddmm_operator:
        op = 1.0 + ( (0 + a**2/dJ[0]**2*(1.0 - np.cos(fJ[0]*dJ[0]*np.pi*2) )) )**(2.0*p) # note there must be a 1+ here
        #op = 1.0 + ( 10*(1.0 + a**2/dJ[0]**2*(1.0 - np.cos(fJ[0]*dJ[0]*np.pi*2) )) )**(2.0*p) # note there must be a 1+ here
        
        ooop = 1.0/op
        # normalize
        ooop = ooop/ooop[0] 

    else:
        # we can use a different kernel in the space domain
        # here is the objective we solve
        # estimate R (rigid transforms) and I (atlas image)
        # |I|^2_{highpass} + |I - R J|^2_{L2}
        # where |I|^2_{highpass} = \int   |L I|^2 dx and L is a highpass operator
        # in the past I used power of laplacian for L
        # we could define L such that it's kernel is a power law (or a guassian or something without ripples)
        # 
        
        # let's try this power law
        op = 1.0 / (xJ[0] - xJ[0][0])
        # jan 11, 2024, it does not decay fast enough, so try below
        op = 1.0 / (xJ[0] - xJ[0][0])**2
        op = 1.0 / np.abs((xJ[0] - xJ[0][0]))**1.5/np.sign((xJ[0] - xJ[0][0]))
        op[0] = 0
        op = np.fft.fft(op).real    # this will build in the necessary symmetry
        # recall this is the low pass
        op = 1/(op + 1*0) # now we have the highpass
        op = op / op[0]
        ooop = 1.0/op
    
    
    
    if draw:
        fig,ax = plt.subplots()
        ax.plot(fJ[0],ooop)
        ax.set_xlabel('spatial frequency')
        ax.set_title('smoothing operator')
        fig.canvas.draw()
    if draw:
        ooop_space = np.fft.ifft(ooop).real
        fig,ax = plt.subplots()
        ax.plot(xJ[0],ooop_space)
        ax.set_xlabel('space')
        ax.set_title('smoothing operator space')
        fig.canvas.draw()
    
    # set up config for registration
    A2d = np.eye(3)[None].repeat(J.shape[1],axis=0)
    config = {
        'A2d':A2d,
        'slice_matching':True,
        'dv':2000.0, # a big number to essentially disable it
        'v_start':100000, # disable it
        'n_iter':10, # a small number
        'ev':0, # disable it
        'eA':0, # disable it
        'eA2d':2e2, # this worked reasonably well for my test data
        'downI':[1,4,4], # extra downsampling for registration
        'downJ': [1,4,4],
        'order': 0, # no contrast mapping
        'full_outputs':True,
        'sigmaM':0.1,
        'sigmaB':0.2,
        'sigmaA':0.5,
    }
    config['n_draw'] = 0 # don't draw
    config.update(kwargs) # update with anything else
    
    
    XJ = np.stack(np.meshgrid(*xJ,indexing='ij'))
    xJd,Jd = downsample_image_domain(xJ,J,config['downJ'])

    fig = plt.figure()
    fig2 = plt.figure()



    # first estimate
    I = np.ones_like(J)*(np.sum(J*W,axis=(-1,-2,-3),keepdims=True)/np.sum(W,axis=(-1,-2,-3),keepdims=True))
    I = compute_atlas_from_slices(J,W,ooop,draw=False,I=I)
    if draw:
        draw(I,xJ,fig=fig,interpolation='none',vmin=0,vmax=1)
        fig.suptitle(f'Initial atlas')
        fig.canvas.draw()    
    
    for it in range(n_steps):
        print(f'starting it {it}')
        # map it
        out = emlddmm(I=I,xI=xJ,J=J,xJ=xJ,W0=W,device='cpu',**config)
        # sometimes this gives an error
        # update Jr and Wr

        tform = compose_sequence([Transform(out['A2d'])],XJ)
        
        WM = out['WM'].cpu().numpy()
        WM = interp(xJd,WM[None],XJ)
        Wr = W[None]*WM.cpu().numpy()
        Wr /= Wr.max()
        
        # the approach on the next two lines really helped wiht artifacts near the border
        '''
        Jr = apply_transform_float(xJ,J*Wr,tform,padding_mode='border') # default padding is border
        Wr_ = apply_transform_float(xJ,Wr,tform,padding_mode='border')[0] # default padding is border
        Jr = Jr/(Wr_[None] + 1e-6)
        '''
        
        
        # the alternative is this
        Jr = apply_transform_float(xJ,J,tform,padding_mode='zeros') # default padding is border
        # but I don't think the final Wr is that good, so let's use this        
        Wr = apply_transform_float(xJ,Wr,tform,padding_mode='zeros')[0] # make sure anything out of bounds is 0        
        #Wr = Wr_
        
        
        if draw:
            draw(Jr,xJ,fig=fig2,interpolation='none',vmin=0,vmax=1)
            fig2.suptitle(f'Recon it {it}')
        # update atlas
        I = compute_atlas_from_slices(Jr,Wr,ooop,draw=False,I=I)
        # initialize for next time
        config['A2d'] = out['A2d']
        # draw the result
        if draw:
            draw(I,xJ,fig=fig,interpolation='none',vmin=0,vmax=1)
            fig.suptitle(f'Atlas it {it}')

            fig.canvas.draw()
            fig2.canvas.draw()

        #fig.savefig(join(outdir,f'atlas_it_{it:06d}.jpg'))
        #fig2.savefig(join(outdir,f'recon_it_{it:06d}.jpg'))

    
    out['I'] = I
    out['Jr'] = Jr
    
    return out
    
def weighted_intraclass_variance(bbox,Ji,ellipse=True):
    '''
    Returns weighted intraclass variance (as in Otsu's method) for an image with a given bounding box.
    
    Parameters
    ----------
    bbox : list
        A tuple of 4 ints. [row0, col0, row1, col1].
    Ji : numpy array
        Row x col x n_channels numpy array to find bounding box
        
    Returns
    -------
    E : float
        The weighted intraclass variance between inside and outside the bounding box.
    
    '''
    bbox = np.round(bbox).astype(int)
    
    if not ellipse:
        mask = np.zeros_like(Ji[...,0],dtype=bool)
        mask[bbox[0]:bbox[2],bbox[1]:bbox[3]] = 1
    else:
        # 
        mask = np.zeros_like(Ji[...,0],dtype=bool)
        mask0 = mask[bbox[0]:bbox[2],bbox[1]:bbox[3]]
        x = [np.linspace(-1.0,1.0,ni) for ni in mask0.shape]
        X = np.meshgrid(*x,indexing='ij')
        R2 = X[0]**2 + X[1]**2
        mask0 = R2 <= 1
        mask[bbox[0]:bbox[2],bbox[1]:bbox[3]] = mask0
    mask_ = np.logical_not(mask)
    E = np.sum(mask)*np.sum(np.var(Ji[mask],axis=0)) + np.sum(mask_)*np.sum(np.var(Ji[mask_],axis=0))
    return E    

def find_bounding_box(Ji,startoff=30,search=20,n_iter=10,bbox=None,fixsize=False):    
    '''
    Computes a bounding box using an Otsu intraclass variance objective function.
    
    Parameters
    ----------
    Ji : numpy array
        Row x col x n_channels numpy array to find bounding box
    startoff : int
        How far away from boundary to put initial guess.  Default 30.
    search : int
        How far to move bounding box edges at each iteration.  Default +/- 10.
    n_iter : int
        How many loops through top, left, bottom, right.
        
    Returns
    -------
    bbox : list
        A tuple of 4 ints. [row0, col0, row1, col1].
    
    '''
    if not Ji.size:
        return np.array([0,0,0,0])  

    
    if bbox is None:
        bbox = np.array([startoff,startoff,Ji.shape[0]-startoff-1,Ji.shape[1]-startoff-1])
    else:
        print('Using initial bounding box and ignoring the startoff option')
    
    if fixsize:
        size0 = bbox[2] - bbox[0]
        size1 = bbox[3] - bbox[1]
    E = weighted_intraclass_variance(bbox,Ji)
   

    
    for sideloop in range(n_iter):
        Estart = E
        # optimize bbox0
        
        start = np.clip(bbox[0]-search,a_min=0,a_max=bbox[2]-1)
        end = np.clip(bbox[0]+search,a_min=0,a_max=bbox[2]-1)
        E_ = []
        for i in range(start,end+1):
            bbox_ = np.array(bbox)
            bbox_[0] = i
            if fixsize:
                bbox_[2] = bbox_[0] + size0
            E_.append( weighted_intraclass_variance(bbox_,Ji))
        ind = np.argmin(E_)
        bbox[0] = start+ind
        if fixsize:
            bbox[2] = bbox[0] + size0
       
    
        E = E_[ind]
        
    
        # optimize bbox 1
        start = np.clip(bbox[1]-search,a_min=0,a_max=bbox[3]-1)
        end = np.clip(bbox[1]+search,a_min=0,a_max=bbox[3]-1)
        E_ = []
        for i in range(start,end+1):
            bbox_ = np.array(bbox)
            bbox_[1] = i
            if fixsize:
                bbox_[3] = bbox_[1] + size1
            E_.append( weighted_intraclass_variance(bbox_,Ji))
        ind = np.argmin(E_)
        bbox[1] = start+ind
        if fixsize:
            bbox[3] = bbox[1] + size1
       
    
        E = E_[ind]
        if fixsize:
            if E == Estart:
                break
            continue
        
        
        # optimize bbox 2     
        start = np.clip(bbox[2]-search,a_min=bbox[0]+1,a_max=Ji.shape[0]-1)
        end = np.clip(bbox[2]+search,a_min=bbox[0]+1,a_max=Ji.shape[0]-1)
        E_ = []
        for i in range(start,end+1):
            bbox_ = np.array(bbox)
            bbox_[2] = i
            E_.append( weighted_intraclass_variance(bbox_,Ji))
        ind = np.argmin(E_)
        bbox[2] = start+ind
       
    
        E = E_[ind]
        
        
        # optimize bbox 3   
        start = np.clip(bbox[3]-search,a_min=bbox[1]+1,a_max=Ji.shape[1]-1)
        end = np.clip(bbox[3]+search,a_min=bbox[1]+1,a_max=Ji.shape[1]-1)
        E_ = []
        for i in range(start,end+1):
            bbox_ = np.array(bbox)
            bbox_[3] = i
            E_.append( weighted_intraclass_variance(bbox_,Ji))
        ind = np.argmin(E_)
        bbox[3] = start+ind
       
    
        E = E_[ind]
        
        if E == Estart:
            break

            
        
    return bbox

def initialize_A2d_with_bbox(J,xJ):
    '''
    Use bounding boxes to find an initial guess of A2d.
    
    On each slice we will compute a bounding box.
    
    Then we will compute a translation vector which will move the slice to the center of the field of view.
    
    Then we will return the inverse.
    
    TODO
    '''
    
        
# now we'll start building an interface
if __name__ == '__main__':
    """
    Raises
    ------
    Exception
        If mode is 'register' and 'config' argument is None.
    Exception
        If mode is 'register' and atlas name is not specified.
    Exception
        If atlas image is not 3D.
    Exception
        If mode is 'register' and target name is not specified.
    Exception
        If target image is not a directory (series of slices), and it is not 3D.
    Exception
        If mode is 'transform' and the transform does not include direction ('f' or 'b').
    Exception
        If transform direction is not f or b.
    Exception
        If mode is 'transform' and neither atlas name nor label name is specified.
    Exception
        If mode is 'transform' and target name is not specified.
    

    """
    # set up command line args
    # we will either calculate mappings or apply mappings
    parser = argparse.ArgumentParser(description='Calculate or apply mappings, or run a specified pipeline.', epilog='Enjoy')
    
    
    parser.add_argument('-m','--mode', help='Specify mode as one of register, transform, or a named pipeline', default='register',choices=['register','transform']) 
    # add other choices
    # maybe I don't need a mode
    # if I supply a -x it will apply transforms
    
    parser.add_argument('-a','--atlas',
                        help='Specify the filename of the image to be transformed (atlas)')
    parser.add_argument('-l','--label', help='Specify the filename of the label image in atlas space for QC and outputs')
    
    parser.add_argument('-t','--target', help='Specify the filename of the image to be transformed to (target)')
    parser.add_argument('-w','--weights', help='Specify the filename of the target image weights (defaults to ones)')
    
    
    parser.add_argument('-c','--config', help='Specify the filename of json config file') # only required for reg
    
    parser.add_argument('-x','--xform', help='Specify a list of transform files to apply, or a previous output directory',action='append')
    parser.add_argument('-d','--direction', help='Specify the direction of transforms to apply, either f for forward or b for backward',action='append')
    
    parser.add_argument('-o','--output', help='A directory for outputs', required=True)
    
    parser.add_argument('--output_image_format', help='File format for outputs (vtk legacy and nibabel supported)', default='.vtk')
    parser.add_argument('--num_threads', help='Optionally specify number of threads in torch', type=int)
    parser.add_argument('--atlas_voxel_scale', help='Optionally specify a scale factor for atlas voxel size (e.g. 1000 to convert mm to microns)', type=float)
    parser.add_argument('--target_voxel_scale', help='Optionally specify a scale factor for target voxel size (e.g. 1000 to convert mm to microns)', type=float)

    args = parser.parse_args()
    
    # TODO don't print namespace because it will contain full paths
    #print(args)
    
    if args.num_threads is not None:
        print(f'Setting numer of torch threads to {args.num_threads}')
        torch.set_num_threads(args.num_threads)
    

    
    
    # if mode is register
    if args.mode == 'register':   
        print('Starting register pipeline')    
        if args.config is None:
            raise Exception('Config file option must be set to run registration')
        
        # don't print because it may contain full paths, don't want for web
        #print(f'Making output directory {args.output}')
        if not os.path.isdir(args.output):
            os.mkdir(args.output)
        #print('Finished making output directory')
    
    
        # load config
        print('Loading config')
        with open(args.config) as f:
            config = json.load(f)
        # I'm getting this for initial downsampling for preprocessing
        downJs = config['downJ']
        downIs = config['downI']
        for k in config:
            print(k,config[k])
        print('Finished loading config')
        
        # load atlas
        #atlas_name = '/home/dtward/data/csh_data/marmoset/Woodward_2018/bma-1-mri-reorient.vtk'
        #label_name = '/home/dtward/data/csh_data/marmoset/Woodward_2018/bma-1-region_seg-reorient.vtk'
        #target_name = '/home/dtward/data/csh_data/marmoset/m1229/M1229MRI/MRI/exvivo/HR_T2/HR_T2_CM1229F-reorient.vtk'

        # TODO check works with nifti
        atlas_name = args.atlas
        if atlas_name is None:
            raise Exception('You must specify an atlas name to run the registration pipeline')
        print(f'Loading atlas {atlas_name}')
        parts = os.path.splitext(atlas_name)
        #if parts[-1] != '.vtk':
        #    raise Exception(f'Only vtk format atlas supported, but this file is {parts[-1]}')
        xI,I,title,names = read_data(atlas_name)
        if args.atlas_voxel_scale is not None:
            xI = [x*args.atlas_voxel_scale]
        
        
        I = I.astype(float)
        # pad the first axis if necessary
        if I.ndim == 3: 
            I = I[None]
        elif I.ndim < 3 or I.ndim > 4:
            raise Exception(f'3D data required but atlas image has dimension {I.ndim}')            
        output_dir = os.path.join(args.output,'inputs/')
        makedirs(output_dir,exist_ok=True)
        
        
        print(f'Initial downsampling so not too much gpu memory')
        # initial downsampling so there isn't so much on the gpu
        mindownI = np.min(np.array(downIs),0)
        xI,I = downsample_image_domain(xI,I,mindownI)
        downIs = [ list((np.array(d)/mindownI).astype(int)) for d in downIs]
        dI = [x[1]-x[0] for x in xI]
        print(dI)
        nI = np.array(I.shape,dtype=int)
        # update our config variable
        config['downI'] = downIs
        
        fig = draw(I,xI)
        fig[0].suptitle('Atlas image')
        fig[0].savefig(os.path.join(output_dir,'atlas.png'))
        print('Finished loading atlas')
        
        # load target and possibly weights
        target_name = args.target
        if target_name is None:
            raise Exception('You must specify a target name to run the registration pipeline')
        print(f'Loading target {target_name}')
        parts = os.path.splitext(target_name)
        if not parts[-1]:
            print('Loading slices from directory')
            xJ,J,W0 = load_slices(target_name)            
            
        elif parts[-1] == '.vtk':
            print('Loading volume from vtk')
            xJ,J,title,names = read_vtk_data(target_name) # there's a problem here with marmoset, too many values t ounpack
            if J.ndim == 3:
                J = J[None]
            elif J.ndim < 3 or J.ndim > 4:
                raise Exception(f'3D data required but target image has dimension {J.ndim}')
            if args.weights is not None:
                print('Loading weights from vtk')
            else:
                W0 = np.ones_like(J[0])
        if args.target_voxel_scale is not None:
            xJ = [x*args.target_voxel_scale]
                
        print(f'Initial downsampling so not too much gpu memory')
        # initial downsampling so there isn't so much on the gpu
        mindownJ = np.min(np.array(downJs),0)
        xJ,J = downsample_image_domain(xJ,J,mindownJ)
        W0 = downsample(W0,mindownJ)
        downJs = [ list((np.array(d)/mindownJ).astype(int)) for d in downJs]        
        # update our config variable
        config['downJ'] = downJs
        
        # draw it
        fig = draw(J,xJ)
        fig[0].suptitle('Target image')
        fig[0].savefig(join(output_dir,'target.png'))        
        print('Finished loading target')
        
        # get one more qc file applying the initial affine
        try:
            A = np.array(config['A']).astype(float)
        except:
            A = np.eye(4)
        # this affine matrix should be 4x4, but it may be 1x4x4
        if A.ndim > 2:
            A = A[0]
        Ai = np.linalg.inv(A)
        XJ = np.stack(np.meshgrid(*xJ,indexing='ij'),-1)
        Xs = (Ai[:3,:3]@XJ[...,None])[...,0] + Ai[:3,-1]
        out = interp(xI,I,Xs.transpose((3,0,1,2)))
        fig = draw(out,xJ)
        fig[0].suptitle('Initial transformed atlas')
        fig[0].savefig(join(output_dir,'atlas_to_target_initial_affine.png'))        

        # default pipeline
        print('Starting registration pipeline')
        try:
            output = emlddmm_multiscale(I=I/np.mean(np.abs(I)),xI=[xI],J=J/np.mean(np.abs(J)),xJ=[xJ],W0=W0,**config)
        except:
            print('problem with registration')
            with open('done.txt','wt') as f:
                pass
            
        if isinstance(output,list):
            output = output[-1]
        print('Finished registration pipeline')
        

        # write outputs      
        try:
            write_outputs_for_pair(
                args.output,output,
                xI,I,xJ,J,WJ=W0,    
            )
        except:
            print(f'Problem with writing outputs')
            with open('done.txt','wt') as f:
                pass            
        
        print('Finished writing outputs')
        with open('done.txt','wt') as f:
            pass
        
        '''
        # this commented out section was old, we need to make sure to merge properly with Bryson
        # write transforms
        print('Starting to write transforms')
        write_transform_outputs(args.output,output)
        print('Finished writing transforms')
        
        # write qc outputs
        # this requires a segmentation image
        if args.label is not None:
            print('Starting to read label image for qc')
            xS,S,title,names = read_data(args.label)            
            S = S.astype(np.int32) # with int32 should be supported by torch
            print('Finished reading label image')
            print('Starting to write qc outputs')
            write_qc_outputs(args.output,output,xI,I,xJ,J,xS=xI,S=S) # TODO: qrite_qc_outputs input format has changed
            print('Finished writing qc outputs')
            
            
        # transform imaging data
        # transform target back to atlas
        Xin = torch.stack(torch.meshgrid([torch.as_tensor(x) for x in xI],indexing='ij'))
        Xout = compose_sequence(args.output,Xin)
        Jt = apply_transform_float(xJ,J,Xout)
        
        # transform atlas to target
        Xin = torch.stack(torch.meshgrid([torch.as_tensor(x) for x in xJ],indexing='ij'))
        Xout = compose_sequence(args.output,Xin,direction='b')
        It = apply_transform_float(xI,I,Xout)
        if args.label is not None:
            St = apply_transform_int(xS,S,Xout)
        # write
        ext = args.output_image_format
        if ext[0] != '.': ext = '.' + ext
        '''
            
    elif args.mode == 'transform':
        
        # now to apply transforms, every one needs a f or a b
        # some preprocessing
        if args.direction is None:
            args.direction = ['f']
        if len(args.direction) == 1:
            args.direction = args.direction * len(args.xform)        
        if len(args.xform) != len(args.direction):
            raise Exception(f'You must input a direction for each transform, but you input {len(args.xform)} transforms and {len(args.direction)} directions')
        for tform,direction in zip(args.xform,args.direction):
            if direction.lower() not in ['f','b']:
                raise Exception(f'Transform directions must be f or b, but you input {direction}')
            
            
            
        # load atlas
        # load target (to get space info)
        # compose sequence of transforms
        # transform the data
        # write it out
        if args.atlas is not None:
            atlas_name = args.atlas
        elif args.label is not None:
            atlas_name = args.label
        else:
            raise Exception('You must specify an atlas name or label name to run the transformation pipeline')
        print(f'Loading atlas {atlas_name}')
        parts = os.path.splitext(atlas_name)        
        xI,I,title,names = read_data(atlas_name)        
        
        # load target and possibly weights
        target_name = args.target
        if target_name is None:
            raise Exception('You must specify a target name to run the transformation pipeline (TODO, support specifying a domain rather than an image)')
        print(f'Loading target {target_name}')
        parts = os.path.splitext(target_name)
        xJ,J,title,names = read_data(target_name)
        
        # to transform an image we start with Xin, and compute Xout
        # Xin will be the grid of points in target
        Xin = torch.stack(torch.meshgrid([torch.as_tensor(x) for x in xJ],indexing='ij'))
        Xout = compose_sequence([(x,d) for x,d  in zip(args.xform,args.direction) ], Xin)
        if args.atlas is not None:
            It = apply_transform_float(xI,I,Xout)
        else:
            It = apply_transform_int(xI,I,Xout)
        
        # write out the outputs        
        ext = args.output_image_format
        if ext[0] != '.': ext = '.' + ext                    
        if not os.path.isdir(args.output): os.mkdir(args.output)
        # name should be atlas name to target name, but without the path
        name = splitext(os.path.split(atlas_name)[1])[0] + '_to_' + os.path.splitext(os.path.split(target_name)[1])[0]
        write_data(join(args.output,name+ext),xJ,It,'transformed data')
        # write a text file that summarizes this
        name = join(args.output,name+'.txt')
        with open(name,'wt') as f:
            f.write(str(args))
         
    # also 
