import numpy as np
import torch
from torch.nn.functional import grid_sample
from scipy.stats import mode
from scipy.interpolate import interpn

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
