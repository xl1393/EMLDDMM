import numpy as np
import torch
from torch.nn.functional import grid_sample
import os
from os.path import join,split,splitext
from os import makedirs
import json
import re
import argparse
from warnings import warn
from IPython.display import display

from .utils import *
from .io import *
from .vis import *

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
    
