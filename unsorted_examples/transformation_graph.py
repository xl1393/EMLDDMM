# %%
import emlddmm
import torch
import numpy as np
import json
import os
import sys
import argparse
from argparse import RawTextHelpFormatter
import pickle

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'
dtype = torch.float

# %%

def add_edge(adj, spaces, src_space, dest_space, out, slice_matching=False):
    """ Add edge between two vertices in registration adjacency list.

    Adds the key:value pair, dest_space:('path/to/tranforms', 'direction') to the dict corresponding to source space in the adjacency list.
    The adjacency list (adj) has the form [{src_space_1 dict}, ..., {src_space_n dict}] for n source spaces.
    Each source space dict has the form {dest_space_1: ('path/to/transforms_1', 'direction'), ..., dest_space_n: ('path/to/transforms_n', 'direction}
    which indicates the spaces to which the source space were registered, and the location of transforms corresponding to that registration.
    For each registration both the forward and backward transformation is recorded in the list. e.g. [{space_1: ('space_0_to_space_1/transforms', 'f')}, {space_0: ('space_0_to_space_1/transforms', 'b')}

    Parameters
    ----------
    adj : list of dicts
        adjacency list of transformations between spaces
    spaces: dict
        user assigned space name strings each assigned an integer value as a key:value pair in a dictionary
    src_space: str
        user assigned source space name
    dest_space: str
        user assigned destination space name
    out: str
        path to registration outputs

    Example
    -------
    >>> spaces = {'HIST': 0, 'MRI': 1, 'CCF': 2, 'CT': 3}
    >>> adj = [{} for i in range(len(spaces))]

    >>> print(adj)
    [{}, {}, {}, {}]

    >>> add_edge(adj, spaces, 'MRI', 'CCF', 'outputs')
    >>> print(adj)
    [{}, {2: ('outputs/CCF/MRI_to_CCF/', 'f')}, {1: ('outputs/CCF/MRI_to_CCF/', 'b')}, {}]

    >>> add_edge(adj, spaces, 'HIST', 'MRI', 'outputs', slice_matching=True)
    >>> print(adj)
    [{1: ('outputs/MRI/HIST_REGISTERED_to_MRI/', 'f')}, {2: ('outputs/CCF/MRI_to_CCF/', 'f'), 0: ('outputs/MRI/HIST_REGISTERED_to_MRI/', 'b')}, {1: ('outputs/CCF/MRI_to_CCF/', 'b')}, {}]
    
    """ 
    if slice_matching:
        transforms_path = os.path.join(out,f'{dest_space}/{src_space}_REGISTERED_to_{dest_space}/')
    else:
        transforms_path = os.path.join(out,f'{dest_space}/{src_space}_to_{dest_space}/')

    adj[spaces[src_space]][spaces[dest_space]] = (transforms_path, 'f')
    adj[spaces[dest_space]][spaces[src_space]] = (transforms_path, 'b')


def BFS(adj, src, dest, v, pred, dist):
    """ Breadth first search

    a modified version of BFS that stores predecessor
    of each vertex in array pred and its distance from source in array dist

    Parameters
    ----------
    adj : list of dicts
        adjacency list of transformations between spaces
    src: int
        int value given by corresponding source in spaces dict
    dest: int 
        dest value given by corresponding destination in spaces dict
    v: int
        length of spaces dict
    pred: list of ints
        stores predecessor of vertex i at pred[i]
    dist: list of ints
        stores distance (by number of vertices) of vertex i from source vertex

    Returns
    -------
    bool
        True if a path from src to dest is found and False otherwise

    """
 

    queue = []
  
    visited = [False for i in range(v)]
    # for each space we initialize the distance from src to be a large number and the predecessor to be -1
    for i in range(v):
 
        dist[i] = 1000000
        pred[i] = -1
     
    # visit source first. Distance from source to itself is 0
    visited[src] = True
    dist[src] = 0
    queue.append(src)
  
    # BFS algorithm
    while (len(queue) != 0):
        u = queue[0]
        queue.pop(0)
        for i in range(len(adj[u])):
         
            if (visited[list(adj[u])[i]] == False):
                visited[list(adj[u])[i]] = True
                dist[list(adj[u])[i]] = dist[u] + 1
                pred[list(adj[u])[i]] = u
                queue.append(list(adj[u])[i])
  
                # We stop BFS when we find
                # destination.
                if (list(adj[u])[i] == dest):
                    return True
  
    return False
  
  
def find_shortest_path(adj, src, dest, v):
    """ Find Shortest Path

    Finds the shortest path between src and dest in the adjacency list and prints its length

    Parameters
    ----------
    adj : list of dicts
        adjacency list of transformations between spaces
    src: int
        int value given by corresponding source in spaces dict
    dest: int 
        dest value given by corresponding destination in spaces dict
    v: int
        length of spaces dict

    Returns
    -------
    path : list of ints
        path from src to dest using integer values of the adjacency list vertices. Integers can be converted to space names by the spaces dict.


    Example
    -------
    >>> adj = [{1: ('outputs/MRI/HIST_REGISTERED_to_MRI/', 'f')},
    ...        {2: ('outputs/CCF/MRI_to_CCF/', 'f'), 0: ('outputs/MRI/HIST_REGISTERED_to_MRI/', 'b')},
    ...        {1: ('outputs/CCF/MRI_to_CCF/', 'b')},
    ...        {}]
    >>> path = find_shortest_path(adj, 0, 2, 4)
    Shortest path length is: 2

    >>> print(path)
    [0,1,2]

    >>> path = transformation_graph.find_shortest_path(adj, 0, 3, 4)
    Given source and destination are not connected

    """
     
    pred=[0 for i in range(v)] # predecessor of space i in path from src to dest
    dist=[0 for i in range(v)] # distance of vertex i by number of vertices from src
  
    if (BFS(adj, src, dest, v, pred, dist) == False):
        print("Given source and destination are not connected")
        
    # path stores the shortest path
    path = []
    crawl = dest
    path.append(crawl)
     
    while (pred[crawl] != -1):
        path.append(pred[crawl])
        crawl = pred[crawl]
     
    path.reverse()

    if len(path) > 1:
        # distance from source is in distance array
        print("Shortest path length is: " + str(dist[dest]), end = '')

    return path


def get_transformation(adj, path):
    """ Get Transformation

    Gets file path or paths to the transforms needed to match src to dest

    Parameters
    ----------
    adj : list of dicts
        adjacency list of transformations between spaces
    path : list of ints
        path from src to dest using integer values of the adjacency list vertices. Integers can be converted to space names by the spaces dict.

    Returns
    -------
    transformation : list of strings
        list of file paths to sequence of transformations matching src to dest

    Example
    -------
    >>> adj = [{1: ('outputs/MRI/HIST_REGISTERED_to_MRI/', 'f')},
    ...        {2: ('outputs/CCF/MRI_to_CCF/', 'f'), 0: ('outputs/MRI/HIST_REGISTERED_to_MRI/', 'b')},
    ...        {1: ('outputs/CCF/MRI_to_CCF/', 'b')},
    ...        {}]
    >>> path = [0,1,2]
    >>> transformation = get_transformation(adj, path)
    >>> print(transformation)
    [('outputs/MRI/HIST_REGISTERED_to_MRI/', 'f'), ('outputs/CCF/MRI_to_CCF/', 'f')]

    """
    transformation = []
    for i in range(len(path)-1):
        transformation.append(adj[path[i]][path[i+1]])

    return transformation


def reg(dest, source, registration, config, out, labels=None):
    """ registration

    Registers a single source image and destination image. Affine matrix and velocity field transformations, and qc images are saved.

    Parameters
    ----------
    dest : str
        path to destination image
    source: str
        path to source image or series of images
    registration: list
        List containing two lists of strings, e.g. [["src_space","src_img"],["dest_space","dest_img"]].
        These are used for naming output files.
    config: str
        path to JSON configuration file specifying registration parameters
    out: str
        path to output directory
    labels: str
        path to labels image used for qc

    Example
    -------
    >>> dest = 'average_template_50.vtk'
    >>> source = 'HR_NIHxCSHL_50um_14T_M1_masked.vtk'
    >>> registration = [['MRI','masked'], ['CCF','average_template_50']]
    >>> config = 'configMD816_MR_to_CCF.json'
    >>> out = 'outputs/example_output'

    >>> reg(dest, source, registration, config, out)

    """
    src_path = source
    src_space = registration[0][0]
    src_img = registration[0][1]
    dest_path = dest
    dest_img = registration[1][1]
    dest_space = registration[1][0]
    config_file = config
    output_dir = out 
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    label_name = labels
    with open(config_file) as f:
        config = json.load(f)
    # I'm getting this for initial downsampling for preprocessing
    downIs = config['downI']
    downJs = config['downJ']

    # atlas
    xI,I,title,names = emlddmm.read_data(dest_path)
    I = I.astype(float)
    # normalize
    I /= np.mean(np.abs(I))
    I /= np.quantile(I, 0.99)

    # initial downsampling so there isn't so much on the gpu
    mindownI = np.min(np.array(downIs),0)
    xI,I = emlddmm.downsample_image_domain(xI,I,mindownI)
    downIs = [ list((np.array(d)/mindownI).astype(int)) for d in downIs]
    # update our config variable
    config['downI'] = downIs

    # target
    xJ,J,title,names = emlddmm.read_data(src_path)
    if 'mask' in names:
        maskind = names.index('mask')
        W0 = J[maskind]
        J = J[np.arange(J.shape[0])!=maskind]
    else:
        W0 = np.ones_like(J[0])
    J = J.astype(float)
    # normalize
    J /= np.mean(np.abs(J))
    # normJ = np.quantile(J, 0.99)
    # J /= normJ

    # initial downsampling so there isn't so much on the gpu
    mindownJ = np.min(np.array(downJs),0)
    xJ,J = emlddmm.downsample_image_domain(xJ,J,mindownJ)
    W0 = emlddmm.downsample(W0,mindownJ)
    downJs = [ list((np.array(d)/mindownJ).astype(int)) for d in downJs]
    # update our config variable
    config['downJ'] = downJs

    if 'A' in config:
        A = np.array(config['A']).astype(float)
    else:
        A = np.eye(4)
    print('Initial affine: \n', A)

    # if 'slice_matching' not in config:
    #     # for simplicity I will add a translation manually
    #     A[:3, -1] = [-4000, 100, 4000]
    #     config['A'] = A

    # update sigma factors based on normalization
    #if 'sigmaR' in config:
    #    config['sigmaR'][0] /= normJ
    #if 'sigmaA' in config:
    #    config['sigmaA'][0] /= normJ
    #if 'sigmaB' in config:
    #    config['sigmaB'][0] /= normJ
    #if 'sigmaM' in config:
    #    config['sigmaM'][0] /= normJ

    if 'device' not in config:
        config['device'] = 'cuda:0'
        # config['device'] = 'cpu'
    output = emlddmm.emlddmm_multiscale(I=I,xI=[xI],J=J,xJ=[xJ],W0=W0,full_outputs=False,**config)
    #write outputs
    print('saving transformations to ' + output_dir + '...')
    emlddmm.write_transform_outputs(output_dir, src_space, dest_space, output[-1], src_path)
    print('saving qc to ' + output_dir + '...')
    if label_name:
        # get labels
        xS,S,title,names = emlddmm.read_data(label_name,endian='l')
        emlddmm.write_qc_outputs(output_dir, src_space, src_img, dest_space, dest_img, output[-1],xI,I,xJ,J,xS=xS,S=S.astype(float))
    else:
        emlddmm.write_qc_outputs(output_dir, src_space, src_img, dest_space, dest_img, output[-1],xI,I,xJ,J)

    return


def run_registrations(reg_list):
    """ Run Registrations

    Runs a sequence of registrations given by reg_list using reg method, which saves transformations and qc images.

    Parameters
    ----------
    reg_list : list of dicts
        each dict in reg_list specifies the source image path, destination image path,
        src and dest space names, output directory, and registration configuration settings.

    Returns
    -------
    adj : list of dicts
        adjacency list of transformations between spaces
    spaces: dict
        user assigned space name strings each assigned an integer value as a key:value pair in a dictionary
    
    Example
    -------
    >>> reg_list = [{'registration':[['MRI','masked'],['CCF','average_template_50']],
                     'source': 'HR_NIHxCSHL_50um_14T_M1_masked.vtk',
                     'dest': 'average_template_50.vtk',
                     'config': 'configMD816_MR_to_CCF.json',
                     'output': 'outputs/example_output'},
                    {'registration':[['HIST','Nissl'],['MRI','masked']],
                     'source': 'MD816_STIF',
                     'dest': 'HR_NIHxCSHL_50um_14T_M1_masked.vtk',
                     'config': 'configMD816_Nissl_to_MR.json',
                     'output': 'outputs/example_output'}]
    >>> adj, spaces = run_registrations(reg_list)
    >>> print(adj)
    [{1: ('outputs/example_output/CCF/MRI_to_CCF/', 'f'), 2: ('outputs/example_output/MRI/HIST_REGISTERED_to_MRI/', 'b')},
     {0: ('outputs/example_output/CCF/MRI_to_CCF/', 'b')},
     {0: ('outputs/example_output/MRI/HIST_REGISTERED_to_MRI/', 'f')}]

    >>> print(spaces)
    {'MRI': 0, 'CCF': 1, 'HIST': 2}

    """
    # input: list of dicts of sources, targets, labels, configs, and the output dir 
    # Return: adj list, dict of space name keys and node number values

    # perform registration using reg_list as input
    for r in reg_list:
        dest = r['dest']
        source = r['source']
        registration = r['registration']
        config = r['config']
        out = r['output']
        if 'label_name' in r:
            label_name = r['label_name']
            reg(dest, source, registration, config, out, labels=label_name)
        else:
            reg(dest, source, registration, config, out)
         
    # construct spaces dict
    spaces = {}
    v = 0
    for i in reg_list:
        for j in [i['registration'][0][0], i['registration'][1][0]]: # for src and dest space names in each registration
            if j not in spaces:
                spaces[j] = v
                v += 1

    adj = [{} for i in range(len(spaces))]

    # add edge to adj list
    for i in range(len(reg_list)):
        src_space = reg_list[i]['registration'][0][0]
        config_file = reg_list[i]['config']
        with open(config_file) as f:
            config = json.load(f)
        if 'slice_matching' in config: # transforms get saved in {src_space}_REGISTERED_to_{dest_space} if slicematching
            slice_matching = True
        else:
            slice_matching = False
        dest_space = reg_list[i]['registration'][1][0]
        out = reg_list[i]['output']
        add_edge(adj, spaces, src_space, dest_space, out, slice_matching)

    return adj, spaces


def apply_transformation(adj, spaces, src_space, src_img, dest_space, out, src_path, dest_path):
    """ Apply Transformation

    Applies affine matrix and velocity field transforms to map dest points to src points. Saves displacement field from dest points to src points
    (i.e. difference between transformed coordinates and input coordinates), and determinant of Jacobian for 3d destination spaces. Also saves transformed image in vtk format.

    Parameters
    ----------
    adj: list of dicts
        adjacency list of transformations between spaces
    spaces: dict
        user assigned space name strings each assigned an integer value as a key:value pair in a dictionary
    src_space: str
        user assigned src space name used for naming outputs
    src_img: str
        user assigned src image name used for naming outputs
    dest_space: str
        user assigned dest image name used for naming outputs
    out: str
        path to registration outputs root
    src_path: str
        path to source image (image to be sampled on transformed points)
    dest_path: str
        path to destination image (image whos points are to be transformed)
    
    Example
    -------
    >>> adj = [{1: ('outputs/example_output/CCF/MRI_to_CCF/', 'f'), 2: ('outputs/example_output/MRI/HIST_REGISTERED_to_MRI/', 'b')},
    ... {0: ('outputs/example_output/CCF/MRI_to_CCF/', 'b')},
    ... {0: ('outputs/example_output/MRI/HIST_REGISTERED_to_MRI/', 'f')}]
    >>> spaces = {'MRI': 0, 'CCF': 1, 'HIST': 2}
    >>> apply_transformation(adj, spaces, 'MRI', 'masked', 'CCF', 'outputs/example_output',
    ... 'HR_NIHxCSHL_50um_14T_M1_masked.vtk', 'average_template_50.vtk')
    """
    # input: image to be transformed (src_path or I), img space to to which the source image will be matched (dest_path, J), adjacency list and spaces dict from run_registration, source and destination space names
    # return: transfromed image
    
    # load source image
    xJ, J, J_title, _ = emlddmm.read_data(src_path) # the image to be transformed
    J = J.astype(float)
    J = torch.as_tensor(J,dtype=dtype,device=device)
    xJ = [torch.as_tensor(np.copy(x),dtype=dtype,device=device) for x in xJ]

    '''in the special case of transforming an image series to the same space, e.g. [[HIST, nissl], [HIST, nissl]], we output HIST_INPUT_to_HIST_REGISTERED images.
        TODO: in the future, we should be able to align different histology stains, e.g. [[HIST, myelin], [HIST, nissl]]. In this case two rigid transformations are needed,
        one to align input myelin to input nissl, and one to align neighboring slices (input to registered space).'''
    if J_title == 'slice_dataset' and dest_space == src_space:
        # get image slice names for naming output images
        dest_slice_names = [os.path.splitext(x)[0] for x in os.listdir(dest_path) if x[-4:] == 'json']
        dest_slice_names = sorted(dest_slice_names, key=lambda x: x[-4:])
        src_slice_names = [os.path.splitext(x)[0] for x in os.listdir(src_path) if x[-4:] == 'json']
        src_slice_names = sorted(src_slice_names, key=lambda x: x[-4:])
        space = dest_space # src_space = dest_space = space

        x_series = xJ
        X_series = torch.stack(torch.meshgrid(x_series, indexing='ij'), -1)
        transforms = os.path.join(out, f'{space}_REGISTERED/{space}_INPUT_to_{space}_REGISTERED/transforms')
        transforms_ls = sorted(os.listdir(transforms), key=lambda x: x.split('_matrix.txt')[0][-4:])

        A2d = []
        for t in transforms_ls:
            A2d_ = np.genfromtxt(os.path.join(transforms, t), delimiter=',')
            # note that there are nans at the end if I have commas at the end
            if np.isnan(A2d_[0, -1]):
                A2d_ = A2d_[:, :A2d_.shape[1] - 1]
            A2d.append(A2d_)

        A2d = torch.as_tensor(np.stack(A2d),dtype=dtype,device=device)
        A2di = torch.inverse(A2d)
        points = (A2di[:, None, None, :2, :2] @ X_series[..., 1:, None])[..., 0] 
        m0 = torch.min(points[..., 0])
        M0 = torch.max(points[..., 0])
        m1 = torch.min(points[..., 1])
        M1 = torch.max(points[..., 1])
        # construct a recon domain
        dJ = [x[1] - x[0] for x in x_series]
        # print('dJ shape: ', [x.shape for x in dJ])
        xr0 = torch.arange(float(m0), float(M0), dJ[1], device=m0.device, dtype=m0.dtype)
        xr1 = torch.arange(float(m1), float(M1), dJ[2], device=m0.device, dtype=m0.dtype)
        xr = x_series[0], xr0, xr1
        XR = torch.stack(torch.meshgrid(xr,indexing='ij'), -1)
        # reconstruct 2d series
        Xs = torch.clone(XR)
        Xs[..., 1:] = (A2d[:, None, None, :2, :2] @ XR[..., 1:, None])[..., 0] + A2d[:, None, None, :2, -1]
        Xs = Xs.permute(3, 0, 1, 2)
        Jr = emlddmm.interp(xJ, J, Xs)

        # save transformed 2d images   
        img_out = os.path.join(out, f'{space}_REGISTERED/{space}_INPUT_to_{space}_REGISTERED/images')
        if not os.path.exists(img_out):
            os.makedirs(img_out)
        for i in range(Jr.shape[1]):
            Jr_ = Jr[:, i, None, ...]
            xr_ = [torch.tensor([xr[0][i], xr[0][i]+10]), xr[1], xr[2]]
            title = f'{space}_input_{src_slice_names[i]}_to_{space}_registered_{dest_slice_names[i]}'
            emlddmm.write_vtk_data(os.path.join(img_out, f'{space}_INPUT_{src_slice_names[i]}_to_{space}_REGISTERED_{dest_slice_names[i]}.vtk'), xr_, Jr_, title)
        return

    # load destination image
    xI, I, I_title, _ = emlddmm.read_data(dest_path) # the space to transform into
    I = I.astype(float)
    I = torch.as_tensor(I, dtype=dtype, device=device)
    xI = [torch.as_tensor(np.copy(x),dtype=dtype,device=device) for x in xI]

    slice_matching = 'slice_dataset' in [I_title, J_title]
    # if slice_matching then construct the reconstructed space XR
    if slice_matching:
        if I_title=='slice_dataset': # then the last transform in transformation_seq should contain A2d files
            # transforms = os.path.join(transformation_seq[-1][0], 'transforms')
            transforms = os.path.join(out, f'{dest_space}_REGISTERED/{dest_space}_INPUT_to_{dest_space}_REGISTERED/transforms')
        else: # otherwise the first transform in transformation_seq should contain A2d filess
            # transforms = os.path.join(transformation_seq[0][0], 'transforms')
            transforms = os.path.join(out, f'{src_space}_REGISTERED/{src_space}_INPUT_to_{src_space}_REGISTERED/transforms') 
        transforms_ls = [f for f in os.listdir(transforms) if 'vtk' not in f and 'A.txt' not in f]
        transforms_ls = sorted(transforms_ls, key=lambda x: x.split('_matrix.txt')[0][-4:])
        # determine which image is constructed from a 2d series, I or J.
        x_series = xI if I_title=='slice_dataset' else xJ
        X_series = torch.stack(torch.meshgrid(x_series, indexing='ij'),-1)

        A2d = []
        for t in transforms_ls:
            A2d_ = np.genfromtxt(os.path.join(transforms, t), delimiter=',')
            # note that there are nans at the end if I have commas at the end
            if np.isnan(A2d_[0, -1]):
                A2d_ = A2d_[:, :A2d_.shape[1] - 1]
            A2d.append(A2d_)

        A2d = torch.as_tensor(np.stack(A2d),dtype=dtype,device=device)
        A2di = torch.inverse(A2d)
        points = (A2di[:, None, None, :2, :2] @ X_series[..., 1:, None])[..., 0] # reconstructed space needs to be created from the 2d series coordinates
        m0 = torch.min(points[..., 0])
        M0 = torch.max(points[..., 0])
        m1 = torch.min(points[..., 1])
        M1 = torch.max(points[..., 1])
        # construct a recon domain
        dJ = [x[1] - x[0] for x in x_series]
        # print('dJ shape: ', [x.shape for x in dJ])
        xr0 = torch.arange(float(m0), float(M0), dJ[1], device=m0.device, dtype=m0.dtype)
        xr1 = torch.arange(float(m1), float(M1), dJ[2], device=m0.device, dtype=m0.dtype)
        xr = x_series[0], xr0, xr1
        XR = torch.stack(torch.meshgrid(xr, indexing='ij'), -1)
        # reconstruct 2d series
        Xs = torch.clone(XR)
        Xs[..., 1:] = (A2d[:, None, None, :2, :2] @ XR[..., 1:, None])[..., 0] + A2d[:, None, None, :2, -1]
        Xs = Xs.permute(3, 0, 1, 2)

    path = find_shortest_path(adj, spaces[src_space], spaces[dest_space], len(spaces))
    if len(path) < 2:
        return
    print("\nPath is:")

    # printing path as sequence of space names
    for i in path:
        for key, value in spaces.items():
            if i == value:
                print(key, end=' ')

    transformation_seq = get_transformation(adj, path)
    print('\nTransformation sequence: ', transformation_seq)

    # if slice_matching and the destination is 2d series, then X = XR
    if I_title == 'slice_dataset':
        X = torch.clone(XR.permute(3,0,1,2)) # the reconstructed registered domain

        # we will also need the input domain for getting to_input displacement and images later
        Xin = torch.clone(X_series) # note that coordinates are on the last dimension e.g. (i,j,k,3)           
        Xin[..., 1:] = ((A2di[:,None,None,:2,:2] @ (Xin[..., 1:][..., None]))[...,0] + A2di[:,None,None,:2,-1])
        Xin = Xin.permute(3,0,1,2) # (3,i,j,k)
        for i in reversed(range(len(transformation_seq))):
            Xin = emlddmm.compose_sequence([transformation_seq[i]], Xin)
    else:
        X = torch.stack(torch.meshgrid([torch.as_tensor(x) for x in xI], indexing='ij'))
    for i in reversed(range(len(transformation_seq))):
        X = emlddmm.compose_sequence([transformation_seq[i]], X)

    # get displacement
    if I_title == 'slice_dataset':
        # for input disp we need to apply compose_sequence to  A2di @ X_series to get Xin and then input_disp = Xin - X_series
        input_disp = (Xin - X_series.permute(3,0,1,2).cpu())[None]
        registered_disp = (X - XR.permute(3,0,1,2).cpu())[None]
        
        # save out displacement from input and from registered space
        input_dir = os.path.join(out, f'{dest_space}_INPUT/{src_space}_to_{dest_space}_INPUT/transforms/')
        if not os.path.isdir(input_dir):
            os.makedirs(input_dir)
        
        registered_dir = os.path.join(out, f'{dest_space}_REGISTERED/{src_space}_to_{dest_space}_REGISTERED/transforms/')
        if not os.path.isdir(registered_dir):
            os.makedirs(registered_dir)

        # get image names of slices for naming outputs
        slice_names = [os.path.splitext(x)[0] for x in os.listdir(dest_path) if x[-4:] == 'json']
        slice_names = sorted(slice_names, key=lambda x: x[-4:])

        for i in range(input_disp.shape[2]):
            x_series_ = [x_series[0][i], x_series[1], x_series[2]]
            x_series_[0] = torch.tensor([x_series_[0], x_series_[0] + 10])
            xr_ = [xr[0][i], xr[1], xr[2]]
            xr_[0] = torch.tensor([xr_[0], xr_[0] + 10])
            # write out input to dest displacement
            output_name = os.path.join(input_dir, f'{dest_space}_INPUT_{slice_names[i]}_to_{src_space}_displacement.vtk')
            title = f'{dest_space}_INPUT_{slice_names[i]}_to_{src_space}_displacement'
            emlddmm.write_vtk_data(output_name, x_series_, input_disp[:,:, i, None, ...], title)
            # write out registered to dest displacement
            output_name = os.path.join(registered_dir, f'{dest_space}_REGISTERED_{slice_names[i]}_to_{src_space}_displacement.vtk')
            title = f'{dest_space}_REGISTERED_{slice_names[i]}_to_{src_space}_displacement'
            emlddmm.write_vtk_data(output_name, xr_, registered_disp[:,:, i, None, ...], title)

            # TODO: write out detjac for 2d displacements

    else:
        # save out 3d displacement
        disp = (X - torch.stack(torch.meshgrid(xI, indexing='ij')).to('cpu'))[None]
        
        if J_title == 'slice_dataset':
            transform_dir = os.path.join(out, f'{dest_space}/{src_space}_REGISTERED_to_{dest_space}/transforms/')
            if not os.path.exists(transform_dir):
                os.makedirs(transform_dir)
            output_name = os.path.join(transform_dir, f'{dest_space}_to_{src_space}_REGISTERED_displacement.vtk')
            title = f'{dest_space}_to_{src_space}_REGISTERED_displacement'

        else:
            transform_dir = os.path.join(out, f'{dest_space}/{src_space}_to_{dest_space}/transforms/')
            if not os.path.isdir(transform_dir):
                os.makedirs(transform_dir)  
            output_name = os.path.join(transform_dir, f'{dest_space}_to_{src_space}_displacement.vtk')
            title = f'{dest_space}_to_{src_space}_displacement'

        emlddmm.write_vtk_data(output_name, xI, disp, title)

        # write out determinant of jacobian (detjac) of the transformed coordinates
        dv = [(x[1]-x[0]).to('cpu') for x in xI]
        jacobian = lambda X,dv : np.stack(np.gradient(X, dv[0],dv[1],dv[2], axis=(1,2,3))).transpose(2,3,4,0,1)
        jac = jacobian(X,dv)
        detjac = np.linalg.det(jac)
        if J_title == 'slice_dataset':
            output_name = os.path.join(transform_dir, f'{dest_space}_to_{src_space}_REGISTERED_detjac.vtk')
            title = f'{dest_space}_to_{src_space}_REGISTERED_detjac'
        else:
            output_name = os.path.join(transform_dir, f'{dest_space}_to_{src_space}_detjac.vtk')
            title = f'{dest_space}_to_{src_space}_detjac'
        emlddmm.write_vtk_data(output_name, xI, detjac[None], title)

    # now apply transformation to image
    # if slice_matching and the source is 2d series, then use xr and Jr
    if J_title == 'slice_dataset':
        # register 2d series
        Jr = emlddmm.interp(xJ, J, Xs)
        # apply transform to registered image
        AphiI = emlddmm.apply_transform_float(xr, Jr, X.to(device))
    else:
        AphiI = emlddmm.apply_transform_float(xJ, J, X.to(device))
    
    # visualize
    if I_title == 'slice_dataset': # if the destination is 2d series
        x = xr
    else:
        x = xI
    # fig = emlddmm.draw(AphiI, x)
    # fig[0].suptitle(f'transformed {src_space} {src_img} to {dest_space}')
    # fig[0].canvas.draw()
    # plt.show()

    # save transformed images
    if I_title == 'slice_dataset':
        # first save out images of src space to registered slices
        registered_out = os.path.join(out, f'{dest_space}_REGISTERED/{src_space}_to_{dest_space}_REGISTERED/images/')
        if not os.path.exists(registered_out):
            os.makedirs(registered_out)
        for i in range(AphiI.shape[1]):
            AphiI_ = AphiI[:, i, None, ...]
            x_ = [torch.tensor([x[0][i], x[0][i]+10]), x[1], x[2]]
            emlddmm.write_vtk_data(os.path.join(registered_out, f'{src_space}_{src_img}_to_{dest_space}_registered_{slice_names[i]}.vtk'), x_, AphiI_, f'{src_space}_{src_img}_to_{dest_space}_registered_{slice_names[i]}')

        # now save images of src space to slices in input space
        AphiI_to_input = emlddmm.apply_transform_float(xJ, J, Xin.to(device))
        input_out = os.path.join(out, f'{dest_space}_INPUT/{src_space}_to_{dest_space}_INPUT/images/')
        if not os.path.exists(input_out):
            os.makedirs(input_out)
        for i in range(AphiI_to_input.shape[1]):
            AphiI_to_input_ = AphiI_to_input[:, i, None, ...]
            xI_ = [torch.tensor([xI[0][i], xI[0][i]+10]), xI[1], xI[2]]
            emlddmm.write_vtk_data(os.path.join(input_out, f'{src_space}_{src_img}_to_{dest_space}_input_{slice_names[i]}.vtk'), xI_, AphiI_to_input_, f'{src_space}_{src_img}_to_{dest_space}_input_{slice_names[i]}')
        
    else:
        if J_title == 'slice_dataset':
            img_out = os.path.join(out, f'{dest_space}/{src_space}_INPUT_to_{dest_space}/images/')
            if not os.path.exists(img_out):
                os.makedirs(img_out)

        else:
            img_out = os.path.join(out, f'{dest_space}/{src_space}_to_{dest_space}/images/')
            if not os.path.exists(img_out):
                os.makedirs(img_out)
        emlddmm.write_vtk_data(os.path.join(img_out, f'{src_space}_{src_img}_to_{dest_space}.vtk'), x, AphiI, f'{src_space}_{src_img}_to_{dest_space}')

    # save text file of transformation order
    if I_title == "slice_dataset":
        # save transformation sequence to INPUT directory
        with open(os.path.join(out, f'{dest_space}_INPUT/{src_space}_to_{dest_space}_INPUT/{src_space}_to_{dest_space}_transform_seq.txt'), 'w') as f:
            for transform in reversed(transformation_seq[1:]):
                f.write(str(transform) + ', ')
            f.write(str(transformation_seq[0]))
        # save transformation sequence to RECONSTRUCTED directory
            with open(os.path.join(out, f'{dest_space}_REGISTERED/{src_space}_to_{dest_space}_REGISTERED/{src_space}_to_{dest_space}_transform_seq.txt'), 'w') as f:
                for transform in reversed(transformation_seq[1:]):
                    f.write(str(transform) + ', ')
                f.write(str(transformation_seq[0]))

    else:
        if J_title == 'slice_dataset':
            output_name = os.path.join(out, f'{dest_space}/{src_space}_REGISTERED_to_{dest_space}/{src_space}_REGISTERED_{src_img}_to_{dest_space}_transform_seq.txt')
        else:
            output_name = os.path.join(out, f'{dest_space}/{src_space}_to_{dest_space}/{src_space}_{src_img}_to_{dest_space}_transform_seq.txt')
        with open(output_name, 'w') as f:
            for transform in reversed(transformation_seq[1:]):
                f.write(str(transform) + ', ')
            f.write(str(transformation_seq[0]))

    return


def main():
    """ Main

    Main function for parsing input arguments, calculating registrations and applying transformations.

    Example
    -------
    $ python transformation_graph.py --infile GDMInput.json

    """

    help_string = "Arg parser looks for one argument, \'--infile\', which is a JSON file with the following entries: \n\
1) \"space_image_path\": a list of lists, each containing the space name, image name, and path to an image or image series. (Required)\n\
2) \"registrations\": a list of lists, each containing two space-image pairs to be registered. e.g. [[[\"HIST\", \"nissl\"], [\"MRI\", \"masked\"]],\n\
                                                                                                    [[\"MRI\", \"masked\"], [\"CCF\", \"average_template_50\"]],\n\
                                                                                                    [[\"MRI\", \"masked\"], [\"CT\", \"masked\"]]]\n\
                    If registrations were previously computed, this argument may be left out, and \"adj\" and \"spaces\" arguments can be used to apply transforms.\n\
3) \"configs\": list of paths to registration config JSON files, the order of which corresponds to the order of registrations listed in the previous value. (Required if computing registrations)\n\
4) \"output\": output directory which will be the output hierarchy root. If none is given, it is set to the current directory. (Required)\n\
5) \"transforms\": transforms to apply after they are computed from registration. Only necessary if \"transform_all\" is False. Format is the same as \"registrations\".\n\
6) \"transform_all\": bool. Applys all possible transformations given the transformation graph (adjacency list) formed by the registrations performed.\n\
7) \"adj\": path to \"adjacency_list.p\" pickle file saved to output after performing registrations. (Only required for reconstructing images from previously computed registrations)\n\
8) \"spaces\": path to \"spaces_dict.p\". Required with \"adj\" for reconstructing previously computed registrations."

    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('--infile', nargs=1,
                        help=help_string,
                        type=argparse.FileType('r'))
    arguments = parser.parse_args()

    input_dict = json.load(arguments.infile[0])

    output = input_dict["output"] if "output" in input_dict else os.getcwd()
    if not os.path.exists(output):
        os.makedirs(output)
    # save out the json file used for input
    with open(os.path.join(output, "infile.json"), "w") as f:
        json.dump(input_dict, f, indent="")

    try:
        space_image_path = input_dict["space_image_path"]
    except KeyError:
        print("space_image_path is a required argument. It is a list of images, with each image being a list of the format: [\"space name\", \"image name\", \"image path\"]")
    if "registrations" in input_dict:
        registrations = input_dict["registrations"]
        try:
            configs = input_dict["configs"] # if registrations then there must also be configs
        except KeyError:
            ("configs must be included in input with registrations. configs is a list of full paths to JSON registration configuration files.")
    if "transforms" in input_dict:
        transforms = input_dict["transforms"]
    if "adj" in input_dict:
        adj = pickle.load(open(input_dict["adj"], "rb"))
        try:
            spaces = pickle.load(open(input_dict["spaces"], "rb")) # if adj then there must also be spaces
        except KeyError:
            ("spaces must be included with adj. spaces is a dictionary of format \{\"spacename\": <space number>\, ...}, where the space number corresponds to the space number in the adjacency list.")
    transform_all = False
    if "transform_all" in input_dict:
        if input_dict["transform_all"].casefold() == "true":
            transform_all = True

    # convert space_image_path to dictionary of dictionaries. (image_name-path key-values in a dict of space-img key-values)
    sip = {} # space-image-path dictionary
    for i in range(len(space_image_path)):
        if not space_image_path[i][0] in sip:
            sip[space_image_path[i][0]] = {}
        new_img = {space_image_path[i][1]: space_image_path[i][2]}
        sip[space_image_path[i][0]].update(new_img)

    print('output: ', output)
    print('space-name-path dict: ', sip)
    if 'registrations' in locals():
        print('registrations: ', registrations)
        print('configs: ', configs)

        reg_list = [] # a list of dicts specifying inputs for each registration to perform
        for i in range(len(registrations)):
            src_space = registrations[i][0][0]
            src_img = registrations[i][0][1]
            dest_space = registrations[i][1][0]
            dest_img = registrations[i][1][1]
            src_path = sip[src_space][src_img]
            dest_path = sip[dest_space][dest_img]

            reg_list.append({'registration': registrations[i], # registrsation format [[src_space, src_img], [dest_space, dest_img]]
                            'source': src_path,
                            'dest': dest_path,
                            'config': configs[i],
                            'output': output})

        print('registration list: ', reg_list, '\n')
        print('running registrations...')
        adj, spaces = run_registrations(reg_list)

        # write out adjacency list and spaces dict
        with open(os.path.join(output, 'adjacency_list.p'), 'wb') as f:
            pickle.dump(adj, f)
        with open(os.path.join(output, 'spaces_dict.p'), 'wb') as f:
            pickle.dump(spaces, f)

    if transform_all: # do every transfrom in both directions and transform all series images to registered space.
        transforms = []
        for i in sip.keys(): # for each space
            if os.path.isdir(sip[i][list(sip[i])[0]]): # if the path is a directory, then this is an image series
                '''We assume there is only one image series in the space. In the future, we plan to include
                support for multiple stains in the same histology space.'''
                dest_img = list(sip[i].keys())[0]
                transforms.append([[i,dest_img],[i,dest_img]])
            for j in sip.keys(): # for every other space
                if j == i:
                    continue
                dest_img = list(sip[j].keys())[0] # we just need any image in destination space so use the first.
                for k in sip[i].keys(): # for each image in current space
                    transforms.append([[i, k], [j, dest_img]])

    if 'transforms' in locals():
        print('transforms are: ')
        for i in transforms:
            print(str(i))
        for trans in transforms:
            src_space = trans[0][0]
            src_img = trans[0][1]
            src_path = sip[src_space][src_img]
            dest_space = trans[1][0]
            dest_img = trans[1][1]
            dest_path = sip[dest_space][dest_img] 
            apply_transformation(adj, spaces, src_space, src_img, dest_space, output,\
                                    src_path=src_path, dest_path=dest_path)

    return



#%%
if __name__ == "__main__":
    main()
