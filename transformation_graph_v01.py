import emlddmm
import numpy as np
import torch
import argparse
from argparse import RawTextHelpFormatter
import json
import os
import pickle
import matplotlib.pyplot as plt

DEBUG = False

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'
dtype = torch.float

class Graph:
    """
    graph object with nodes and edges representing image spaces and transformations between them.

    Attributes
    ----------
    spaces : dict
        Integer keys map to the space name and the domain of the corresponding image space.
    adj: list
       Adjacency list. List of dictionaries holding the transforms needed to map between connecting spaces.
    """

    def __init__(self, adj=[], spaces={}):
        self.adj = adj
        self.spaces = spaces
    

    def add_space(self, space_name, x=[]):
        v = len(self.spaces)
        if space_name not in self.spaces:
            self.spaces.update({space_name: [v, x]})


    def add_edge(self, transforms, src_space, target_space):
        #print(f'adding edge from {src_space} to {target_space}')
        #print(f'source id is {self.spaces[src_space][0]}')
        #print(f'target id is {self.spaces[target_space][0]}')
        self.adj[self.spaces[src_space][0]].update({self.spaces[target_space][0]: transforms})


    def BFS(self, src, target, v, pred, dist):
        """ Breadth first search

        a modified version of BFS that stores predecessor
        of each vertex in array pred and its distance from source in array dist

        Parameters
        ----------
        src: int
            int value given by corresponding src in spaces dict
        target: int 
            int value given by corresponding target in spaces dict
        v: int
            length of spaces dict
        pred: list of ints
            stores predecessor of vertex i at pred[i]
        dist: list of ints
            stores distance (by number of vertices) of vertex i from source vertex

        Returns
        -------
        bool
            True if a path from src to target is found and False otherwise

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
            for i in range(len(self.adj[u])):
            
                if (visited[list(self.adj[u])[i]] == False):
                    visited[list(self.adj[u])[i]] = True
                    dist[list(self.adj[u])[i]] = dist[u] + 1
                    pred[list(self.adj[u])[i]] = u
                    queue.append(list(self.adj[u])[i])
    
                    # We stop BFS when we find
                    # destination.
                    if (list(self.adj[u])[i] == target):
                        return True
    
        return False


    def shortest_path(self, src, target):
        """ Find Shortest Path

        Finds the shortest path between target and src in the adjacency list and prints its length

        Parameters
        ----------
        src: int 
            src value given by corresponding source in spaces dict
        target: int
            int value given by corresponding source in spaces dict

        Returns
        -------
        path : list of ints
            path from target to src using integer values of the adjacency list vertices. Integers can be converted to space names by the spaces dict.


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
        Given target and source are not connected

        """
        v = len(self.spaces)
        
        pred=[0 for i in range(v)] # predecessor of space i in path from target to src
        dist=[0 for i in range(v)] # distance of vertex i by number of vertices from src
    
        if (self.BFS(src, target, v, pred, dist) == False):
            print("Given target and source are not connected") # TODO: make this more informative
            print('printing source')
            print(src)
            print('printing target')
            print(target)
            print('printing spaces')
            print(self.spaces)
            print('printing adjacency')
            for i in range(len(self.adj)):
                print(i)                
                print(self.adj[i])
            
        # path stores the shortest path
        path = []
        crawl = target
        path.append(crawl)
        
        while (pred[crawl] != -1):
            path.append(pred[crawl])
            crawl = pred[crawl]
        
        path.reverse()

        # if len(path) > 1:
            # distance from source is in distance array
            # print(f"Shortest path length is: {dist[target]} \n")

        return path
    

    def transforms(self, path):
        ''' Returns a list of transforms following a path.

        Parameters
        ----------
        path : list
            A path as returned by the self.shortest_path function.
        
        Returns
        -------
        transforms : list
            A list of Transform objects defined in emlddmm.py.

        '''
        transforms = []
        for i in range(len(path)-1):
            transforms =  transforms + self.adj[path[i]][path[i+1]]

        return transforms
    

    def map_points_(self, src_space, transforms, xy_shift=None):
        # this is replaced below
        '''Applies a sequence of transforms to points in source space. If mapping to an image series, maps to the registered domain.

        Parameters
        ----------
        srs_space : str
            name of the source space
        transforms : list of emlddmm Transform objects
        xy_shift : torch Tensor
            R^2 vector. Applies a translation in xy for reconstructing in registered space. 

        Returns
        -------
        X : torch tensor
            transformed points
        
        Note
        -----
        The xy_shift optional argument is necessary when reconstructing in registered space or when the origins of the source and target space
        are far apart, e.g. one defines the origin at bregma and the other at the image center. The problem arises because the xy translation is
        contained entirely in the 2D affine transforms. 
        
        
        '''
        xI = self.spaces[src_space][1]
        xI = [torch.as_tensor(x) for x in xI]
        XI = torch.stack(torch.meshgrid(xI, indexing='ij'))
        # series of 2d transforms can only be applied to the space on which they were computed because the size must match the number of slices.
        # to avoid an error, check if a volumetric transform is followed by a 2d series transform.
        check = False
        ids = []
        for i in range(len(transforms)):
            A2d = transforms[i].data.ndim == 3
            if A2d and check:
                ids.append(i)
            elif not A2d:
                check = True
        transforms = [j for i, j in enumerate(transforms) if i not in ids]
        X = emlddmm.compose_sequence(transforms, XI)
        if xy_shift is not None:
            X[1:] += xy_shift[...,None,None,None]
        return X
    

    def map_points(self, src_space, transforms, xy_shift=None, slice_locations=None):
        # this is reimplemented
        '''Applies a sequence of transforms to points in source space. 

        Parameters
        ----------
        srs_space : str
            name of the source space
        transforms : list of emlddmm Transform objects
        xy_shift : torch Tensor
            R^2 vector. Applies a translation in xy for reconstructing in registered space. 

        Returns
        -------
        X : torch tensor
            transformed points
        
        Note
        -----
        The xy_shift optional argument is necessary when reconstructing in registered space or when the origins of the source and target space
        are far apart, e.g. one defines the origin at bregma and the other at the image center. The problem arises because the xy translation is
        contained entirely in the 2D affine transforms. 
        
        Note
        ----
        From daniel.  Mapping from 2D slices to anything is fine.  The shape will always be the 2D slices.  This is used when mapping imaging data to a 2D slice.
        Mapping to a 2D slice is harder. But I think we can do it somehow with nearest neighbor interpolation.
        
        '''
        xI = self.spaces[src_space][1]
        xI = [torch.as_tensor(x) for x in xI]
        XI = torch.stack(torch.meshgrid(xI, indexing='ij'))        
        X = torch.as_tensor(XI.clone(),dtype=transforms[0].data.dtype)  # I do not want to share data, still this prints a warning
        for t in transforms:
            # let's check for special cases
            if t.data.ndim == 3:
                #print('this is a series of 2d affine')
                if X.shape[1] == t.data.shape[0]:
                    #print('shapes are compatible')
                    X = t.apply(X)
                else:

                    # TODO (done)
                    # for each z coordinate, find the closest slice
                    # then map the xy coordinates based on the matrix for their closest slice
                    # also, snap the z coordinate exactly to the slice, I think this will be necessary for interpolation      
                    # this will happen when mapping imaging data from a 2d space
                    # in this case we need slice_locations as an optional argument
                    #               
                    print('*'*80)
                    print('shapes not compatible')
                    if slice_locations is None:
                        print('skipping for now because there are no slice locations')
                        continue
                    else:
                        print('using slice snapping')
                    
                    X0ind = torch.round( (X[0] - slice_locations[0])/(slice_locations[1] - slice_locations[0]) ).int() # the slice coordinate
                    X0ind[X0ind<0]=0
                    X0ind[X0ind>=len(slice_locations)] = len(slice_locations)-1
                    Xnew = X.clone()
                    for i in range(len(slice_locations)):
                        ind = X0ind == i
                        X12 = (t.data[i,:2,:2]@(X[1:,ind])) + t.data[i,:2,-1,None]                        
                        X0 = slice_locations[i]
                        # assign
                        Xnew[0,ind] = X0
                        Xnew[1:,ind] = X12
                        X = Xnew

            else:
                # simple case
                X = t.apply(X)
                                            
                    

        # not doing this
        #X = emlddmm.compose_sequence(transforms, XI)
        if xy_shift is not None:
            X[1:] += xy_shift[...,None,None,None]
        return X    


    def map_image_(self, src_space, image, target_space, transforms, xy_shift=None):
        # this one is obsolte
        '''Map an image from source space to target space.

        Parameters
        ----------
        src_space : str
            name of source space
        image : array
        target_space : str
            name of target space
        transforms : list of emlddmm Transform objects
        xy_shift : torch Tensor
            R^2 vector. Applies a translation in xy for reconstructing in registered space.

        Returns
        -------
        image : array
            transformed image data

        Note
        -----
        The xy_shift optional argument is necessary when reconstructing in registered space or when the origins of the source and target space
        are far apart, e.g. one defines the origin at bregma and the other at the image center. The problem arises because the xy translation is
        contained entirely in the 2D affine transforms
        '''
        # if the last transforms are 2d series affine, then we will first apply them to the target space and resample the target image in registered space.
        # then apply the other transforms to the source space and resample the registered target image.
        ids = []
        A2d = []
        for i in reversed(range(len(transforms))):
            if transforms[i].data.ndim == 3:
                ids.append(i)
                A2d.insert(0,transforms[i])
            else:
                break
        transforms = [j for i, j in enumerate(transforms) if i not in ids]
        xI = self.spaces[src_space][1]
        xI = [torch.as_tensor(x) for x in xI]
        if len(A2d) > 0:
            image = torch.as_tensor(image)
            XI = torch.stack(torch.meshgrid(xI, indexing='ij'))
            if xy_shift is not None:
                XI[1:] -= xy_shift[...,None,None,None]
            XR = emlddmm.compose_sequence(A2d, XI)
            image = emlddmm.interp(xI,image, XR)
        # any 2D affine transforms at the end of the sequence will be ignored by map_points
        if len(transforms) > 0:
            X = self.map_points(target_space, transforms, xy_shift=xy_shift)
            image = emlddmm.interp(xI, image, X)

        return image
    def map_image(self, src_space, image, target_space, transforms, xy_shift=None, **kwargs):
        '''Map an image from source space to target space.

        Parameters
        ----------
        src_space : str
            name of source space
        image : array
        target_space : str
            name of target space
        transforms : list of emlddmm Transform objects
        xy_shift : torch Tensor
            R^2 vector. Applies a translation in xy for reconstructing in registered space.
        kwargs : dict
            keword args to be passed to emlddmm interpolation, which will be passed to torch grid sample
        Returns
        -------
        image : array
            transformed image data

        Note
        -----
        The xy_shift optional argument is necessary when reconstructing in registered space or when the origins of the source and target space
        are far apart, e.g. one defines the origin at bregma and the other at the image center. The problem arises because the xy translation is
        contained entirely in the 2D affine transforms
        '''
        
        xI = self.spaces[src_space][1]
        xI = [torch.as_tensor(x) for x in xI]
        image = torch.as_tensor(image)
        if isinstance(transforms,list):
            X = self.map_points(target_space, transforms, xy_shift=xy_shift)    
        else:
            X = transforms
        image = emlddmm.interp(xI, image, X,**kwargs)

        return image


    def merge(self, new_graph):
        ''' Merge two graphs

        Parameters
        ----------
        new_graph : emlddmm Graph object

        Returns
        -------
        graph : emlddmm Graph object
            Current graph merged with the new graph.
        '''
        graph = self
        # merge spaces dict
        id_map = {} # we need a dict to map new space indices to the existing ones
        for key, value in new_graph.spaces.items():
            if key in graph.spaces:
                id_map.update({value[0]:graph.spaces[key][0]})
            else:
                id_map.update({value[0]:len(graph.spaces)})
                value[0] = len(graph.spaces)
                graph.spaces.update({key:value})
                graph.adj.append({}) # add a node in the adjacency list for each new space

        # merge adjacency list
        for i, node in enumerate(new_graph.adj): # for each node in the graph
            src = id_map[i] # get the original index for the space
            for j in node: # and for each node to which it connects
                target = id_map[j]
                transform = node[j]
                graph.adj[src].update({target:transform}) # update the graph

        return graph


def graph_reconstruct_(graph, out, I, target_space, target_fnames=[]):
    # this version is obsolete
    ''' Apply Transformation

    Applies affine matrix and velocity field transforms to map source points to target points. Saves displacement field from source points to target points
    (i.e. difference between transformed coordinates and input coordinates), and determinant of Jacobian for 3d source spaces. Also saves transformed image in vtk format.

    Parameters
    ----------
    graph : emlddmm Graph object
    out: str
        path to registration outputs parent directory
    I : emlddmm Image   
    target_space : str
        name of the space to which image I will be transformed.
    target_fnames : list
        list of file names; only necessary if target is a series of 2d slices.
        
    TODO
    ----
    Check why the registered space histology is not working. (march 27, 2023)
    I think the issue is that there is actually no time to do it.
    If I say to reconstruct one space to itself, then it says not connected and gives an error.  
    There needs to be another way.

    '''
    jacobian = lambda X,dv : np.stack(np.gradient(X, dv[0],dv[1],dv[2], axis=(1,2,3))).transpose(2,3,4,0,1)

    dtype = torch.float
    device = 'cpu'
    # convert data to torch
    # J.x = [torch.as_tensor(x, dtype=dtype, device=device) for x in J.x]
    # J.data = torch.as_tensor(J.data, dtype=dtype, device=device)

    # first we get the sample points in the target space J
    xJ = graph.spaces[target_space][1]
    xJ = [torch.as_tensor(x, dtype=dtype, device=device) for x in xJ]
    target_space_idx = graph.spaces[target_space][0]

    # then we get the sample points in the source space I
    I.x = [torch.as_tensor(x, dtype=dtype, device=device) for x in I.x]
    I.data = torch.as_tensor(I.data, dtype=dtype, device=device)
    src_space_idx = graph.spaces[I.space][0]

    # backward transform, map the points in target space J back to the source space
    path = graph.shortest_path(target_space_idx, src_space_idx)
    transforms = graph.transforms(path)
    XJ = torch.stack(torch.meshgrid(xJ, indexing='ij'))    
    fXJ = graph.map_points(target_space, transforms)
    # and then transform the image by sampling it at these points
    fI = graph.map_image(I.space, I.data, target_space, transforms)

    # now we are going to write the outputs.  This involves several different cases
    '''
    Three cases:

    1) series to series
        Save reconstructions of I slices in J space. 
        Apply A2di to J points and resample I.
    2) volume to series
        A) Save volume to registered images in {target_space}_registered/{I.space}_{I.name}_to_{target_space}_registered/images/,
        and volume to input images in {target_space}_input/{I.space}_{I.name}_to_{target_space}_input/images/
        B) Save volume to registered and volume to input displacement in {target_space}_registered/{I.space}_{I.name}_to_{target_space}_registered/transforms/ and 
        {target_space}_input/{I.space}_{I.name}_to_{target_space}_input/transforms/, respectively.
    3) series to volume
        A) save series input to registered space images in {I.space}_registered/{I.space}_{I.name}_input_to_{I.space}_registered/images/
        C) Save series to volume image in {target_space}/{I.space}_{I.name}_input_to_{I.space}/images/
        D) Save series to volume detjac and displacement in {target_space}/{I.space}_{I.name}_registered_to_{target_space}/transforms/
    4) volume to volume 
        A) Save out I to J image in {target_space}/{I.space}_to_{target_space}/images.
        B) Save out I to J detjac and displacement in {target_space}/{I.space}_to_{target_space}/transforms/
    '''
    from_series = I.title == 'slice_dataset'
    to_series = len(target_fnames) != 0 # we don't have the image title so we need to check for a list of file names
    if from_series and to_series:
        print(f'reconstructing {I.space} {I.name} in {target_space} space')
        # series to series
        # Assumes J and I have the same space dimensions
        I_to_J_out = os.path.join(out, f'{target_space}_input/{I.space}_{I.name}_input_to_{target_space}_input/images/')
        if not os.path.exists(I_to_J_out):
            os.makedirs(I_to_J_out)
        for i in range(I.data.shape[1]):
            x = [[I.x[0][i], I.x[0][i]+10], I.x[1], I.x[2]]
            # I to J
            img = fI[:, i, None, ...]
            title = f'{I.space}_input_{I.fnames()[i]}_to_{target_space}_input_{target_fnames[i]}'
            emlddmm.write_vtk_data(os.path.join(I_to_J_out, f'{I.space}_input_{I.fnames()[i]}_to_{target_space}_input_{target_fnames[i]}.vtk'), x, img, title)

    elif to_series:
        print(f'reconstructing {I.space} {I.name} in {target_space} space')
        # volume to series
        # we need I transformed to J registered space
        path = graph.shortest_path(graph.spaces[target_space][0], graph.spaces[I.space][0])
        # omit the first 2d series transforms (R^-1) which takes points from 2d to 2d or input to registered.
        # in this case we can just remove all 2d series transforms.
        transforms = graph.transforms(path)
        for i, t in enumerate(transforms):
            if t.data.ndim == 3:
                mean_translation = torch.mean(t.data[:,:2,-1], dim=0)
                del transforms[i]
        phiiAiXJ = graph.map_points(target_space, transforms, xy_shift=mean_translation)
        AphiI = graph.map_image(I.space, I.data, target_space, transforms, xy_shift=mean_translation)
        # get I to J registered and I to J input displacements
        reg_disp = (phiiAiXJ - XJ)[None]
        input_disp = (fXJ - XJ)[None]

        # setup output paths
        I_to_Ji_out = os.path.join(out, f'{target_space}_input/{I.space}_{I.name}_to_{target_space}_input/images/')
        if not os.path.exists(I_to_Ji_out):
            os.makedirs(I_to_Ji_out)
        I_to_Jr_out = os.path.join(out, f'{target_space}_registered/{I.space}_{I.name}_to_{target_space}_registered/images/')
        if not os.path.exists(I_to_Jr_out):
            os.makedirs(I_to_Jr_out)
        reg_disp_out = os.path.join(out, f'{target_space}_registered/{I.space}_{I.name}_to_{target_space}_registered/transforms/')
        if not os.path.exists(reg_disp_out):
            os.makedirs(reg_disp_out)
        input_disp_out = os.path.join(out, f'{target_space}_input/{I.space}_{I.name}_to_{target_space}_input/transforms/')
        if not os.path.exists(input_disp_out):
            os.makedirs(input_disp_out)
        for i in range(len(xJ[0])):
            x = [[xJ[0][i], xJ[0][i]+10], xJ[1], xJ[2]]
            # volume to input series
            # save image
            img = fI[:, i, None, ...]
            title = f'{I.space}_{I.name}_to_{target_space}_input_{target_fnames[i]}'
            emlddmm.write_vtk_data(os.path.join(I_to_Ji_out, title + '.vtk'), x, img, title)
            # save displacement
            disp = input_disp[:, :, i, None]
            title = f'{target_space}_input_{target_fnames[i]}_to_{I.space}_displacement'
            emlddmm.write_vtk_data(os.path.join(input_disp_out, title + '.vtk'), x, disp, title)
            # volume to registered series
            img = AphiI[:, i, None, ...]
            title = f'{I.space}_{I.name}_to_{target_space}_registered_{target_fnames[i]}'
            emlddmm.write_vtk_data(os.path.join(I_to_Jr_out, title + '.vtk'), x, img, title)
            # save displacement
            disp = reg_disp[:, :, i, None]
            title = f'{target_space}_registered_{target_fnames[i]}_to_{I.space}_displacement'
            emlddmm.write_vtk_data(os.path.join(reg_disp_out, title + '.vtk'), x, disp, title)

    
    elif from_series:
        print(f'reconstructing {I.space} {I.name} in {target_space} space')
        # series to volume
        # I input to I registered space
        path = graph.shortest_path(graph.spaces[target_space][0], graph.spaces[I.space][0])
        # get the last 2d series transforms (R) which take points from 2d to 2d or registered to input
        transforms = graph.transforms(path)
        idx = 0
        for i,t in enumerate(transforms[::-1]):
            if t.data.ndim != 3:
                idx = i
                break
        A2ds = transforms[-idx:]
        mean_translation = torch.mean(A2ds[0].data[:,:2,-1], dim=0)
        RiI = graph.map_image(I.space, I.data, I.space, A2ds, xy_shift=mean_translation)

        Ii_to_Ir_out = os.path.join(out, f'{I.space}_registered/{I.space}_input_to_{I.space}_registered/images/')
        if not os.path.exists(Ii_to_Ir_out):
            os.makedirs(Ii_to_Ir_out)

        # input to registered images
        img = RiI[:, i, None, ...]
        title = f'{I.space}_input_{I.fnames()[i]}_to_{I.space}_registered_{I.fnames()[i]}'
        emlddmm.write_vtk_data(os.path.join(Ii_to_Ir_out, title + '.vtk'), I.x, img, title)

        # I to J
        img = graph.map_image(I.space, I.data, target_space, transforms, xy_shift=mean_translation)
        title = f'{I.space}_{I.name}_input_to_{target_space}'
        I_to_J_imgs = os.path.join(out, f'{target_space}/{I.space}_{I.name}_input_to_{target_space}/images/')
        if not os.path.exists(I_to_J_imgs):
            os.makedirs(I_to_J_imgs)
        emlddmm.write_vtk_data(os.path.join(I_to_J_imgs, title + '.vtk'), xJ, img, title)
        # disp
        # we need J to I registered points
        disp = (fXJ - XJ)[None]
        title = f'{I.space}_{I.name}_registered_to_{target_space}_displacement'
        I_to_J_transforms = os.path.join(out, f'{target_space}/{I.space}_{I.name}_registered_to_{target_space}/transforms/')
        if not os.path.exists(I_to_J_transforms):
            os.makedirs(I_to_J_transforms)
        emlddmm.write_vtk_data(os.path.join(I_to_J_transforms, title + '.vtk'), xJ, disp, title)
        # determinant of jacobian
        dv = [(x[1]-x[0]) for x in xJ]
        jac = jacobian(fXJ, dv)
        detjac = np.linalg.det(jac)[None]
        title = f'{I.space}_{I.name}_registered_to_{target_space}_detjac'
        emlddmm.write_vtk_data(os.path.join(I_to_J_transforms, title + '.vtk'), xJ, detjac, title)
    else: # this is the volume to volume case
        print(f'reconstructing {I.space} {I.name} in {target_space} space')
        # volume to volume
        # I to J
        # save image
        img = fI
        title = f'{I.space}_{I.name}_to_{target_space}'
        I_to_J_imgs = os.path.join(out, f'{target_space}/{I.space}_{I.name}_to_{target_space}/images/')        
        if not os.path.exists(I_to_J_imgs):
            os.makedirs(I_to_J_imgs)
        emlddmm.write_vtk_data(os.path.join(I_to_J_imgs, title + '.vtk'), xJ, img, title)
        # save displacement
        disp = (fXJ - XJ)[None]
        title = f'{I.space}_{I.name}_to_{target_space}_displacement'
        I_to_J_transforms = os.path.join(out, f'{target_space}/{I.space}_{I.name}_to_{target_space}/transforms/')
        if not os.path.exists(I_to_J_transforms):
            os.makedirs(I_to_J_transforms)
        emlddmm.write_vtk_data(os.path.join(I_to_J_transforms, title + '.vtk'), xJ, disp, title)
        # save determinant of jacobian
        dv = [(x[1]-x[0]) for x in xJ]
        jac = jacobian(fXJ, dv)
        detjac = np.linalg.det(jac)[None]
        title = f'{I.space}_{I.name}_to_{target_space}_detjac'
        emlddmm.write_vtk_data(os.path.join(I_to_J_transforms, title + '.vtk'), xJ, detjac, title)


def graph_reconstruct(graph, out, I, target_space, target_fnames=[]):
    # this version is modified by daniel and does not treat "registered as a special case"
    # todo, transform all images.  every time you transform an image, you also write it out, and write out the transform
    # if you transform an annotation image to a slice dataset, you should also output geojson
    # in this case we'll have to find an ontology to work with cshl
    ''' Apply Transformation

    Applies affine matrix and velocity field transforms to map source points to target points. Saves displacement field from source points to target points
    (i.e. difference between transformed coordinates and input coordinates), and determinant of Jacobian for 3d source spaces. Also saves transformed image in vtk format.

    Parameters
    ----------
    graph : emlddmm Graph object
    out: str
        path to registration outputs parent directory
    I : emlddmm Image   
    target_space : str
        name of the space to which image I will be transformed.
    target_fnames : list
        list of file names; only necessary if target is a series of 2d slices.
        
    TODO
    ----
    Check why the registered space histology is not working. (march 27, 2023)
    I think the issue is that there is actually no time to do it.
    If I say to reconstruct one space to itself, then it says not connected and gives an error.  
    There needs to be another way.

    '''
    jacobian = lambda X,dv : np.stack(np.gradient(X, dv[0],dv[1],dv[2], axis=(1,2,3))).transpose(2,3,4,0,1)

    dtype = torch.float
    device = 'cpu'


    
    print(f'about to transform {I.space}, {I.title} to the space {target_space}')
    

    # now we are going to write the outputs.  This involves several different cases
    '''
    Three cases:

    1) series to series
        Save reconstructions of I slices in J space. 
        Apply A2di to J points and resample I.
    2) volume to series
        A) Save volume to registered images in {target_space}_registered/{I.space}_{I.name}_to_{target_space}_registered/images/,
        and volume to input images in {target_space}_input/{I.space}_{I.name}_to_{target_space}_input/images/
        B) Save volume to registered and volume to input displacement in {target_space}_registered/{I.space}_{I.name}_to_{target_space}_registered/transforms/ and 
        {target_space}_input/{I.space}_{I.name}_to_{target_space}_input/transforms/, respectively.
    3) series to volume
        A) save series input to registered space images in {I.space}_registered/{I.space}_{I.name}_input_to_{I.space}_registered/images/
        C) Save series to volume image in {target_space}/{I.space}_{I.name}_input_to_{I.space}/images/
        D) Save series to volume detjac and displacement in {target_space}/{I.space}_{I.name}_registered_to_{target_space}/transforms/
    4) volume to volume 
        A) Save out I to J image in {target_space}/{I.space}_to_{target_space}/images.
        B) Save out I to J detjac and displacement in {target_space}/{I.space}_to_{target_space}/transforms/
    '''
    from_series = I.title == 'slice_dataset'
    to_series = len(target_fnames) != 0 # we don't have the image title so we need to check for a list of file names
    print(f'Is the source a series? {from_series}')
    print(f'Is the target a series? {to_series}')

    # convert data to torch
    # J.x = [torch.as_tensor(x, dtype=dtype, device=device) for x in J.x]
    # J.data = torch.as_tensor(J.data, dtype=dtype, device=device)

    # first we get the sample points in the target space J
    xJ = graph.spaces[target_space][1]
    xJ = [torch.as_tensor(x, dtype=dtype, device=device) for x in xJ]
    target_space_idx = graph.spaces[target_space][0]

    # then we get the sample points in the source space I            
    src_space_idx = graph.spaces[I.space][0]

    # backward transform, map the points in target space J back to the source space
    path = graph.shortest_path(target_space_idx, src_space_idx)
    transforms = graph.transforms(path)
    if DEBUG:
        print(transforms)
        for t in transforms:
            if t.data.shape == torch.Size([4,4]):
                print(t.data)        

    XJ = torch.stack(torch.meshgrid(xJ, indexing='ij'))    
    if from_series and not to_series:
        # special case for mapping 2D data to 3D
        # we need to snap onto grid points
        #         
        # note in the case of atlas to registered space
        # there is no issue, since the transforms are just v A
        # the slice locations argument will be ignored        
        fXJ = graph.map_points(target_space, transforms, slice_locations=I.x[0])

    else:
        fXJ = graph.map_points(target_space, transforms)
    
    # and then transform the image by sampling it at these points
    if I.data is not None:
        I.x = [torch.as_tensor(x, dtype=dtype, device=device) for x in I.x]
        Idtype = I.data.dtype
        I.data = torch.as_tensor(I.data, dtype=dtype, device=device)
        
        if I.annotation:            
            print('found annotation, using nearest')
            fI = graph.map_image(I.space, I.data, target_space, fXJ, mode='nearest')
            # convert the dtype back
            if Idtype == np.uint8:
                fI = torch.as_tensor(fI,dtype=torch.uint8)
            elif Idtype == np.uint16:
                fI = torch.as_tensor(fI,dtype=torch.uint16) # I think this doesn't exist
            elif Idtype == np.uint32:
                fI = torch.as_tensor(fI,dtype=torch.uint32) 
        else:
            fI = graph.map_image(I.space, I.data, target_space, fXJ)
        
        # write out a qc file        
        fig,ax = emlddmm.draw(fI,xJ)
        qc_out = os.path.join(out, f'{target_space}/{I.space}_to_{target_space}/qc/')
        os.makedirs(qc_out, exist_ok=True)
        fig.savefig(os.path.join(qc_out,f'{I.space}_{I.name}_to_{target_space}_full.jpg'))
        
    else:
        fI = None

    # daniel asks, does the above map points and images work in all cases? 

    
    
    
    
    if from_series and to_series:
        print('This is a 2D series to a 2D series')
        # this one is pretty straight forward
        # we write out transforms and we write out images
        # series to series
        # Assumes J and I have the same space dimensions
        I_to_J_out = os.path.join(out, f'{target_space}/{I.space}_to_{target_space}/images/')
        reg_disp_out = os.path.join(out, f'{target_space}/{I.space}_to_{target_space}/transforms/')
        
        if fI is not None:
            os.makedirs(I_to_J_out, exist_ok=True)
        
        os.makedirs(reg_disp_out, exist_ok=True)
        for i in range(len(xJ[0])):#range(I.data.shape[1]): # note the number of slices in I and J must be equal
            
            # I to J
            if fI is not None: # if there are no images, don't write them
                x = [[I.x[0][i], I.x[0][i]+10], I.x[1], I.x[2]]
                img = fI[:, i, None, ...]
                title = f'{I.space}_{I.name}_{I.fnames()[i]}_to_{target_space}_{target_fnames[i]}'
                emlddmm.write_vtk_data(os.path.join(I_to_J_out, f'{I.space}_{I.name}_{I.fnames()[i]}_to_{target_space}_{target_fnames[i]}.vtk'), x, img, title)
            # also write out transforms as a matrix
            #print(transforms)
            if np.all([t.data.ndim==3 for t in transforms]):
                #print('All transforms are matrices')
                #print(f'{transforms[0].data.shape}')
                output = transforms[0].data[i].clone()
                for t in transforms[1:]:
                    output = t.data[i]@output
                # TODO: something is wrong with this path, come back to it later
                output_transform_name = os.path.join(reg_disp_out,f'{target_space}_{target_fnames[i]}_to_{I.space}_{I.fnames()[i]}_matrix.txt')
                print(output_transform_name)
                emlddmm.write_matrix_data(output_transform_name, output)
                
            else:
                # write out displacement fields                
                print('writing displacement fields for 2d to 2d not implement yet')
                asdf
                pass
        
    elif to_series:
        print('This is 3D to 2D')
        # recall I don't need special cases for registered space anymore
        reg_disp = (fXJ - XJ)[None] # recall we output a 1x3xslicexrowxcol
        # get the output 


        # setup output paths
        I_to_J_out = os.path.join(out, f'{target_space}/{I.space}_to_{target_space}/images/')        
        os.makedirs(I_to_J_out, exist_ok=True)        
        reg_disp_out = os.path.join(out, f'{target_space}/{I.space}_to_{target_space}/transforms/')        
        os.makedirs(reg_disp_out, exist_ok=True)

        if I.annotation:
            #print('*'*80)
            #print('This is an annotation, we will make geojson outputs')
            geojson_out = os.path.join(out, f'{target_space}/{I.space}_to_{target_space}/geojson/')
            os.makedirs(geojson_out, exist_ok=True)
        
        qc_out = os.path.join(out, f'{target_space}/{I.space}_to_{target_space}/qc/')
        os.makedirs(qc_out, exist_ok=True)

        fig,ax = plt.subplots()
        for i in range(len(xJ[0])): # loop over all slices
            ax.cla()
            x = [[xJ[0][i], xJ[0][i]+10], xJ[1], xJ[2]] #TODO this +10 is hard coded just to make it work in 3D, it's not actually slice thickness

            # save image
            img = fI[:, i, None, ...]
            title = f'{I.space}_{I.name}_to_{target_space}_{target_fnames[i]}'
            emlddmm.write_vtk_data(os.path.join(I_to_J_out, title + '.vtk'), x, img, title)
            # save displacement
            disp = reg_disp[:, :, i, None]
            title = f'{target_space}_{target_fnames[i]}_to_{I.space}_displacement'
            emlddmm.write_vtk_data(os.path.join(reg_disp_out, title + '.vtk'), x, disp, title)

            # TODO: for qc I would need to get the target fnames and load them
            # TODO: we need to load an ontology
            if I.annotation:                
                output_geojson = {'type': 'FeatureCollection', 'features': []}
                # generate geojson curves for each label
                labels = np.unique(img.cpu().numpy())
                
                count = 0
                for l in labels[1:]: # ignore background label
                    coordinates = [] # one set of coordinates per label
                    cs = ax.contour(xJ[-1],xJ[-2],(img.cpu().numpy()[0,0]==l).astype(float),[0.5],linewidths=1.0,colors='k')
                    paths = cs.collections[0].get_paths()
                    for path in paths:
                        vertices = np.array([seg[0] for seg in path.iter_segments()])
                        meanpos = (np.max(vertices,0) + np.min(vertices,0))/2
                        # put some text
                        if vertices.shape[0] > 20:
                            ax.text(meanpos[0],meanpos[1],str(l),
                                    horizontalalignment='center', verticalalignment='center',
                                    fontsize=4, bbox={'color':np.array([1.0,1.0,1.0,0.5]), 'pad':0})
                        coordinates.append([[ list(seg[0]) for seg in path.iter_segments()]])
                    # TODO get an actual ontology
                    geometry = {'type': 'MultiPolygon', 'coordinates': coordinates}
                    
                    # TODO: for the first feature add something to properties that says how big the suggested image is
                    # it should be a dictionary like 'suggested_image':{'n':[],'o':[],'d':[]}
                    # we will use the convention that we go from the first pixel to the last pixel i
                    
                    properties = {'name':str(l), 'acronym':str(l)}
                    if count == 0:
                        # what does the 32x upsampled space look like?
                        # each pixel becomes 32 pixels, centered at the same spot
                        nup = 32
                        offsets = (np.arange(nup) - (nup-1)/2)/nup
                        dJ = np.array([np.array(x[1] - x[0]) for x in xJ])
                        
                        tmp0 = (np.array(xJ[-2][...,None]) + offsets*np.array(dJ[-2])).reshape(-1)
                        tmp1 = (np.array(xJ[-1][...,None]) + offsets*np.array(dJ[-1])).reshape(-1)
                        xup = [tmp0,tmp1]
                        properties['suggested_image'] = {'n':[len(x) for x in xup],
                                                         'd':[dJ[1].item()/nup,dJ[2].item()/nup],
                                                         'o':[xup[0][0].item(),xup[1][0].item()]}                        
                    output_geojson['features'].append({'type': 'Feature', 'id': int(l), 'properties': properties, 'geometry': geometry})

                    
                    count += 1                
                with open(os.path.join(geojson_out,f'{I.space}_{I.name}_to_{target_space}_{target_fnames[i]}.geojson'),'wt') as f:
                    json.dump(output_geojson, f, indent=2)
                fig.savefig(os.path.join(qc_out,f'{I.space}_{I.name}_to_{target_space}_{target_fnames[i]}.jpg'))
        plt.close(fig)
                    

    elif from_series:
        print(f'reconstructing {I.space} {I.name} in {target_space} space')
        print('This is mapping 2D images to a 3D space,')
        
        # again, we don't do anything special here with introducing new spaces        
        # we just write out fI and FXJ
        reg_disp = (fXJ - XJ)[None]                

        # set up the paths
        I_to_J_out = os.path.join(out, f'{target_space}/{I.space}_to_{target_space}/images/')
        if fI is not None:
            os.makedirs(I_to_J_out, exist_ok=True)
        reg_disp_out = os.path.join(out, f'{target_space}/{I.space}_to_{target_space}/transforms/')
        os.makedirs(reg_disp_out, exist_ok=True)

        # write out image
        if fI is not None:
            title = f'{I.space}_{I.name}_to_{target_space}'
            emlddmm.write_vtk_data(os.path.join(I_to_J_out, title + '.vtk'), xJ, fI, title)

        # write out transform, note I have modified map points so these will point exactly at a slice, so that we can do matrix multiplication, if necessary
        title = f'{target_space}_to_{I.space}_displacement'
        emlddmm.write_vtk_data(os.path.join(reg_disp_out, title + '.vtk'), xJ, reg_disp, title)

    else: # this is the volume to volume case
        print(f'reconstructing {I.space} {I.name} in {target_space} space')
        print(f'This is mapping a 3D image to a 3D image')
        # question, does the registered space count as a 3D image? No, not for our purposes
        
        
        # volume to volume
        # I to J
        # save image
        img = fI
        title = f'{I.space}_{I.name}_to_{target_space}'
        I_to_J_imgs = os.path.join(out, f'{target_space}/{I.space}_to_{target_space}/images/')                
        os.makedirs(I_to_J_imgs, exist_ok=True)
        emlddmm.write_vtk_data(os.path.join(I_to_J_imgs, title + '.vtk'), xJ, img, title)
        # save displacement
        disp = (fXJ - XJ)[None]
        title = f'{I.space}_to_{target_space}_displacement'
        I_to_J_transforms = os.path.join(out, f'{target_space}/{I.space}_to_{target_space}/transforms/')
        os.makedirs(I_to_J_transforms, exist_ok=True)
        emlddmm.write_vtk_data(os.path.join(I_to_J_transforms, title + '.vtk'), xJ, disp, title)
        # save determinant of jacobian
        dv = [(x[1]-x[0]) for x in xJ]
        jac = jacobian(fXJ, dv)
        detjac = np.linalg.det(jac)[None]
        title = f'{I.space}_{I.name}_to_{target_space}_detjac'
        emlddmm.write_vtk_data(os.path.join(I_to_J_transforms, title + '.vtk'), xJ, detjac, title)


def run_registrations(reg_list):
    """ Run Registrations

    Runs a sequence of registrations given by reg_list. Saves transforms, qc images, reconstructed images,
    displacement fields, and determinant of Jacobian of displacements. Also builds and writes out the transform graph. 

    Parameters
    ----------
    reg_list : list of dicts
        each dict in reg_list specifies the source image path, target image path,
        source and target space names, registration configuration settings, and output directory.
  

    
    Returns
    -------
    reg_graph : emlddmm graph

    Example
    -------
    >>> reg_list = [{'registration':[['CCF','average_template_50'],['MRI','masked']],
                     'source': '/path/to/average_template_50.vtk',
                     'target': '/path/to/HR_NIHxCSHL_50um_14T_M1_masked.vtk',
                     'config': '/path/to/configMD816_MR_to_CCF.json',
                     'output': 'outputs/example_output'},
                    {'registration':[['MRI','masked'], ['HIST','Nissl']],
                     'source': '/path/to/HR_NIHxCSHL_50um_14T_M1_masked.vtk',
                     'target': '/path/to/MD816_STIF',
                     'config': '/path/to/configMD816_Nissl_to_MR.json',
                     'output': 'outputs/example_output'}]
    >>> run_registrations(reg_list)

    """

    # initialize graph
    # note from daniel
    # for 2D registration, the spaces will just say "histology".  Is that good enough?
    graph = Graph()
    for i in reg_list:
            for j in [i['registration'][0][0], i['registration'][1][0]]: # for src and target space names in each registration
                if j not in graph.spaces: 
                    print(f'adding space {j} to graph')
                    graph.add_space(j)
    graph.adj = [{} for i in range(len(graph.spaces))]

    # perform registrations
    for r in reg_list:
        source = r['source']
        target = r['target']
        registration = r['registration']
        config = r['config']
        output_dir = r['output']
        print(f"registering {source} to {target}")
        with open(config) as f:
            config = json.load(f)
        I = emlddmm.Image(space=registration[0][0], name=registration[0][1], fpath=source)
        print(f'Source I shape {I.data.shape}')
        if I.title == 'slice_dataset': # if series to series, both images must share the same coordinate grid
            J = emlddmm.Image(space=registration[1][0], name=registration[1][1], fpath=target, mask=True, x=I.x)
        else:
            J = emlddmm.Image(space=registration[1][0], name=registration[1][1], fpath=target, mask=True)
        print(f'Target J shape {J.data.shape}')
        # add domains to graph before downsampling for later
        graph.spaces[I.space][1] = I.x
        graph.spaces[J.space][1] = J.x
        # initial downsampling
        downIs = config['downI']
        downJs = config['downJ']
        mindownI = np.min(np.array(downIs),0)
        mindownJ = np.min(np.array(downJs),0)
        I.x, I.data, I.mask = I.downsample(mindownI)
        J.x, J.data, J.mask = J.downsample(mindownJ)
        downIs = [ list((np.array(d)/mindownI).astype(int)) for d in downIs]
        downJs = [ list((np.array(d)/mindownJ).astype(int)) for d in downJs]
        # update our config variable
        config['downI'] = downIs
        config['downJ'] = downJs
        # registration
        output = emlddmm.emlddmm_multiscale(I=I.data,xI=[I.x],J=J.data,xJ=[J.x],W0=J.mask,full_outputs=False,**config)
        '''
        for series to series:
            1) Save rigid transforms in {source}_input/{target}_to_{source}_input
            2) Save  qc source input and target to source_input in {source}_input/{target}_to_{source}_input/qc/
            3) Save qc target input and source to target_input in {target}_input/{source}_to_{target}_input/qc/

        for volume (source) to series (target):
            1) Save rigid transforms (R) in {target}_registered/{target}_input_to_registered/transforms/
            2) A.txt and velocity.vtk map points in source to match target, which are
            used to resmaple target images in source space. Save them in {source}/{target}_registered_to_{source}/transforms/
            3) Save qc source original and target to source in {source}/{target}_registered_to_{source}/qc/
            4) Save qc target_input and source to target_input in {target}_input/{source}_to_{target}_input/qc/
            5) Save qc target_registered and source to target_registered in {target}_registered/{source}_to_{target}_registered/qc/

        for volume to volume:
            1) Save A.txt and velocity.vtk in {source}/{target}_to_{source}/transforms/
        '''
        #print('about to write transformation outputs')
        emlddmm.write_transform_outputs(output_dir, output[-1], I, J)
        # TODO: check if there are annotations in this space, if so add them below
        emlddmm.write_qc_outputs(output_dir, output[-1], I, J)

        A = emlddmm.Transform(output[-1]['A'], 'f')
        Ai = emlddmm.Transform(output[-1]['A'], 'b')
        xv = output[-1]['xv']
        v = output[-1]['v']
        phi = emlddmm.Transform(v, direction='f', domain=xv)
        phii = emlddmm.Transform(v, direction='b', domain=xv)
        if 'A2d' in output[-1]:
            A2d = emlddmm.Transform(output[-1]['A2d'], 'f')
            A2di = emlddmm.Transform(output[-1]['A2d'], 'b')

            # check if this is a 3D to 2D map
            if J.title == 'slice_dataset' and I.title != 'slice_dataset':
                # NOTE from daniel
                # what I want to do here is create a new space, called nissl_registered
                # when I define the registered space I'll have to define the coordinates, which will be shifted using the mean shift
                # and add two sets of edges one containing only the A2d

                print('This is a 3D to 2D map')
                print(f'From space {I.space} to space {J.space}')
                print(f'Adding a new registered space {J.space}_registered')
                registered_space = J.space+'_registered'

                # if the mean translation from registered to input
                # is 5 pixels up
                # then we want the sample points in registered space to be 5 pixels down
                mean_translation = torch.mean(A2d.data[:,:2,-1], dim=0).clone().detach().cpu().numpy()
                
                #print(f'calculated mean translation {mean_translation}')
                # note we use the same z coordinate (J.x[0])
                registered_x = [J.x[0], J.x[1] - mean_translation[0], J.x[2] - mean_translation[1]]
                

                graph.add_space(registered_space)            
                graph.spaces[registered_space][1] = registered_x
                # NOTE: I must append an empty dictionary to the adjacency            
                graph.adj.append({})

                #print('adding an edge from I space to registered space')
                graph.add_edge([phi, A], I.space, registered_space)
                #print('adding an edge from registered space to J space') 
                graph.add_edge([A2d], registered_space, J.space) # this one is giving an error
                #print('adding an edge from registered space to I space')
                graph.add_edge([Ai,phii],registered_space,I.space)
                #print('adding an edge from J space to registered space')
                graph.add_edge([A2di],J.space,registered_space)
            else:
                #print('This is a 2d to 2d map, using only 2D transforms')
                graph.add_edge([A2d], I.space, J.space)
                graph.add_edge([A2di], J.space, I.space)
            


        else:    
            graph.add_edge([phi, A], I.space, J.space)
            graph.add_edge([Ai, phii], J.space, I.space)

    return graph


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
                    If registrations were previously computed, this argument may be omitted, and the \"graph\" argument can be used to perform additional reconstructions.\n\
3) \"configs\": list of paths to registration config JSON files, the order of which corresponds to the order of registrations listed in the previous value. (Required if computing registrations)\n\
4) \"output\": output directory which will be the output hierarchy root. If none is given, it is set to the current directory.\n\
5) \"transforms\": transforms to apply after they are computed from registration. Only necessary if \"transform_all\" is False. Format is the same as \"registrations\".\n\
6) \"transform_all\": bool. Performs all possible reconstructions given the transform graph.\n\
7) \"graph\": path to \"graph.p\" pickle file saved to output after performing registrations. (Only required for reconstructing images from previously computed registrations)"

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

    # get space_image_path
    try:
        space_image_path = input_dict["space_image_path"]
    except KeyError:
        print("space_image_path is a required argument. It is a list of images,\
             with each image being a list of the format: [\"space name\", \"image name\", \"image path\"]")
    # convert space_image_path to dictionary of dictionaries. (image_name:path key-values in a dict of space:img key-values)
    sip = {} # space-image-path dictionary
    for i in range(len(space_image_path)):
        if not space_image_path[i][0] in sip:
            sip[space_image_path[i][0]] = {}
        new_img = {space_image_path[i][1]: space_image_path[i][2]}
        sip[space_image_path[i][0]].update(new_img)

    # compute registrations
    if "registrations" in input_dict:
        try:
            configs = input_dict["configs"] # if registrations then there must also be configs
        except KeyError:
            ("configs must be included in input with registrations. configs is a list of full paths to JSON registration configuration files.")
        registrations = input_dict["registrations"]
        reg_list = [] # a list of dicts specifying inputs for each registration to perform
        for i in range(len(registrations)):
            src_space = registrations[i][0][0]
            src_img = registrations[i][0][1]
            src_path = sip[src_space][src_img]
            target_space = registrations[i][1][0]
            target_img = registrations[i][1][1]
            target_path = sip[target_space][target_img]

            reg_list.append({'registration': registrations[i], # registration format [[src_space, src_img], [target_space, target_img]]
                            'source': src_path,
                            'target': target_path,
                            'config': configs[i],
                            'output': output})
        print('registration list: ', reg_list, '\n')
        print('running registrations...')
        graph = run_registrations(reg_list)
        # save graph. Note: if a graph was supplied as an argument, it will be merged with the new one before saving.
        if "graph" in input_dict:
            with open(input_dict["graph"], 'rb') as f:
                tmp_graph = pickle.load(f)
            graph  = graph.merge(tmp_graph)
        with open(os.path.join(output, 'graph.p'), 'wb') as f:
            pickle.dump(graph, f)
    elif "graph" in input_dict: # if we do not specify registrations, but we do specify a graph
        with open(input_dict["graph"], 'rb') as f:
            graph = pickle.load(f)
            #print(graph.adj,graph.spaces)

    # daniel says, before we transform, we should update the sip    
    # importantly one space does not have an image in it! (the registered space)
    #print('printing space image path')
    #print(sip)
    for space in graph.spaces:
        #print(space)
        if space not in sip:
            sip[space] = {} # these will be the registered spaces
    

    
    if "transform_all" in input_dict and (input_dict["transform_all"] == True or input_dict['transform_all'].lower() == 'true'):
        if DEBUG: 
            print('Starting transform_all')
            
        for src_space in sip:
            if DEBUG:
                print(f'starting to transform from source {src_space}')
            # now what if there are no images in this space, we still want to output transforms
            images_to_iterate = sip[src_space]            
            if not images_to_iterate:
                if DEBUG:
                    print('*'*80)
                    print('No images to iterate over, adding a None')
                images_to_iterate = [None]                
            for src_image in images_to_iterate:
                if DEBUG:
                    print(f'starting to transform from source {src_space} image {src_image}')
                
                if src_image is not None:
                    src_path = sip[src_space][src_image]
                    I = emlddmm.Image(src_space, src_image, src_path, x=graph.spaces[src_space][1])
                else:
                    # we need to get a dummy image
                    if DEBUG:
                        print('No image here, getting a dummy image')
                    # give it a None for path
                    source_space_unregistered = src_space.replace('_registered','')                        
                    source_image = list(sip[source_space_unregistered].keys())[0] # get the first image. This is just to get file names if it is a series.
                    source_path = sip[source_space_unregistered][source_image]
                    I = emlddmm.Image(src_space, src_image, src_path, x=graph.spaces[src_space][1])
                    I.data = None
                    I.title = 'slice_dataset'
                    #I.x = graph.spaces[src_space][1] # shouldn't be stricly necessary
                    

                # reconstruct in every other space
                for target_space in [n for n in sip if n != src_space]:
                    print(f'starting to transform from source {src_space} image {src_image} to target space {target_space}')
                    if sip[target_space]: 
                        # if this is not an empty dictionary
                        # in the registered space it will be an empty dictionary
                        target_image = list(sip[target_space].keys())[0] # get the first image. This is just to get file names if it is a series.
                        target_path = sip[target_space][target_image]
                    else:
                        if DEBUG:
                            print('*'*80)
                            print(f'hi registered space {target_space}')
                        # in this case we still need the fnames for naming the outputs
                        target_space_unregistered = target_space.replace('_registered','')
                        #print(f'looking at this space {target_space_unregistered}')
                        target_image = list(sip[target_space_unregistered].keys())[0] # get the first iasdfmage. This is just to get file names if it is a series.
                        target_path = sip[target_space_unregistered][target_image]
                    if DEBUG:
                        print('about to start graph_reconstruct')
                    if os.path.splitext(target_path)[-1] == '': # this means a slice dataset
                        fnames = emlddmm.fnames(target_path)
                        graph_reconstruct(graph, output, I, target_space, target_fnames=fnames)
                    else:
                        graph_reconstruct(graph, output, I, target_space)
        
            
    # I still need to work on transforms to make it compatible with the above
    # also if the nissl space is mentioned
    elif "transforms" in input_dict:
        raise Exception('List of transformations not currently supported, only transform_all=True. TODO')
        transforms = input_dict["transforms"]
        # this requires a graph which can be output from run_registrations or included in input json
        if "graph" in input_dict:
            with open(input_dict["graph"], 'rb') as f:
                graph = pickle.load(f)
        assert "graph" in locals(), "\"graph\" argument is required when only applying new transforms."
        for t in transforms:
            I = emlddmm.Image(t[0][0], t[0][1], sip[t[0][0]][t[0][1]], x=graph.spaces[t[0][0]][1])

            target_space = t[1][0]
            target_image = t[1][1]
            target_path = sip[target_space][target_image]
            if os.path.splitext(target_path)[-1] == '':
                fnames = emlddmm.fnames(target_path)
                graph_reconstruct(graph, output, I, target_space, target_fnames=fnames)
            else:
                graph_reconstruct(graph, output, I, target_space)

    if "registered_qc" in input_dict and input_dict['registered_qc']:
        # this will be a special flag
        # TODO: after this is all done, we should get a special qc in registered space
        # now that all the data is written out, we can just read it
        # I can implement it here and move it somewhere later
        # the idea is we find the registered space, 
        # We load all data (atlas nissl fluoro) for each slice
        # we load labels and render them as curves
        print('In registered qc')
        # find the registered space from the graph
        
        
        print(input_dict)
        print('TODO: Registered QC is in progress')
        

        

    return

if __name__ == "__main__":
    main()
