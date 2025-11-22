import argparse
import json
import os
import torch
import numpy as np
from os.path import join, splitext
from os import makedirs
from .io import read_data, write_data, load_slices, write_qc_outputs, write_transform_outputs, write_outputs_for_pair, read_vtk_data
from .vis import draw
from .utils import downsample, downsample_image_domain, interp
from .core import emlddmm_multiscale, compose_sequence, apply_transform_float, apply_transform_int

def main():
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

if __name__ == '__main__':
    main()
