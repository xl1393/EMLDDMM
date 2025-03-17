'''
manual_point_align.py is a program for interactively computing linear transformations between two 2D images.

In our workflow, the goal will be to replace a rigid transform matrix with a corrected matrix.

For our MOUSE ATLAS nose dataset, we want to use section 100 (100 is a placeholder for the first aligned section)
to the other sections 0 to 99.  In this case, we need an additional transformation.

First we would load image 99, and its registered to input transformation matrix.  We would display it in registered space.

Then we would load image 100, and its registered to input transformation matrix.  We would display it in registered space.

Then we would align the image of slice 99 to slice 100.  So slice 99 would be the moving image, and slice 100 would be the fixed image.

We would use this matrix to produce an output that would be a correction to slice 99.

In this case the "inverse" transformation would be the correct output to use as a replacement.

In my example, load inverse_output_matrix as an initializer for moving, and you'll see it lines up.

After the transformations are fixed, you should just regenerate all the outputs.

TODO:
Add interface to regenerate images and contours after you've done it.
Maybe for now just output the transformed image.  Then this can be used for the next step.


Workflow for correcting a nissl series by aligning to neighbors
---------------------------------------------------------------
Take the fixed image as a registered space vtk file for the neighbor which is good.
Take the moving image as the original (downsampled) tif file for the slice which is not good.
The defaults for reading tifs should match csh standards (origin in middle, pixel size =0.46*32).

For inputs
"-t","rigid",
"-f","/home/dtward/data/csh_data/Mouse_CCF/mouse_registered_may_31_2024/coronal_registered_v02/Nissl_nissl_MD961-N302-2023.09.23-13.14.25_MD961_3_0906_to_Nissl-registered_nissl_MD961-N302-2023.09.23-13.14.25_MD961_3_0906_20um.vtk",
"-m","/home/dtward/mounts/mg5root/nfs/data/main/M38/mba_converted_imaging_data/MD961/MD961/MD961-N303-2023.09.20-18.45.35_MD961_1_0907.tif",

                
For outputs
"-o","output_matrix_907.txt",
"-i","inverse_output_matrix_907.txt",
"--moving-output-file","moving_output_907.vtk"

Note this image and these two matrices should be used to replace previous low quality versions.  
So likely they will ultimately be given the same names as the files they replace.



Workflow for correcting a nissl series by aligning to a 3D atlas
----------------------------------------------------------------

Workflow for correcting a nissl to fluoro series
------------------------------------------------

'''

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interpn
from os.path import basename, join, split, splitext




def main(args):
    verbose = args.verbose
    if verbose: print(f'Reading moving image {args.moving_image_file}')
    xI0 = None
    if args.moving_image_file.endswith('.vtk'):
        print('ends with vtk')
        import emlddmm
        xI,I,titleI,namesI = emlddmm.read_data(args.moving_image_file)
        xI0 = xI.pop(0)
        dI = [x[1] - x[0] for x in xI]
        # move color channel last?  and convert 3D to 2D
        I = I.transpose(1,2,3,0).squeeze()
        nI = np.array(I.shape)
    else:
        I = plt.imread(args.moving_image_file) #
        nI = np.array(I.shape)
        if verbose: print(f'Found nI: {nI}')
        dI = np.array((args.moving_resolution_row,args.moving_resolution_col))    
        if verbose: print(f'Found dI: {dI}')
        if args.moving_origin_row is None:
            if verbose: print(f'Setting row origin to center')
            xI0 = np.arange(nI[0])*dI[0] - (nI[0]-1)/2*dI[0]
        else:
            if verbose: print(f'found row origin {args.moving_origin_row}')
            xI0 = np.arange(nI[0])*dI[0] + args.moving_origin_row
        if args.moving_origin_col is None:
            if verbose: print(f'Setting col origin to center')
            xI1 = np.arange(nI[1])*dI[1] - (nI[1]-1)/2*dI[1]
        else:
            if verbose: print(f'Found col origin {args.moving_origin_col}')
            xI1 = np.arange(nI[1])*dI[1] + args.moving_origin_col
        xI = [xI0,xI1]
    extentI = (xI[1][0]-dI[1]/2, xI[1][-1]+dI[1]/2, xI[0][-1]+dI[0]/2, xI[0][0]-dI[0])
        

    if verbose: print(f'Reading fixed image {args.fixed_image_file}')
    xJ0 = None
    if args.fixed_image_file.endswith('.vtk'):
        print('ends with vtk')
        import emlddmm
        xJ,J,titleJ,namesJ = emlddmm.read_data(args.fixed_image_file)
        xJ0 = xJ.pop(0)
        dJ = [x[1] - x[0] for x in xJ]
        # move color channel last?  and convert 3D to 2D
        J = J.transpose(1,2,3,0).squeeze()
        nJ = np.array(J.shape)
    else:
        J = plt.imread(args.fixed_image_file)
        nJ = np.array(J.shape)
        if verbose: print(f'Found nJ: {nJ}')
        dJ = np.array((args.fixed_resolution_row,args.fixed_resolution_col))    
        if verbose: print(f'Found dJ: {dJ}')
        if args.fixed_origin_row is None:
            if verbose: print(f'Setting row origin to center')
            xJ0 = np.arange(nJ[0])*dJ[0] - (nJ[0]-1)/2*dJ[0]
        else:
            if verbose: print(f'found row origin {args.fixed_origin_row}')
            xJ0 = np.arange(nJ[0])*dJ[0] + args.fixed_origin_row
        if args.fixed_origin_col is None:
            if verbose: print(f'Setting col origin to center')
            xJ1 = np.arange(nJ[1])*dJ[1] - (nJ[1]-1)/2*dJ[1]
        else:
            if verbose: print(f'Found col origin {args.fixed_origin_col}')
            xJ1 = np.arange(nJ[1])*dJ[1] + args.fixed_origin_col
        xJ = [xJ0,xJ1]
    extentJ = (xJ[1][0]-dJ[1]/2, xJ[1][-1]+dJ[1]/2, xJ[0][-1]+dJ[0]/2, xJ[0][0]-dJ[0])

    if args.normalization is None:
        if verbose: print('not normalizing')    
    else:
        if verbose: print('normalizing by {args.normalization} percentile')
        I = I / np.quantile(I,float(args.normalization)/100.0,axis=(0,1),keepdims=True)
        J = J / np.quantile(J,float(args.normalization)/100.0,axis=(0,1),keepdims=True)
    
    XI = np.stack(np.meshgrid(*xI,indexing='ij'),-1)
    XJ = np.stack(np.meshgrid(*xJ,indexing='ij'),-1)
    
    # let's load any existing matrices
    if args.moving_initial_affine is not None:
        # read in xy order
        with open(args.moving_initial_affine) as f:
            A0 = []
            for line in f:                
                A0.append([float(x) for x in line.split(',')])
            A0 = np.array(A0)
            
            A0 = A0[[1,0,2]]
            A0 = A0[:,[1,0,2]]
            
    else:
        A0 = np.eye(3)

    if args.fixed_initial_affine is not None:
        # read in xy order
        with open(args.fixed_initial_affine) as f:
            A1 = []
            for line in f:                
                A1.append([float(x) for x in line.split(',')])
            A1 = np.array(A1)
            A1 = A1[[1,0,2]]
            A1 = A1[:,[1,0,2]]
    else:
        A1 = np.eye(3)

    # we will display images that have been transformed already
    # note that sometimes tey will get transformed out of frame
    # note A0 is expected to be a "registered to input space" transform.
    # that means to compute the image in registered space, we don't take the inverse
    # when I use my previous result as an initializer here, it doesn't quite seem to line up
    # the translation seems to be off    
    # fixed
    # NOTE both images are sampled on the points in J
    print('transforming moving image with')
    print(A0)
    AI = interpn(xI,I,(A0[:2,:2]@XJ[...,None])[...,0] + A0[:2,-1],bounds_error=False,fill_value=0)
    AJ = interpn(xJ,J,(A1[:2,:2]@XJ[...,None])[...,0] + A1[:2,-1],bounds_error=False,fill_value=0)


    if args.layout == 2:
        fig,ax = plt.subplots(2,2)
        ax = ax.ravel()
    elif args.layout == 3:
        fig,ax = plt.subplots(1,3)

    h_imageI = ax[0].imshow(AI,extent=extentJ) # they will all be in extent J now
    ax[0].set_title('Moving')
    h_imageJ = ax[1].imshow(AJ,extent=extentJ)
    ax[1].set_title('Fixed')


    if args.grid is None:
        distance_0 = np.abs(xJ[0][-1] - xJ[0][0])
        distance_1 = np.abs(xJ[1][-1] - xJ[1][0])
        distance = np.min([distance_0, distance_1])
        grid0 = grid1 = distance/5.0
    else:
        grid0 = grid1 = args.grid
    
    squares0 = xJ[0]%grid0 >= grid0/2
    squares1 = xJ[1]%grid1 >= grid1/2
    Squares = ((squares0[:,None]-0.5)*(squares1[None,:]-0.5) ) > 0
    if J.ndim == 3:
        Squares = Squares[...,None]
    h_imageIJ = ax[2].imshow(AI*Squares + AJ*(1.0 - Squares),extent=extentJ)
    ax[2].set_title('Transformed with overlay')

    pointsI = []
    pointsJ = []
    h_pointsI = None
    h_pointsJ = None
    h_pointsIJ = None
    h_pointsJJ = None
    A = np.eye(3)
    while True:
                   
        fig.suptitle('Click a point on moving image (enter to finish)')        
        plt.pause(0.001)
        pointI = fig.ginput(timeout=0)
        if not pointI: break
        pointsI.extend(pointI)
        pointsI_ = np.array(pointsI)[:,::-1] # row column order
        
        if h_pointsI is not None:
            h_pointsI.remove()
        h_pointsI = ax[0].scatter(pointsI_[:,1],pointsI_[:,0],c='b')
        if verbose: print(f'point I {pointI}')






        fig.suptitle('Click a point on fixed image (enter to finish)')
        plt.pause(0.001)
        pointJ = fig.ginput(timeout=0)
        if not pointJ: break
        pointsJ.extend(pointJ)
        pointsJ_ = np.array(pointsJ)[:,::-1] # row column order
        
        if h_pointsJ is not None:
            h_pointsJ.remove()
        h_pointsJ = ax[1].scatter(pointsJ_[:,1],pointsJ_[:,0],c='r')
        if verbose: print(f'point J {pointJ}')

        
    
        # now fine the optimal transform
        if args.transformation == 'translation' or len(pointsI)<2:
            A = np.eye(3)
            A[:2,-1] = np.mean(pointsJ_,0) - np.mean(pointsI_,0)
        elif (args.transformation == 'rigid' or len(pointsI)<3):
            # first subtract com
            comI = np.mean(pointsI_,0)
            comJ = np.mean(pointsJ_,0)
            pointsI0 = pointsI_ - comI
            pointsJ0 = pointsJ_ - comJ
            # now find the cross covariance
            S = pointsI0.T@pointsJ0
            # now find the svd
            u,s,vh = np.linalg.svd(S)
            # now the rotation
            R = vh.T@u
            # now the translation
            T = comJ - R@comI            
            A = np.eye(3)
            A[:2,:2] = R
            A[:2,-1] = T            


        elif args.transformation == 'affine':
            pointsI__ = np.concatenate((pointsI_,np.ones_like(pointsI_[:,-1][...,None])),-1)
            pointsJ__ = np.concatenate((pointsJ_,np.ones_like(pointsJ_[:,-1][...,None])),-1)
            A = np.linalg.solve(pointsI__.T@pointsI__, pointsI__.T@pointsJ__).T
        pointsIJ_ = (A[:2,:2]@pointsI_.T).T + A[:2,-1]
        Ai = np.linalg.inv(A)
        
        Xs = (Ai[:2,:2]@XJ[...,None])[...,0] + Ai[:2,-1]
        print(f'transforming moving image using matrix')
        print(Ai)
        AAI = interpn(xJ,AI,Xs,bounds_error=False,fill_value=0) # note both images are sampled on the pixels in J
        if h_imageIJ is not None:
            h_imageIJ.remove()
        h_imageIJ = ax[2].imshow(AAI*Squares + AJ*(1.0 - Squares),extent=extentJ)
        if h_pointsIJ is not None:
            h_pointsIJ.remove()
        h_pointsIJ = ax[2].scatter(pointsIJ_[:,1],pointsIJ_[:,0],fc='none',ec='m')
        if h_pointsJJ is not None:
            h_pointsJJ.remove()
        h_pointsJJ = ax[2].scatter(pointsJ_[:,1],pointsJ_[:,0],c='r')
        plt.pause(0.001)
    
    # now we write out
    if args.transform_output is None:
        movingbase = splitext(basename(args.moving_image_file))[0]
        fixedbase = splitext(basename(args.fixed_image_file))[0]
        outname = movingbase + '_to_' + fixedbase + '_matrix.txt'
    else:
        outname = args.transform_output
    if args.inverse_output is None:
        movingbase = splitext(basename(args.moving_image_file))[0]
        fixedbase = splitext(basename(args.fixed_image_file))[0]
        outnamei = fixedbase + '_to_' + movingbase + '_matrix.txt'
    else:
        outnamei = args.inverse_output

    
    
    if verbose: print(f'output: {outname}')
    if verbose: print(f'inverse_output: {outnamei}')

    # okay now what do I actually want to write out?
    # some combination of A0 and A
    # the sequence of transforms I apply to my image sample points
    # 1. Apply A0
    # 2. Apply A^{-1}
    # that means the sequence of operations I would apply to points in the space is
    # 1. Apply A
    # 2. Apply A0^{-1}
    # we hope that the first one is a better version of A0
    # we hope the second one is a better version of A0^{-1}
    # note that the second one is my "forward matrix"
    # the first one is my "inverse matrix"
    # as a double check, if I use my "inverse matrix" as an initial guess for moving, it should appear aligned to fixed
    print(A)
    #A_ = np.linalg.inv(A0)@A # this looked incorrect
    A_ = A@np.linalg.inv(A0) # try this, this looked correct
    print('writing out updated inverse matrix')
    print(A_)
    Aout = A_.copy()[[1,0,2]] # swap xy
    Aout = Aout[:,[1,0,2]]    # swap xy
    print(Aout)
    with open(outname,'wt') as f:
        f.write(f'{Aout[0,0]}, {Aout[0,1]}, {Aout[0,2]}\n')
        f.write(f'{Aout[1,0]}, {Aout[1,1]}, {Aout[1,2]}\n')
        f.write(f'{Aout[2,0]}, {Aout[2,1]}, {Aout[2,2]}\n')
    Ai_ = np.linalg.inv(A_)
    Aout = Ai_.copy()[[1,0,2]] # swap xy
    Aout = Aout[:,[1,0,2]]    # swap xy
    print(Aout)
    with open(outnamei,'wt') as f:
        f.write(f'{Aout[0,0]}, {Aout[0,1]}, {Aout[0,2]}\n')
        f.write(f'{Aout[1,0]}, {Aout[1,1]}, {Aout[1,2]}\n')
        f.write(f'{Aout[2,0]}, {Aout[2,1]}, {Aout[2,2]}\n')

    # output transformed images
    if args.moving_output_file is not None:
        # this one is already calculated
        # it's called AAI
        # write it out here
        # but on what sampling grid? The sampling grid of the fixed image.  For our nose correct, these are the same. In other cases I may want something else.
        # if it is supposed to be some kind of a "fix" maybe it should be on its original sampling grid
        # I should check by regenerating AAI wit A_        
        Xs = (Ai_[:2,:2]@XJ[...,None])[...,0] + Ai_[:2,-1]
        AAI = interpn(xI,I,Xs,bounds_error=False,fill_value=0)
        
        if args.moving_output_file.endswith('.vtk'):
            if xI0 is not None:
                print(f'Setting origin to 0 and and slice thickness to 20 for vtk output')
                xJuse = [[0.0,20.0],xJ[0],xJ[1]]
            else:
                print(f'Setting slice thickness to 20 for vtk output')
                xJuse = [[xI0,xI0+20.0],xJ[0],xJ[1]]
            emlddmm.write_data(args.moving_output_file, xJuse, AAI[None].transpose(-1,0,1,2), 'manually edited' )
            
        else:
            plt.imsave(args.moving_output_file, AAI.clip(0,1))
        
        
    # we don't need to output the fixed image because it is not changing
    
    
if __name__ == '__main__':
    print('hello world')

    # set up the parse
    parser = argparse.ArgumentParser(
        prog='python manual_point_align.py',
        description='This program can produce affine transformations to align two images based on mouse clicks',
        epilog='Author: Daniel Tward'
    )
    # add arguments        
    parser.add_argument('-m','--moving-image-file',type=str,required=True,help='Filename for the moving image (sometimes called "atlas")')
    parser.add_argument('-f','--fixed-image-file',type=str,required=True,help='Filename for the fixed image (sometimes called "target")')
    parser.add_argument('-o','--transform-output',type=str,help='Name of output file (default: {moving}_to_{fixed}_matrix.txt)')
    parser.add_argument('-i','--inverse-output',type=str,help='Name of output file for inverse (default: {fixed}_to_{moving}_matrix.txt)')
    parser.add_argument('-t','--transformation',type=str,choices=['translation','rigid','affine'],default='rigid',help='Transformation type (default rigid)')
    parser.add_argument('--moving-origin-row',type=float,default=None,help='Coordinate of the first row of the moving image (default: origin is in center of image)')
    parser.add_argument('--moving-origin-col',type=float,default=None,help='Coordinate of the first column of the moving image (default: origin is in center of image)')
    parser.add_argument('--moving-resolution-row',type=float,default=0.46*32,help='Resolution of the rows of the moving image (default: 0.46*32)')
    parser.add_argument('--moving-resolution-col',type=float,default=0.46*32,help='Resolution of the columns of the moving image (default: 0.46*32)')
    parser.add_argument('--fixed-origin-row',type=float,default=None,help='Coordinate of the first row of the fixed image (default: origin is in center of image)')
    parser.add_argument('--fixed-origin-col',type=float,default=None,help='Coordinate of the first column of the fixed image (default: origin is in center of image)')
    parser.add_argument('--fixed-resolution-row',type=float,default=0.46*32,help='Resolution of the rows of the fixed image (default: 0.46*32)')
    parser.add_argument('--fixed-resolution-col',type=float,default=0.46*32,help='Resolution of the columns of the fixed image (default: 0.46*32)')
    parser.add_argument('-v','--verbose',type=bool,default=False,help='Print info useful for debugging')
    parser.add_argument('-l','--layout',type=int,default=2,choices=[2,3],help='2 for 2x2 layout, 3 for 1x3 layout')
    parser.add_argument('-g','--grid',type=float,default=None,help='Size of grid in overlay (default is one fifth of smallest dimension)')
    parser.add_argument('-n','--normalization',type=float, default=None, help='Normalize images for display if not none (default).  Enter a number from 0 to 100 to normalize by this percentile.  This is often necessary for high dynamic range images (like fluorescence).')    
    parser.add_argument('--moving-initial-affine',type=str,default=None, help='an initial affine transform for the moving image (text file).  This is expected to be a "registered_to_input" matrix, which will be used to reconstruct the image in registered space and would be located in the registered space folder.')
    parser.add_argument('--fixed-initial-affine',type=str,default=None, help='an initial affine transform for the fixed image(text file). This is expected to be a "registered_to_input" matrix, which will be used to reconstruct the image in registered space and would be located in the registered space folder.')
    parser.add_argument('--moving-output-file',type=str,default=None, help='A filename for the transformed moving image.')
    

    # parse
    args = parser.parse_args()
    print(args)

    main(args)