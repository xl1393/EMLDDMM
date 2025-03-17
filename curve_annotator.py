''' Curve annotator

ex

python curve_annotator.py /home/dtward/bmaproot/nafs/dtward/data/merfish/jean_fan_2021/OneDrive_1_8-5-2021/datasets_mouse_brain_map_BrainReceptorShowcase_Slice1_Replicate1_cell_metadata_S1R1_rasterized.npz
'''

import argparse
from os.path import split,join,splitext
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import tifffile

if __name__ == '__main__':
    print('hello world')

    parser = argparse.ArgumentParser(
        prog='curve_annotator',
        description='Takes an image as an input (npz format x I), and provides an interface for annotating curves',
    )

    parser.add_argument(
        'filename',
        help='Filename for image data.  This should be a npz file containing "x,y" and "I" or "J"',
    )
    parser.add_argument(
        '-o,','--output',
        default=None,
        help='Defaults to input name with a suffix "_curves.npy"',
    )
    parser.add_argument('-D','--pixel_size',
                        default=None,
                        type=float,
                        help='If loading a raw image, use pixel size')
    # todo add origin, add independent x and y


    args = parser.parse_args()
    print(args)

    # load the image
    # start with a special case
    if args.filename.isnumeric():
        ind = int(args.filename)
        files = glob('/home/dtward/bmaproot/nafs/dtward/merfish/jean_fan_2021/OneDrive_1_8-5-2021/*_rasterized.npz')
        files.sort()
        args.filename = files[ind]
        print(f'special case, using index {ind} to select file {args.filename}')


    # get input extension
    ext = splitext(args.filename)[-1]
    
    # load the data from a numpy file
    if ext == '.npz':
        try:
            data = np.load(args.filename)
        except:
            raise Exception(f'Could not load input image {args.filename}')
        print(f'image contains ')
        print([k for k in data])
        # let's draw the image
        x = data['x']
        y = data['y']
        I = data['I'][0] # 0 for grayscale
        
    elif ext == '.tif'  or ext == '.tiff':
        I = tifffile.imread(args.filename)
        if I.dtype == np.uint8:
            I = I / 255.0
        elif I.dtype == np.uint16:
            I = I / np.quantile(I,0.99,axis=(0,1))
        if args.pixel_size is None:
            raise Exception('When using raw image, pixel size is required')
        x = np.arange(I.shape[1])*args.pixel_size - (I.shape[1]-1)*args.pixel_size/2.0
        y = np.arange(I.shape[0])*args.pixel_size - (I.shape[0]-1)*args.pixel_size/2.0
        
        
    else:
        raise Exception(f'Extension {ext} not understood')

    # get the output name
    if args.output is None:
        output = args.filename.replace(ext,'_curves.npy')
    print(f'output name {output}')


    dx = x[1] - x[0]
    dy = y[1] - y[0]
    
    fig,ax = plt.subplots()
    # we need to add some zoom functionality
    def zoom(event):
        print(event)
        pass
    fig.canvas.mpl_connect('scroll_event',zoom)
    ax.imshow(I,extent=(x[0]-dx/2.0,x[-1]+dx/2,y[-1]+dy/2,y[0]-dy/2))
    #plt.show(block=False)


    data = {}
    try:
        data = np.load(output,allow_pickle=True).item()
    except:

        print(f'could not load previous')


    for k in data:
        points = data[k]
        ax.plot([p[0] for p in points],[p[1] for p in points])
    plt.show(block=False)
    
    count = 0
    while True:
        name = input(f'Enter name for curve {count}: ')
        if len(name) == 0:
            break
        ax.set_title(name)        
        fig.suptitle(f'annotate curve {count} then press middle')        
        plt.show(block=False)
        points = plt.ginput(-1)
        if len(points) == 0:
            break

        print(points)
        ax.plot([p[0] for p in points],[p[1] for p in points])
        plt.show(block=False)
        data[name] = points
        count += 1

        


    print(data)
    np.save(output,data,)
    
    

    
