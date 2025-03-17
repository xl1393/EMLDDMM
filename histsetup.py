import os
from pkgutil import extend_path
import re
import sys
import getopt
import pandas as pd
import shutil
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
import h5py

def _detect_img_ext(fnames):
    extensions = [os.path.splitext(f)[-1] for f in fnames]
    extensions = list(set(extensions))
    extension = [ext for ext in extensions if ext not in ['.csv','.tsv','.json']]
    if len(extension) != 1:
        raise Exception("Image extension is ambiguous. You may set the extension name explicitly with the 'ext' argument.")
    return extension[0]


def downsample_slices(subject_dir, output_dir, ext='', slice_downfactor=1, image_downfactor=1, sep='_',fnumidx=-1):
    """ Downsample Slices

    Copy a subset of optionally downsampled 2D images to the output directory.

    Parameters
    ----------
    subject_dir: str
        Path to image series
    output_dir: str
        Output path
    ext: str
        subject image file extension (e.g. 'png')
    slice_downfactor: int
        Specifies n, where every nth image will be copied to the output folder.
    image_downfactor: int
        Image downsampling factor.

    Example
    -------
    >>> subject_dir = '/path/to/histology/data
    >>> output_dir = 'example_outputs'
    >>> downsample_slices(subject_dir, output_dir, ext='png', slice_downfactor=20, image_downfactor=32)

    Raises
    ------
    FileNotFoundError
        If the subject directory does not exist.

    """

    try:
        fnames = os.listdir(subject_dir)
    except FileNotFoundError:
        print('subject directory does not exist')
        sys.exit(1)
        
    if len(fnames) == 0:
        print('subject directory is empty')
        sys.exit(1)
    if ext == '':    
        ext = _detect_img_ext(fnames)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # copy geometry.csv to output if it exists
    if 'geometry.csv' in fnames:
        shutil.copy(os.path.join(subject_dir, 'geometry.csv'), output_dir)

    fnames = [i for i in fnames if i.endswith(ext)]
    fnames = sorted(fnames, key=lambda x: x.split(sep)[fnumidx][:4])

    # get the number of slices (missing slices included)
    max_slice = int(fnames[-1].split('_')[-1][:4])

    # get list of slices to be kept after downsampling (these are evenly spaced numbers that may include missing slices)
    slices_to_keep = np.arange(1, max_slice+1, slice_downfactor)

    # Find paths to existing slices in the downsampled list (intersection of existing slices and downsampled slices)
    slice_nums = [int(f.split('_')[-1][:4]) for f in fnames]
    slice_nums_down = [x for x in slices_to_keep if x in slice_nums]
    fnames_down = [f for f in fnames if int(f.split('_')[-1][:4]) in slice_nums_down]
    fpaths_down = [os.path.join(subject_dir,x) for x in fnames_down]

    for i in range(len(fpaths_down)):
        # downsample images if specified
        if image_downfactor > 1:
            img = plt.imread(fpaths_down[i])
            img_resized = resize(img, (img.shape[0] // image_downfactor, img.shape[1] // image_downfactor),\
                 anti_aliasing=True, preserve_range=False)
            plt.imsave(os.path.join(output_dir, os.path.splitext(os.path.basename(fpaths_down[i]))[0]+'.png'), img_resized)
        else:
            shutil.copy(fpaths_down[i], output_dir)


def make_samples_tsv(subject_dir, ext='', slice_downfactor=1, max_slice=None, sep='_',fnumidx=-1):
    """ Make 'samples.tsv' file
    
    Saves a tsv file listing the images in the folder.

    Parameters
    ----------
    subject_dir: path
        Path to the dataset.
    ext: str
        image file extension (e.g. 'png')    
    max_slice: int
        Number of slices in the original dataset (not currently used)

    Example
    -------
    >>> subject_dir = 'example_outputs'
    >>> make_samples_tsv(subject_dir, 'png')

    Raises
    ------
    FileNotFoundError
        If the subject directory does not exist.

    """
    # create list of files from source directory (subject_dir) in order of slice number
    fnames = os.listdir(subject_dir)
    if len(fnames) == 0:
        print('subject directory is empty')
        sys.exit(1)
    if ext == '':
        ext = _detect_img_ext(fnames)

    fnames = [i for i in fnames if i.endswith(ext)]
    fnames = sorted(fnames, key=lambda x: x.split(sep)[fnumidx][:4])
    # uncomment this block to include missing images in tsv file (not currently implemented for registration)
    # get missing image numbers
    img_nums = [int(x.split('_')[-1][:4]) for x in fnames]
    if not max_slice:
        max_slice = img_nums[-1]
    slice_range = np.arange(1, max_slice+1, slice_downfactor)
    missing_imgs = list(set(img_nums) - set(slice_range)) + list(set(slice_range) - set(img_nums))
    missing_imgs = sorted(missing_imgs)

    print("missing images: \n", missing_imgs)
    # insert missing image names into fnames
    missing_ids = []
    for i in range(len(missing_imgs)):
        fname = fnames[0].split('-')[0]  + '-_' + str(missing_imgs[i]).zfill(4) + '.' + fnames[0].split('.')[-1]
        N = len(img_nums)
        for j in range(N):
            if img_nums[j] > missing_imgs[i]:
                fnames.insert(j, fname)
                img_nums.insert(j, missing_imgs[i])
                missing_ids.append(j)
                break
            elif j==(N-1):
                fnames.append(fname)
                img_nums.append(missing_imgs[i])
                missing_ids.append(j+1)
                break

    # make samples.tsv file
    with open(os.path.join(subject_dir, 'samples.tsv'), 'w') as f:
        # f.write('sample_id    participant_id    species    status\n')
        f.write('sample_id\tparticipant_id\tspecies\tstatus\n')
        status = 'present'
        j = 0
        for i in range(len(fnames)):
            status = 'present'
            for idx in missing_ids:
                if i == idx:
                    status = 'missing'
                    break
            f.write('{0}\t{1}\tMus Musculus\t{2}\n'.format(fnames[i], fnames[i][:5], status))


# Set up metadata JSON file
# Need Pixel Size, field of view, thickness, offset
def generate_sidecars(subject_dir, ext='', max_slice=None, dtype='uint8', dv=[14.72,14.72,10.], slice_downfactor=1, sep='_', fnumidx=-1, space='right-inferior-posterior'):
    """ Generate Sidecar Files
    
    Saves out JSON format sidecare files for each image in the dataset.

    Parameters
    ----------
    subject_dir: str
        Path to image series
    ext: str
        image file extension (e.g. 'png')
    max_slice: int
        Number of slices in the original dataset (used to calcuate spatial information for the slices).
    dtype: str
        Image data type
    dv: list of float
        voxel spacing in microns (ordered: row, col, slice)
    slice_downfactor: int
        Factor used to reduce the number of images from the original dataset. 

    Example
    -------
    >>> subject_dir = 'example_outputs'
    >>> generate_sidecars(subject_dir, 'png', max_slice=1389, dv=[14.72,14.72,10.0])
    MD787-N1-2019.03.28-21.52.46_MD787_1_0001.json
    MD787-N7-2019.03.28-22.05.43_MD787_3_0021.json
    MD787-N14-2019.03.28-22.20.46_MD787_2_0041.json
    ...

    Raises
    ------
    FileNotFoundError
        If the subject directory does not exist.

    """

    fnames = os.listdir(subject_dir)
    # Make sure the directory is not empty
    if len(fnames) == 0:
        print('subject directory is empty')
        sys.exit(1)
    if ext == '':
        ext = _detect_img_ext(fnames)

    sizes = []
    space_directions = []
    space_origin = []
    
    geometry = 'geometry.csv' in fnames
    fnames = [i for i in fnames if i.endswith(ext)] # this removes geometry.csv from fnames
    fnames = sorted(fnames, key=lambda x: x.split(sep)[fnumidx][:4])

    # get geometry info
    if geometry:
        # first get geometry info
        geometry = pd.read_csv(os.path.join(subject_dir, 'geometry.csv'), index_col=False)
        dv = []
        for i in range(len(geometry)):
            dv.append([geometry.iloc[i, 4], geometry.iloc[i, 5], geometry.iloc[i, 6]])
            sizes.append([4, geometry.iloc[i, 1], geometry.iloc[i, 2], geometry.iloc[i, 3]])
            space_directions.append([[dv[i][0], 0., 0.], [0., dv[i][1], 0.], [0., 0., dv[i][2]*slice_downfactor]])
            space_origin.append([geometry.iloc[i, 7], geometry.iloc[i, 8], geometry.iloc[i, 9]])

    else:
        if not max_slice:
            max_slice = int(fnames[-1].split(sep)[fnumidx][:4])
        dx, dy, dz = dv
        center = (max_slice - 1) / 2

        for i in range(len(fnames)):
            slice_num = int(fnames[i].split('_')[-1][:4])
            img_path = os.path.join(subject_dir, fnames[i])
            try:
                img_shape = plt.imread(img_path).shape
            except:
                if 'h5' in ext:
                    with h5py.File(img_path,'r') as f:
                        img_shape = list(f[list(f.keys())[0]][:].shape[:2])
                        img_shape.append(1) # append 1 to represent channels
                else:
                    print('file type not recognized')
            x_origin = -(img_shape[1] - 1) / 2 * dx
            y_origin = -(img_shape[0] - 1) / 2 * dy
            space_origin.append([x_origin, y_origin, dz / slice_downfactor * (slice_num-1 - center)])
            sizes.append([img_shape[2], img_shape[1], img_shape[0], 1])
            space_directions.append([[dx, 0., 0.], [0., dy, 0.], [0., 0., dz]])

    sample_ids = [os.path.splitext(x)[0] for x in fnames]
    for i in range(len(fnames)):
        print(sample_ids[i] + '.json')
        img_path = os.path.join(subject_dir, fnames[i])
        with open(os.path.join(subject_dir, sample_ids[i] + '.json'), 'w') as f:
            f.write('{{\n'
                    '  \"DataFile\": \"{0}\",\n'
                    '  \"Type\": \"{1}\",\n'
                    '  \"Dimension\": 4,\n'
                    '  \"Sizes\": {2},\n'
                    '  \"Endian\": \"big\",\n'
                    '  \"Space\": \"{3}\",\n'
                    '  \"SpaceDimension\": 3,\n'
                    '  \"SpaceUnits\": [\"um\", \"um\", \"um\"],\n'
                    '  \"SpaceDirections\": ["none", {4}, {5}, {6}],\n'
                    '  \"SpaceOrigin\": {7}\n'
                    '}}'.format(fnames[i], dtype, sizes[i], space, space_directions[i][0], space_directions[i][1],
                                space_directions[i][2],
                                space_origin[i]))


def main():
    subject_dir = '.'
    output_dir = '.'
    ext = ''
    slice_down = 1
    res_down = 1
    max_slice = None
    dv = [14.72, 14.72, 10.]
    space = "right-inferior-posterior"
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hi:o:d:e:s:r:m:")
    except getopt.GetoptError:
            print('histsetup.py -i <inputDir> -o <outputDir> -d <dv> -e <fileExtension> -s <sliceDownFactor> -r <resolutionDownFactor> -m <maxSlice> -p <space>')
            sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('histsetup.py -i <inputDir> -o <outputDir>  -d <dv> -e <fileExtension> -s <sliceDownFactor> -r <resolutionDownFactor> -m <maxSlice> -p <space>')
            sys.exit()
        elif opt == '-i':
            subject_dir = arg
        elif opt == '-o':
            output_dir = arg
        elif opt == '-d':
            dv = list(map(float, arg[1:-1].split(',')))
        elif opt == '-e':
            ext = arg
        elif opt == '-s':
            slice_down = int(arg)
        elif opt == '-r':
            res_down = int(arg)
        elif opt == '-m':
            max_slice = int(arg)
        elif opt == 'p':
            space = str(arg)

    dv = [dv[0]*res_down, dv[1]*res_down, dv[2]*slice_down]

    print('subject directory: ', subject_dir, '\n',
          'output directory: ', output_dir, '\n',
          'slice downsample factor: ', slice_down, '\n',
          'resolution downsample factor: ', res_down, '\n',
          'maximum slice number: ', max_slice, '\n',
          'dv: ', dv)

    downsample_slices(subject_dir, output_dir, ext, slice_downfactor=slice_down, image_downfactor=res_down)
    if res_down > 1:
        ext = 'png'
    make_samples_tsv(output_dir, ext=ext, slice_downfactor=slice_down, max_slice=max_slice)
    generate_sidecars(output_dir, ext=ext, max_slice=max_slice, dv=dv, slice_downfactor=slice_down, space=space)


if __name__ == '__main__':
    main()
