File Formats
============


We propose to use VTK formatted data whenever possible. Currently we use simple legacy .vtk files (https://docs.vtk.org/en/latest/design_documents/VTKFileFormats.html). This supports vector and raster graphics, works well with visualization software including web viewers, and is largely human readable.  It does not support compression, and so other formats are also used.


JSON geometry files
^^^^^^^^^^^^^^^^^^^

Every imaging file that does not store geometry information (e.g. 2D slices stored as pngs/tifs/etc) will have a corresponding short JSON sidecar file located in the same directory, with the same name filename, and the extension .json appended.  The information stored here should be as close as possible to an NRRD header (http://teem.sourceforge.net/nrrd/format.html).  Note that 2D images are described as though they are 3D, with pixel size in the z dimension referring to section thickness. 

Such sidecar files are inspired by the BIDS standard, and contain information typically stored in an NRRD header.  Each sidecar file must contain the following fields:

* “DataFile”: the image file name 
* “SpaceDirections”: a list of vectors for each image dimension specifying the unit conversion from pixel indices (row,column) to input space coordinates. Note that the z-coordinate conversion indicates the slice thickness plus the spacing between slices.  This is a list of vectors in x,y,z order, where the xyz coordinate system is defined in the “input space” section.
* “SpaceOrigin”:  The world coordinate of the image origin, in x,y,z order.


Other metadata can be stored in the json file, but only the above 3 are generally used for the data loader functions.

Note that each 2D image is modeled as a 3D image with a single slice (i.e. a size of 1 in).

An example is shown below::

    {
      "DataFile": "MD787_small_nissl/MD787-N27-2019.03.28-22.55.54_MD787_2_0080.png",
      "Type": "Float32",
      "Dimension": 3,
      "Endian": "big",
      "Sizes": [3, 392, 480, 1],
      "Space": "inferior-right-posterior",
      "SpaceDimension": 3,
      "SpaceUnits": ["um", "um","um" ],
      "SpaceDirections": [
        "none", 
        [44.160000000000004, 0.0, 0.0],
	    [0.0, 44.160000000000004, 0.0],
	    [0.0, 0.0, 200	] 
      ],
      "SliceThickness" : 10.0
      "SpaceOrigin": [-8633.28, -10576.32, -120100.0]
    }

Note when data is read, the space directions are reversed to line up with common conventions for image array axes.  The last axis of an image array corresponds to columns (x), the second last corresponds to rows (y), and the third last corresponds to slices (z).



Dataset lists
^^^^^^^^^^^^^
Since sections may be missing or require other comments, we include a tsv file in the same directory describing every slice in the dataset. The required fields are sample_id and status, the latter should contain present or absent.  An example is shown below::

    sample_id 	 participant_id 	 species 	 status
    MD787-N7-2019.03.28-22.05.43_MD787_2_0020.png	MD787	Mus Musculus	present
    MD787-N14-2019.03.28-22.20.46_MD787_1_0040.png	MD787	Mus Musculus	present
    MD787-N20-2019.03.28-22.36.39_MD787_3_0060.png	MD787	Mus Musculus	present
    MD787-N27-2019.03.28-22.55.54_MD787_2_0080.png	MD787	Mus Musculus	present
    MD787-N34-2019.03.28-23.15.58_MD787_1_0100.png	MD787	Mus Musculus	present
    MD787-N40-2019.03.28-23.33.43_MD787_3_0120.png	MD787	Mus Musculus	present
    MD787-N47-2019.03.28-23.54.40_MD787_2_0140.png	MD787	Mus Musculus	present
    MD787-N54-2019.03.29-00.15.46_MD787_1_0160.png	MD787	Mus Musculus	present
    MD787-N60-2019.03.29-00.33.42_MD787_3_0180.png	MD787	Mus Musculus	present
    MD787-N67-2019.03.29-00.56.05_MD787_2_0200.png	MD787	Mus Musculus	present
    MD787-N74-2019.03.29-01.18.34_MD787_1_0220.png	MD787	Mus Musculus	present
    MD787-N80-2019.03.29-01.36.50_MD787_3_0240.png	MD787	Mus Musculus	present
    MD787-N87-2019.03.29-01.57.37_MD787_2_0260.png	MD787	Mus Musculus	present
    MD787-N94-2019.03.29-02.19.41_MD787_1_0280.png	MD787	Mus Musculus	present
    MD787-N100-2019.03.29-02.40.34_MD787_3_0300.png	MD787	Mus Musculus	present
    MD787-N107-2019.03.29-03.04.17_MD787_2_0320.png	MD787	Mus Musculus	present
    MD787-N114-2019.03.29-03.28.07_MD787_1_0340.png	MD787	Mus Musculus	present
    MD787-N120-2019.03.29-03.49.00_MD787_3_0360.png	MD787	Mus Musculus	present


Legacy CSV geometry files
^^^^^^^^^^^^^^^^^^^^^^^^^
In older versions of our pipeline, we stored information in a csv file, with the following 10 fields for each image.

* Filename
* Nx ny nz: number of pixels in the x y and z direction (nz=1 if 2D image)
* Dx dy dz: pixel size in x y and z direction (dz = slice thickness if 2D image)
* X0 y0 z0: coordinate of the first pixel in x y and z direction (z0 = location within dataset if 2D image)

Cold Spring Harbor legacy geometry files
^^^^^^^^^^^^^^^^^^^^^^^^^^^
Cold Spring Harbor is storing geometry data in a plain text file.  For example::


    2021-10-05 14:30:13.177668
    Registered : Y
    Input Path:/nfs/data/main/M32/RegistrationData/Data_Marmoset/m6344/Transformation_OUTPUT/m6344_img/
    Output Path:/nfs/data/main/M32/Cell_Detection/CellDetPass1_reg/m6344/
    Number of Files Detected:386
    Resolution:0.92
    Resolution in Json: 1 micron/pixel



.. _vtkref:

3D imaging data
^^^^^^^^^^^^^^^


Our standard is to use simple vtk legacy format for 3D (see https://examples.vtk.org/site/VTKFileFormats/#simple-legacy-formats). Note that this data is always stored in big endian, regardless of machine defaults. These have simple human readable headers that contain the 9 pieces of information above.  Our pipeline provides basic support for nifti images using the :code:`nibabel` python package.



2D microscopy images from Cold Spring Harbor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Acquired microscopy data is stored at Cold Spring Harbor Laboratory in jp2 format at full resolution (generally 0.46 microns per pixel). The filename is generated by the scanner, following a template schema that the Mitra lab uses in a standard manner. An example is::

   MD787-N3-2019.03.28-21.57.34_MD787_3_0009.jp2 

Where the meaning of each hyphen separated field is::

   {sample id}-{N/F/IHC for nissl fluoro or ihc}-{slide number}-{date}-{time}_{sample id}_{what position on slide}_{section number id in anterior to posterior order (generally)}.

Note that no geometry information is stored in filenames here, so this should be added as a json companion file.



2D datasets for registration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Typically data is downsampled by a factor of 32 and saved as a .tif with the same filename.

Registration data can be safely downsampled to approximately the same resolution as atlas images (10-50 micron).

For 2D serial section datasets images should be stored in a single directory using standard imaging formats (i.e. to be read by matplotlib's imread function), downsampled by approximately 32 times (e.g. 14.72 microns for CSH data).  While our pipelines do support downsampling to desired resolutions, code will run more efficiently if these sections are already downsampled.

Slice datasets must contain sidecar json files, and data set list tsv files. The script, :mod:`histsetup`, generates sidecar files and dataset lists given a subject dataset and voxel spacing (where spacing in the z axis indicates slice thickness plus slice spacing). 





Affine Transformations
^^^^^^^^^^^^^^^^^^^^^^
Affine transformations are stored as 4x4 matrices written in a text file.  Each column is separated by spaces.  Each row is separated by a new line.  Coordinates are in xyz order.  When read into python using our library, they will be converted to zyx order to be consistent with our conventions for indexing image arrays.

Deformations
^^^^^^^^^^^^
Deformations are as 3 component displacement fields (not position fields) in vtk files.  In python we work in zyx order, but when writing to vtk fields we switch to xyz order which is the vtk convention.



Velocity fields
^^^^^^^^^^^^^^^
Velocities are nt x 3 component vector fields in vtk files.  In python we work in zyx order, but when writing to vtk fields we switch to xyz order which is the vtk convention.



Annotations
^^^^^^^^^^^
2D annotations are stored as geojson files using the multipolygon data type.  Each structure is given a name, and an integer ID. Metadata stores information about which atlas is used, and which 2D image file the annotations correspond to.

These files will also contain atlas coordinate gridlines.


Point sets
^^^^^^^^^^
Point sets are stored in vtk polydata format.

Pixel indexed point sets
^^^^^^^^^^^^^^^^^^^^^^^^
Point sets that describe detected cells are stored in geojson format.  These point sets have some constraints based on how they will be displayed using open layers or angular on the web.







