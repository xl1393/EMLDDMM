# EMLDDMM

Please see our online documentation [here](https://twardlab.github.io/emlddmm/build/html/index.html).  This document provides only a brief overview.  As an additional resource a web interface is provided, including an example dataset, [here](https://twardlab.com/reg).


## Introduction
Expectation Maximization Large Deformation Diffeomorphic Metric Mapping is an image registration method for aligning datasets in the presence of differing contrast profiles and missing tissue or artifacts.

It uses an [expectation maximization algorithm](https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm) to handle missing data, and leverages the powerful [Large Deformation Diffeomorphic Metric Mapping](https://en.wikipedia.org/wiki/Large_deformation_diffeomorphic_metric_mapping)  (LDDMM) [ref](https://doi.org/10.1023/B:VISI.0000043755.93987.aa) paradaigm  to ensure mappings are diffeomorphisms [ref]( https://www.jstor.org/stable/43638248 ), and are generated in a Reimannian framework suitable for statistical analysis such as PCA [ref](https://doi.org/10.1155/2013/205494).

These concepts were brought together into the EMLDDMM algorithm described in [ref](https://doi.org/10.3389/fnins.2020.00052),[ref](https://doi.org/10.1101/2020.03.22.002618).

Our package is designed for 3D to 3D image registration, or 3D to 2D serial sections, and suports pipelines to efficiently register datasets with multiple modalities of images.

## File formats

### 3D data
We use vtk data as a standard, and use ([vtk simple legacy format](https://kitware.github.io/vtk-examples/site/VTKFileFormats/) because it has a human readable header.  It supports images and vector fields, as well as polydata (points, edges, triangulations) under a common standard.  For visualization we use [paraview](https://www.paraview.org/), or [itksnap](http://www.itksnap.org/pmwiki/pmwiki.php) (web viewer support coming soon).

For input data we include nibabel and pynrrd as a dependency, and support several other formats supported by these libraries.

### 2D serial section data
For 2D serial section datasets images should be stored in a single directory using standard imaging formats (i.e. to be read by matplotlib's imread function).  While our pipelines do support downsampling to desired resolutions, code will run more efficiently if these sections are already downsampled.

#### JSON Sidecar files
2D images have their geometry specified in a json sidecar file. Such sidecar files are inspired by the [BIDS](https://bids-apps.neuroimaging.io/) standard, and contain information typically stored in an [NRRD](http://teem.sourceforge.net/nrrd/format.html) header.  Each 2D image should have a sidecar file with the same filename, and the extension `.json`.

Note that each 2D image is modeled as a 3D image with a single slice.  An example is shown here:
```
{
  "DataFile": "MD787_small_nissl/MD787-N27-2019.03.28-22.55.54_MD787_2_0080.png",
  "Type": "Float32",
  "Dimension": 3,
  "Endian": "big",
  "Sizes": [
    3,
    392,
    480,
    1
  ],
  "Space": "inferior-right-posterior",
  "SpaceDimension": 3,
  "SpaceUnits": [
    "um",
    "um",
    "um"
  ],
  "SpaceDirections": [
    "none",
    [
      44.160000000000004,
      0.0,
      0.0
    ],
    [
      0.0,
      44.160000000000004,
      0.0
    ],
    [
      0.0,
      0.0,
      200
    ]
  ],
  "SliceThickness": 10.0,
  "SpaceOrigin": [
    -8633.28,
    -10576.32,
    -120100.0
  ]
}
```

#### Dataset lists
Since sections may be missing or require other comments, we include a tsv file in the same directory describing every slice in the dataset.  The required fields are sample_id and status, the latter should contain present or absent.  An example is shown below.

```
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
```


## Config files

In python, configuration options are passed as dictionaries.  For command line use these are stored in `json` files.  See the examples for examples of various parameters.

Here is an example with typical parameters.  When arguments are separated by commas, they refer to each iteration of a multi scale (coarse to fine) approach.

```
{
    "n_iter":[1000,200],
    "downI":[[4,4,4],[2,2,2]],
    "downJ":[[4,4,4],[2,2,2]],        
    "a":[200.0],
    "sigmaR":[5e6],
    "sigmaM":[2.0],
    "sigmaB":[4.0],
    "sigmaA":[6.0],
    "ev":[1e-0],
    "eA":[1e6],
    "priors":[[0.9,0.05,0.05]],
    "update_muA":[0],
    "update_muB":[0],
    "muB":[0.0],
    "update_sigmaM":[0],
    "update_sigmaA":[0],
    "update_sigmaB":[0],
    "order":[3],
    "n_draw":[50],
    "n_e_step":[3],    
    "v_start":[500,0]
}

```
## Spaces

### 3D spaces

#### Atlases

A typical workflow is to register new imaging data to a well characterized atlas, containing annotations.

##### Mouse

We use the allen CCF version 3


##### Human

We use MNI space atlases including those from mricloud.org.



#### Other images
We typically work with MRI containing various contrasts, CT, and cleared tissue microscopy.  Workflows will work best when these images are oriented the same way as the standard atlas, but this is not a requirement.

### 2D spaces
Datasets with serial sections are sampled in two different spaces.

#### Input histology space
Images are in the orientation of the originally acquired data.  Typically the position and orientation of each slice differs from its neighbors.

#### Registered histology space
In our mapping pipeline we apply a rigid transform to each histology slice, so that each slice aligns with its neighors, and the overall geometry matches the 3D dataset being registered.

Since no data was acquired in this space, it can be defined by various conventions.  We typically sample it with the same resolution as the input space, on a grid that is the maximum size of the input space in each dimension.


## Input arguments
We support pipelines for registering several datasets to each other, and reconstructing data from one dataset in the space of any other dataset.
### Names of spaces
Registrations are computed between pairs of spaces.  Each space should be given a unique name. (e.g. "atlas", "CT", "exvivoMRI","invivoMRI", "Histology").

### Names of images
Each space may have more than one imaging dataset sampled in it (for example multiple MRI scans with different contrasts).  Each image within a space should be given a unique name.  (e.g. "exvivoMRI -> T1", "exvivoMRI -> T2", "invivoMRI -> T1", "Histology")

### Filenames
Each image should have a filename (for 3D data), or a directory (for 2D data) associated to it.

### Registration tuples
To register a complex multimodal dataset, we specify a list of (space/image to map from, space/image to map to ) tuples. These correspond to edges in a graph and should span the set of spaces.  This set of transformations will be computed using our optimization procedure.

### Reconstruction tuples
After transformations are computed, we can reconstruct data from one space in any other space. Tuples of the form (space/image to map from, space to map to) are specified. Given the registration tuples, a path of transformations will be computed, which may involve the composition of more than one calculated transform. We can also choose to reconstruct each image in every other space instead of specifying each mapping with a tuple.

### Example
For example we can run registration and reconstruction with the command
```
python transformation_graph.py --infile <input json file>
```

Where the input json file contains

```
{
    "space_image_path": [["MRI", "masked", "/home/brysongray/data/MD816_mini/HR_NIHxCSHL_50um_14T_M1_masked.vtk"],
                       ["CCF", "average_template_50", "/home/brysongray/data/MD816_mini/average_template_50.vtk"],
                       ["MRI", "unmasked", "/home/brysongray/data/MD816_mini/HR_NIHxCSHL_50um_14T_M1.vtk"],
                       ["CT", "masked", "/home/brysongray/data/MD816_mini/ct_mask.vtk"],
                       ["HIST", "nissl", "/home/brysongray/data/MD816_mini/MD816_STIF_mini"]],
    "registrations": [[["HIST", "nissl"], ["MRI", "masked"]],
                       [["MRI", "masked"], ["CCF", "average_template_50"]],
                       [["MRI", "masked"], ["CT", "masked"]]],
    "configs": ["/home/brysongray/emlddmm/config787small.json",
                "/home/brysongray/emlddmm/configMD816_MR_to_CCF.json",
                "/home/brysongray/emlddmm/configMD816_MR_to_CT.json"],
    "output": "/home/brysongray/emlddmm/transformation_graph_outputs",
    "transforms": [[["HIST", "nissl"], ["CCF", "average_template_50"]],
                   [["MRI", "masked"], ["CT", "masked"]]],
    "transform_all": "False"
}
```

## Output data format
Output data structure contains transformations between pairs of named spaces (always), transformed images (suggested but not necessary), and other data types such as points and geojson annotations.

These pairs are organized in a hierarchical tree, where the parent directories contain data *in* a given space, and the child directories contain data *from* a given space.




### Notes 
1. Output raster data is stored using simple legacy vtk file format. 
1. Output point data is stored using simple legacy vtk file format, with polydata.
1. json is shown only for data from atlas to a 2D space.  
1. Meanxyz is shown only for a 2D space to the atlas. 
1. Transforms are stored as a rigid transformation matrix only for maps from a 2D space to another 2D space.
1. Note the “_to_” in the naming of transforms is opposite to images.  This is intentional.
1. Note that in 2D directories, image names are appended to space names for uniqueness, separated by an underscore. 
1. Qc figures are not standard, as they will vary by dataset
Other outputs

### Example
Example output data structure is shown here. Lists are used to show directory hierarchy:
This supports an arbitrary number of folders.

```
{Space i}
  {Space j}_to_{space i}
    Transforms (always)
      {space i}_to_{space j}_displacement.vtk (3D to 3D, or 3D to registered space, NOT 3D to input which does not exist as a displacement field)
      {space i}_{image k}_to_{space j}_{image k’}_matrix.txt (2D to 2D only)
      {space i}_{image k}_to_{space j}_displacement.vtk (i 2D to j 3D only)
    Images (suggested)
      {space j}_{image k}_to_{space i}.vtk
      {space j}_{image k}_to_{space i}_{image k’}.vtk (for 2D to 2D)
    Points (optional)
      {space j}_{image k}_detects_to_{space i}.vtk
    Json (for atlas only)
      Atlas_to_{space j}_{image k}.geojson
    Meanxyz (for atlas only)
      {space j}_{image k}_detects_to_atlas_meanxyz.txt
  Qc (optional)
    Composite_{image slice name}_QC.jpg
```


## Software Checklist
Source code is provided in the github repository at https://github.com/twardlab/emlddmm.  It has also been set up for use at https://twardlab.com/reg.  A guest account is provided with username guest and password  84983c60.

Two small datasets are included in the examples folder of the github repository.  Another small dataset is available for download from the website.





### System requirements
This software is cross platform and should run on any system that supports python.  The python interface has been tested on linux, windows, and mac using jupyter notebooks.  The command line interface has been tested only on linux.

Required libraries and their version numbers are listed in requirements.txt.

No non-standard hardware is required, but this library uses pytorch which can use gpu acceleration if a gpu is available.

### Installation guide

Installation instructions are typical for python modules, and can be found [here](https://twardlab.github.io/emlddmm/build/html/installation.html).

Installation consists of cloning source code from github (about 1 minute), and installing dependencies using pip (about 10 minutes).

### Demo
Examples are in the examples folder of the github repository, and are illustrated in our documentation [here](https://twardlab.github.io/emlddmm/build/html/examples.html).

Each example shows how to run the code interactively in a jupyter notebook, as well as how to save config files and run it from a linux command line.

The jupyter notebook produces visualizations but no outputs (mappings are stored as python variables in memory).  The commandline produces outputs written to disk following our output format [here](https://twardlab.github.io/emlddmm/build/html/output_specification.html).

The demos provided take less than 20 minutes to run on a desktop computer with no gpu accelleration.

### Instructions for use.

Full documentation is located [here](https://twardlab.github.io/emlddmm/build/html/index.html).  Workflows for specific examples are shown in the examples section.