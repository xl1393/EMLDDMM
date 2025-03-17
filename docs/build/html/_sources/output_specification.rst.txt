Output Specification
====================


Our output data structure contains transformations between pairs of named spaces (always), transformed images (suggested but not necessary), and other data types such as points and geojson annotations.


These pairs are organized in a hierarchical tree, where the parent directories contain data in a given space, and the child directories contain data from a given space.



Example
^^^^^^^

Example output data structure is shown here. Lists are used to show directory hierarchy: This supports an arbitrary number of folders.::

    {Space i}
      {Space j}_to_{space i}
        Transforms (always)
          {space i}_to_{space j}_displacement.vtk (3D to 3D, or 3D to registered space, NOT 3D to input which does not exist as a displacement field)
          {space i}_{image k}_to_{space j}_{image k’}_matrix.txt (2D to 2D only)
          {space i}_{image k}_to_{space j}_displacement.vtk (i 2D to 3D only)
        Images (suggested)
          {space j}_{image k}_to_{space i}.vtk
          {space j}_{image k}_to_{space i}_{image k’}.vtk (for 2D to 2D)
        Points (optional)
          {space j}_{image k}_detects_to_{space i}.vtk
        Json (for atlas only)
          Atlas_to_{space j}_{image k}.geojson
        Meanxyz (for atlas only)
          {space j}_{image k}_detects_to_atlas_meanxyz.txt
      QC (optional)
        Composite_{image slice name}_QC.jpg


Notes
^^^^^
Some important notes are below:

#. Output raster data is stored using simple legacy vtk file format (see :ref:`here <vtkref>`).
#. Output point data is stored using simple legacy vtk file format, with polydata.
#. json is shown only for data from atlas to a 2D space.
#. Mean xyz is shown only for a 2D space to the atlas.
#. Transforms are stored as a rigid transformation matrix only for maps from a 2D space to another 2D space.
#. Note the “to” in the naming of transforms is opposite to images. This is intentional.
#. Note that in 2D directories, image names are appended to space names for uniqueness, separated by an underscore.
#. QC figures are not standard, as they will vary by dataset.


