Installation
============


Clone the repository::

   git clone github.com/twardlab/emlddmm

Change to the directory emlddmm has been cloned to and install the requirements::
   
   pip install -r requirements.txt
   
When running interactively in python, add the appropriate path::

   import sys
   sys.path.append('PATH_TO_EMLDDMM_LIBRARY')

When running from command line::

   python PATH_TO_EMLDDMM_LIBRARY/transformation_graph_v01.py --in INPUT_FILENAME
   
For details on the command line interface, see :doc:`input specification <input_specification>`.
   


   


