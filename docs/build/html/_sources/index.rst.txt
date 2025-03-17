.. GDM documentation master file, created by
   sphinx-quickstart on Tue Apr 19 09:18:59 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to GDM's documentation!
===============================

Generative diffeomorphic mapping (GDM) is a deformable image registration algorithm designed for aligning multimodal neuroimaging datasets to one another for subsequent analysis.  Our package has several important novel features including estimation of any differences in contrast or color between datasets, identification of missing tissues or artifactions, diverse geometries such as mapping 3D volumes to a sequence of 2D datasets, and complex multimodality registration setups described by transformation graphs.

This documentation can automatically be rendered as a pdf using the sphinx package but is best viewed in html (https://twardlab.github.io/emlddmm/build/html/index.html).  The pdf may have missing or broken links and images.

.. toctree::
   :maxdepth: 1
   :caption: Contents:
   
   introduction
   installation
   coordinate_systems
   file_formats
   input_specification
   output_specification
   examples   
   documentation   
   todo





Installing
==========
* Make sure you have python 3 installed with pip
* Use pip to install the packages in the requirements.txt file (pip in stall -r requrements.txt)
* Clone the repository on github (git clone https://github.com/twardlab/emlddmm)
* When running interactively in python, make sure you add the path (import sys; sys.path.append('/LOCATION/OF/REPOSITORY'))

Examples 
========
We include data and code for two examples, in the "examples" folder.  
Both examples show code run interactively in a jupyter notebook, and show how the command line interface is used.

* 3D Human MRI example
* Mouse Nissl serial section alignment example.

Important Functions
===================

Python functions to be run interactively
----------------------------------------
* :py:func:`emlddmm.emlddmm`: Run the emlddmm algorithm.
* :py:func:`emlddmm.emlddmm_multiscale`: Run the emlddmm algorithm iteratively at different scales.  


Command line functions
----------------------
* :py:mod:`transformation_graph_v01`: Run registration between two or more datasets using the transformation graph command line interface.  



Module and function documentation
=================================

All functions are automatically documented with sphinx and napoleon.  See the Function Reference section.


Web interface
=============
To improve accessibility, we provide a web interface at https://twardlab.com/reg which can be used for small jobs.  
Please email the Daniel Tward (dtward@mednet.ucla.edu) to request an account.  
We provide a guest account for review purposes.  Username: guest, password: 84983c60.  No identifying information is recorded in the guest account.  We request email addresses when creating your own account.