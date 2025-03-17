''' test_emlddmm.py: Unit testing for the emlddmm module.

This includes unit tests for methods in the emlddmm.py package.

Example
-------
Running the unit tests::

    pytest test_emlddmm.py
    



'''

import pytest
import sys
sys.path.insert(0, '/home/brysongray/emlddmm')
import emlddmm
import numpy as np
import os

# test input output functions

# construct binary 3D ellipsoid
@pytest.fixture
def ellipsoid(scope="module"):
    ni = 100
    nj = 120
    nk = 110
    xI = [np.arange(ni)-(ni-1)/2,np.arange(nj)-(nj-1)/2,np.arange(nk)-(nk-1)/2]
    XI = np.stack(np.meshgrid(xI[0],xI[1],xI[2], indexing='ij'))
    condition = lambda x,a,b,c : x[0]**2 / a**2 + x[1]**2 / b**2 + x[2]**2 / c**2
    a = 15
    b = 30
    c = 20
    I = np.where(condition(XI,a,b,c) < 1.0, 1.0, 0.0)[None]

    return xI, I

@pytest.mark.io
def test_read_write_vtk_data(tmp_path, ellipsoid):
    xI, I = ellipsoid
    title = 'ellipsoid'
    # write out image in vtk format
    emlddmm.write_vtk_data(os.path.join(tmp_path,'ellipsoid.vtk'), xI, I, title)
    _,J,_,_ = emlddmm.read_vtk_data(os.path.join(tmp_path,'ellipsoid.vtk'))
    assert os.path.exists(os.path.join(tmp_path,'ellipsoid.vtk'))
    assert np.allclose(J,I)

@pytest.mark.io
@pytest.mark.parametrize("ext", [
    ".nii",
    ".vtk"
])
def test_read_write_data(tmp_path, ellipsoid, ext):
    xI, I = ellipsoid
    title = 'ellipsoid'
    # write out image in vtk format
    emlddmm.write_data(os.path.join(tmp_path,'ellipsoid'+ext), xI, I, title)
    _,J,_,_ = emlddmm.read_data(os.path.join(tmp_path,'ellipsoid'+ext))
    assert os.path.exists(os.path.join(tmp_path,'ellipsoid'+ext))
    assert np.allclose(J,I)
