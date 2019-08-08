# FEniCS_homogenization
A collection of homogenization scripts for linear elasticity. See https://comet-fenics.readthedocs.io/en/latest/index.html for more details. This repository is inspired by the corresponding 2D calculations in `COMET` and, therefore, much of the code follows the style used there. 


# Some sample cases
The above code `3dtest.py` runs elementary load cases (6 of them in 3D) to numerically determine the effective modulus (6 x 6 tensor). See for instance, below, a "simple cubic" unit cell with a spherical void placed at the geometric center of the cube. The corresponding displacement(s) for uniaxial tension along `x` and a simple shear along the `x-z` directions, respectively, are plotted below.  

![axial tension](/Images/xx_clipped.png "Axial strain of 0.01")![simple shear](/Images/xz_clipped.png "Axial strain of 0.01"){:height="50%" width="50%"}

