from netgen.csg import *

left = Plane(Pnt(0,0,0),Vec(-1,0,0))
bottom = Plane(Pnt(0,0,0),Vec(0,-1,0))
back = Plane(Pnt(0,0,0),Vec(0,0,-1))
right = Plane(Pnt(1,0,0),Vec(1,0,0))
top = Plane(Pnt(0,1,0),Vec(0,1,0))
front = Plane(Pnt(0,0,1),Vec(0,0,1))

cube = left * right * bottom * top * back * front
cube_w_hole = cube - Sphere(Pnt(0.5,0.5,0.5),0.2)
geo = CSGeometry()
geo.Add(cube_w_hole)
geo.PeriodicSurfaces(left,right)
geo.PeriodicSurfaces(bottom,top)
geo.PeriodicSurfaces(back,front)

from ngsolve import * 
msh = geo.GenerateMesh(maxh=0.05)
msh.Export('test_mesh.msh','Gmsh2 Format')