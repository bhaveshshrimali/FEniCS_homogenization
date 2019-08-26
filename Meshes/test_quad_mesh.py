import numpy as np 
import os, sys, meshio
import netgen.gui
from netgen.geom2d import SplineGeometry
from dolfin import *
import ngsolve

comm = MPI.comm_world

set_log_level(10)
# geo = SplineGeometry()
# geo.AddRectangle(
#     p1 = (0.,0),
#     p2 = (2.,4),
#     leftdomain = 1,
#     rightdomain = 0
# )
# geo.AddCircle(
#     c = (1.,2),
#     r = 0.5,
#     leftdomain = 0,
#     rightdomain = 1
# )

# ngmsh = geo.GenerateMesh(maxh=0.1,quad_dominated= True)
# ngs_mesh = ngsolve.Mesh(ngmsh)
# elems = ngmsh.Elements2D()
# nodes = ngs_mesh.vertices
# mshPath 
mio_mesh = meshio.read('test_quadMesh.xdmf')
verts_mio = mio_mesh.cells['quad']
points_mio = mio_mesh.points

meshio.write_points_cells(
    "trial_msh.vtk",
    points_mio,
    mio_mesh.cells,
)
# print(type(mio_mesh.cells['quad']))
# ngmsh.Export('trialMesh.msh','Gmsh2 Format')

num_points = sum([1 for i in points_mio])
num_cells = sum([1 for i in verts_mio])
print(num_cells)
print(num_points)

msh = Mesh() 
editr = MeshEditor()
editr.open(msh,'quadrilateral',2,2) # topological and geometric dimensions
editr.init_vertices(num_points)
editr.init_cells(num_cells)


for i,v in enumerate(points_mio):
    editr.add_vertex(i,[*v])

for j,e in enumerate(verts_mio):
    # list_of_nodes = [int(i) for i in np.array(e.points,int)]
    print(e.astype(np.uintp))
    editr.add_cell(j,e.astype(np.uintp)[[0,1,3,2]])

editr.close()

with XDMFFile(comm,'trial.xdmf') as f:
    f.write(msh)