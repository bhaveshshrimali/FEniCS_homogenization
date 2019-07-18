from dolfin import * 
import numpy as np 
import os 
import pickle as pkl
set_log_level(30)
parameters["linear_algebra_backend"] = "PETSc"
# parameters['krylov_solver']['error_on_nonconvergence'] = False 
# parameters['krylov_solver']['maximum_iterations'] = 3000  #see the best number of iterations
parameters['krylov_solver']['monitor_convergence'] = True
parameters["form_compiler"]["cpp_optimize"] = True
ffc_options = {"optimize": True, \
               "eliminate_zeros": True, \
               "precompute_basis_const": True, \
               "precompute_ip_const": True}
comm = MPI.comm_world
cdir = os.getcwd()
msh_path = os.path.join(cdir,'MeshFile.xml')
# msh_path = '/projects/meca/bshrima2/FEniCSPlates/3DHomogenizationDolfin/MeshFile.xml'
# msh_path = 'MeshFile.xml'
msh = Mesh(msh_path)
L=1.
Lx= np.sqrt(3.)*L
Ly = 2.*L 
Lz = L
vol_of_solid = Lx * Ly * Lz
vertices = np.array([[0.,0,0],   # 0: Origin
                     [Lx,0,0],   # 1: Right
                     [0,Ly,0],   # 2: Top
                     [0,0,Lz]])  # 3

"""Victor's Mapping """ 

class PeriodicBoundary(SubDomain):
    def __init__(self, vertices):
        """ vertices stores the coordinates of the 4 unit cell corners"""
        SubDomain.__init__(self)
        self.vv = vertices
        self.a1 = self.vv[1,:]-self.vv[0,:] # first vector generating periodicity
        self.a2 = self.vv[2,:]-self.vv[0,:] # second vector generating periodicity
        self.a3 = self.vv[3,:]-self.vv[0,:] # third vector generating periodicity
        # check if UC vertices form indeed a parallelogram
        # assert np.linalg.norm(self.vv[2, :]-self.vv[3, :] - self.a1) <= 1e-8
        # assert np.linalg.norm(self.vv[2, :]-self.vv[1, :] - self.a2) <= 1e-8
        
    def inside(self, x, on_boundary):         #also for firedrake you do not explicitly need near function (just check using default machine-eps directly)
        # return True if on left or front or bottom boundary AND NOT on one of the edges with right, back, or top faces
        left=bool(near(x[0],self.vv[0,0]))               #doesn't near by default return a "bool" why bool-ing it again ?
        front=bool(near(x[1],self.vv[0,1]))          
        bottom=bool(near(x[2],self.vv[0,0]))
        right=bool(near(x[0],self.vv[1,0]))
        back=bool(near(x[1],self.vv[2,1]))
        top=bool(near(x[2],self.vv[3,2]))
                
        return bool((left or front or bottom) and 
                    (not ((left and top) or (left and back) or (front and right) or (front and top) or (bottom and back) or (bottom and right))) and on_boundary)

    def map(self, x, y):
        if near(x[0], self.vv[1,0]) and near(x[1], self.vv[2,1]) and near(x[2], self.vv[3,2]): # if on right-back-top corner
            y[0] = x[0] - (self.a1[0]+self.a2[0]+self.a3[0])
            y[1] = x[1] - (self.a1[1]+self.a2[1]+self.a3[1])
            y[2] = x[2] - (self.a1[2]+self.a2[2]+self.a3[2])
        elif near(x[0], self.vv[1,0]) and near(x[1], self.vv[2,1]): # if on right-back edge
            y[0] = x[0] - (self.a1[0]+self.a2[0])
            y[1] = x[1] - (self.a1[1]+self.a2[1])
            y[2] = x[2] - (self.a1[2]+self.a2[2])
        elif near(x[1], self.vv[2,1]) and near(x[2], self.vv[3,2]): # if on back-top edge
            y[0] = x[0] - (self.a2[0]+self.a3[0])
            y[1] = x[1] - (self.a2[1]+self.a3[1])
            y[2] = x[2] - (self.a2[2]+self.a3[2])
        elif near(x[2], self.vv[3,2]) and near(x[0], self.vv[1,0]): # if on top-right edge
            y[0] = x[0] - (self.a3[0]+self.a1[0])
            y[1] = x[1] - (self.a3[1]+self.a1[1])
            y[2] = x[2] - (self.a3[2]+self.a1[2])
        elif near(x[0], self.vv[1,0]): # if on right boundary
            y[0] = x[0] - self.a1[0]
            y[1] = x[1] - self.a1[1]
            y[2] = x[2] - self.a1[2]
        elif near(x[1], self.vv[2,1]): # if on back boundary
            y[0] = x[0] - self.a2[0]
            y[1] = x[1] - self.a2[1]
            y[2] = x[2] - self.a2[2]
        elif near(x[2], self.vv[3,2]): # if on top boundary
            y[0] = x[0] - self.a3[0]
            y[1] = x[1] - self.a3[1]
            y[2] = x[2] - self.a3[2]
        else:
            y[0] = -1.
            y[1] = -1.
            y[2] = -1.

       
class corner(SubDomain):
    def inside(self,x, on_boundary):
        return near(x[0], 0.) and near(x[1], 0.) and near(x[2], 0.)
        
class right(SubDomain):
    def inside(self,x, on_boundary):
        return near(x[0], Lx) and near(x[1], 0.) and near(x[2], 0.)       
        
class back(SubDomain):
    def inside(self,x, on_boundary):
        return near(x[0], 0.) and near(x[1], Ly) and near(x[2], 0.)     
        
class top(SubDomain):
    def inside(self,x, on_boundary):
        return near(x[0], 0.) and near(x[1], 0.) and near(x[2], Lz) 



""" Mapping Ends """
# class used to define the periodic boundary map
# class PeriodicBoundary(SubDomain):
#     def __init__(self, vertices, tolerance=DOLFIN_EPS):
#         """ vertices stores the coordinates of the 4 unit cell corners"""
#         SubDomain.__init__(self, tolerance)
#         self.tol = tolerance
#         self.vv = vertices
#         self.a1 = self.vv[1,:]-self.vv[0,:] # first vector generating periodicity
#         self.a2 = self.vv[2,:]-self.vv[0,:] # second vector generating periodicity 
#         self.a3 = self.vv[3,:]-self.vv[0,:] # third vector generating periodicity

        
#     def inside(self, x, on_boundary):
#         """
#             return True if on left, bottom, or back faces
#             and not on one of the top, front or right faces 
#             or associate edges (and vertices) as defined below 
#         """
#         # faces
#         left = near(x[0],self.vv[0,0]) 
#         bottom = near(x[1],self.vv[0,1]) 
#         back = near(x[2],self.vv[0,2])
#         right = near(x[0],self.vv[1,0])
#         top = near(x[1],self.vv[2,1])
#         front = near(x[2],self.vv[3,2])

#         # line-segments (bottom 4; top 4; vertical 4)
#         bottom_front = bottom and front 
#         bottom_right = bottom and right 
        
#         top_left = top and left 
#         top_back = top and back 

#         left_front = left and front 
#         right_back = right and back 

#         return bool((left or back or bottom) and 
#                     (not( (top_left) or (left_front) or (top_back) or (right_back) or (bottom_right) or (bottom_front))) and on_boundary)

#         # return bool( bool(left and not((top_left) or (left_front))) or bool(back and not((top_back) or (right_back))) or 
#         #         bool(bottom and not((bottom_right) or (bottom_front))) and on_boundary )

#     def map(self, x, y):
#         """ Mapping the right boundary to left and top to bottom"""
        
#         # faces
#         right = near(x[0],self.vv[1,0])
#         top = near(x[1],self.vv[2,1])
#         front = near(x[2],self.vv[3,2])


#         # line-segments 
#         top_right = top and right 
#         top_front = top and front 
#         right_front = right and front 
#         point_6 = right and front and top 

#         if point_6:
#             y[0] = x[0] - (self.a1[0] + self.a2[0] + self.a3[0])
#             y[1] = x[1] - (self.a1[1] + self.a2[1] + self.a3[1])
#             y[2] = x[2] - (self.a1[2] + self.a2[2] + self.a3[2])
#         elif top_right:
#             y[0] = x[0] - (self.a1[0] + self.a2[0])
#             y[1] = x[1] - (self.a1[1] + self.a2[1])
#             y[2] = x[2] - (self.a1[2] + self.a2[2])
#         elif top_front:
#             y[0] = x[0] - (self.a2[0] + self.a3[0])
#             y[1] = x[1] - (self.a2[1] + self.a3[1])
#             y[2] = x[2] - (self.a2[2] + self.a3[2])
#         elif right_front: 
#             y[0] = x[0] - (self.a1[0] + self.a3[0])
#             y[1] = x[1] - (self.a1[1] + self.a3[1])
#             y[2] = x[2] - (self.a1[2] + self.a3[2])
#         elif right:
#             y[0] = x[0] - (self.a1[0])
#             y[1] = x[1] - (self.a1[1])
#             y[2] = x[2] - (self.a1[2])
#         elif front:
#             y[0] = x[0] - (self.a3[0])
#             y[1] = x[1] - (self.a3[1])
#             y[2] = x[2] - (self.a3[2])
#         elif top:
#             y[0] = x[0] - (self.a2[0])
#             y[1] = x[1] - (self.a2[1])
#             y[2] = x[2] - (self.a2[2])
#         else: 
#             y[0] = -1. 
#             y[1] = -1. 
#             y[2] = -1. 

# class OriginPoint(SubDomain):  # Point 0
#     def __init__(self, vertices,tolerance=DOLFIN_EPS):
#         SubDomain.__init__(self, tolerance)
#         self.vv = vertices

#     def inside(self, x,  on_boundary):
#         return near(x[0],0.) and near(x[1],0.) and near(x[2],0.)

# class bottomright(SubDomain):  # Point 1
#     def __init__(self, vertices,tolerance=DOLFIN_EPS):
#         SubDomain.__init__(self, tolerance)
#         self.vv = vertices

#     def inside(self, x,  on_boundary):
#         Lx = np.sqrt(3.)*L
#         return near(x[0],Lx) and near(x[1],0.) and near(x[2],0.) 

# class topleft(SubDomain):   # Point 3
#     def __init__(self, vertices,tolerance=DOLFIN_EPS):
#         SubDomain.__init__(self, tolerance)
#         self.vv = vertices

#     def inside(self, x,  on_boundary):
#         Ly = 2.*L
#         return near(x[0], 0.) and near(x[1],Ly) and near(x[2],0.)

# class bottomfront(SubDomain):   # Point 3
#     def __init__(self, vertices,tolerance=DOLFIN_EPS):
#         SubDomain.__init__(self, tolerance)
#         self.vv = vertices

#     def inside(self, x,  on_boundary):
#         Lz = L
#         return near(x[0], 0.) and near(x[1],Ly) and near(x[2],Lz)

def strain2voigt(eps):
    return as_vector([eps[0, 0], eps[1, 1], eps[2, 2], 2*eps[0, 1], 2*eps[0, 2], 2*eps[1, 2] ])

def stress2Voigt(s):
    return as_vector([s[0, 0], s[1, 1], s[2, 2], s[0, 1], s[0, 2], s[1, 2] ])

def voigt2stress(S):
    ss = [[S[0], S[3], S[4]],
          [S[3], S[1], S[5]],
          [S[4], S[5], S[2]]]
    return as_tensor(ss)

def macro_strain(i,scale):
    """returns the macroscopic curvature for the 3 elementary cases"""
    Gamm_Voight = np.zeros(6)
    Gamm_Voight[i] = 1.*scale
    print(Gamm_Voight[0])
    return np.array([[Gamm_Voight[0],    Gamm_Voight[3]/2., Gamm_Voight[4]/2.],
                     [Gamm_Voight[3]/2., Gamm_Voight[1],    Gamm_Voight[5]/2.],
                     [Gamm_Voight[4]/2., Gamm_Voight[5]/2., Gamm_Voight[2]]])

def eps(v):
    return sym(grad(v))

def sigma(v, Eps):
    E, nu = material_parameters   #avoid using global variables 
    lmbda = E*nu/(1+nu)/(1-2*nu)
    mu = E/2./(1+nu)
    return lmbda*tr(eps(v) + Eps)*Identity(3) + 2*mu*(eps(v)+Eps)

""" Instantiating the corner-classes """
# bot_right = bottomright(vertices)
# orgn = OriginPoint(vertices)
# top_lft = topleft(vertices)
# bot_front = bottomfront(vertices)

orgn = corner()
bot_right = right()
top_lft = back()
bot_front = top()


"""
`Mesh` and Material parameters:
`Emat` and `nu_mat` for the matrix
`Eh` and `nu_h` for the inclusion """
tol_geom = 1.e-6
deg = 1
nu_mat = 0.25
Emat = 2.2e9 #  2.*mu_m*(1+nu_mat)
material_parameters = [Emat, nu_mat]
nu_h = 0.  #.4 
# cdir = '/home/bshrima2/PlatesTrial/'   # name of the binding directory in singularity
Gamm_bar = Constant(((0, 0, 0), (0, 0, 0), (0, 0, 0)))
scl = 1.e-2
L_hom = np.zeros((6,6))
fname_Ltil = os.path.join(cdir,'Ltil.csv')


Ue = VectorElement('CG',msh.ufl_cell(),deg)
Re = VectorElement('R',msh.ufl_cell(),0)
We = MixedElement([Ue,Re])
Ve = FunctionSpace(msh,We,constrained_domain=PeriodicBoundary(vertices))

# du = TestFunction(Ve)
# u_ = TrialFunction(Ve)
# u = Function(Ve)
du, dlamb = TestFunctions(Ve)
u_, lamb_ = TrialFunctions(Ve)
u_lamb = Function(Ve)


# bc1 = DirichletBC(Ve,Constant((0.,0.,0)),orgn,method='pointwise')
# bc2 = DirichletBC(Ve.sub(1),Constant(0.),bot_right,method='pointwise')
# bc3 = DirichletBC(Ve.sub(2),Constant(0.),top_lft,method='pointwise')
# bc4 = DirichletBC(Ve.sub(0),Constant(0.),bot_front,method='pointwise')
# bcs = [bc1,bc2,bc3,bc4]
# bcs = [bc1]
bcs = []
a_mu_v = inner(sigma(u_,Gamm_bar),eps(du))*dx 
a_mu_v += (inner(lamb_,du) + inner(dlamb,u_))*dx

L_w, f_w = lhs(a_mu_v), rhs(a_mu_v)
# y = SpatialCoordinate(msh)
for j,case in enumerate(['xx','yy','zz','xy','xz','yz']):
    Gamm_bar.assign(Constant(macro_strain(j,scl)))
    # print(Constant(macro_strain(j,scl)))
    # solve(L_w == f_w, u, bcs)
    # solve(L_w == f_w, u_lamb, bcs) 
    solve(L_w == f_w, u_lamb, bcs, solver_parameters={'linear_solver':'gmres','preconditioner':'amg'}) # try for parallel
    u_,lamb_ = u_lamb.split(True)
    y = SpatialCoordinate(msh)
    # u_full = u + dot(Gamm_bar,y)
    # w,thta,lamb_w,lamb_thta = w_thta.split(True) 
    sigma_til = np.zeros((6,))
    Eps_til = np.zeros((6,))
    if case == 'xx':
        strnvals = assemble(strain2voigt(eps(u_))[0]*dx)/vol_of_solid
        print(strnvals)
    # print(Gamm_bar[0,0].values)
    for k in range(sigma_til.shape[0]):
        sigma_til[k] = float(assemble(stress2Voigt(sigma(u_,Gamm_bar))[k]*dx))/vol_of_solid
        Eps_til[k] = float(assemble(strain2voigt(eps(u_)+Gamm_bar)[k]*dx))/vol_of_solid
    L_hom[j, :] = sigma_til.copy()/scl

    fname_cuv = os.path.join(cdir,'Eps_{}.csv'.format(case))
    fname_mom = os.path.join(cdir,'Stil_{}.csv'.format(case))

    np.savetxt(fname_cuv,Eps_til)
    np.savetxt(fname_mom,sigma_til)

    # Vt = FunctionSpace(msh,VectorElement('CG',msh.ufl_cell(),deg))
    # u_plot = Function(Vt)
    # u_plot.assign(project(u_full, Vt))
    # # thta_plot.assign(project(thta_full, Vt))

    # xdmf_fname = os.path.join(cdir,'data_vals_{}.xdmf'.format(case))

    # with XDMFFile(comm,xdmf_fname) as res_fil:
    #     res_fil.parameters["flush_output"] = True
    #     res_fil.parameters["functions_share_mesh"] = True
    #     res_fil.write(u_plot,0)
np.savetxt(fname_Ltil,L_hom,delimiter=',')
# Btil = np.loadtxt(fname_Btil,delimiter=',')
# Mhomogenized[:,:,ishp,i_alph,i_fo] = Btil.copy() 