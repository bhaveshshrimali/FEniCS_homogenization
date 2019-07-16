from dolfin import * 
import numpy as np 
import os 
import pickle as pkl
set_log_level(30)
parameters["linear_algebra_backend"] = "PETSc"
parameters['krylov_solver']['error_on_nonconvergence'] = False 
parameters['krylov_solver']['maximum_iterations'] = 3000  #see the best number of iterations
parameters['krylov_solver']['monitor_convergence'] = True
parameters["form_compiler"]["cpp_optimize"] = True
ffc_options = {"optimize": True, \
               "eliminate_zeros": True, \
               "precompute_basis_const": True, \
               "precompute_ip_const": True}
comm = MPI.comm_world

msh_path = '/projects/meca/bshrima2/FEniCSPlates/3DHomogenization/MeshFile.xml'
msh = Mesh(msh_path)
# class used to define the periodic boundary map
class PeriodicBoundary(SubDomain):
    def __init__(self, vertices, tolerance=DOLFIN_EPS):
        """ vertices stores the coordinates of the 4 unit cell corners"""
        SubDomain.__init__(self, tolerance)
        self.tol = tolerance
        self.vv = vertices
        self.a1 = self.vv[1,:]-self.vv[0,:] # first vector generating periodicity
        self.a2 = self.vv[3,:]-self.vv[0,:] # second vector generating periodicity 
        self.a3 = self.vv[4,:]-self.vv[0,:] # third vector generating periodicity

        
    def inside(self, x, on_boundary):
        """
            return True if on left, bottom, or back faces
            and not on one of the top, front or right faces 
            or associate edges (and vertices) as defined below 
        """
        # faces
        left = near(x[0],self.vv[0,0]) 
        bottom = near(x[1],self.vv[0,1]) 
        back = near(x[2],self.vv[0,2])
        right = near(x[0],self.vv[1,0])
        top = near(x[1],self.vv[3,1])
        front = near(x[2],self.vv[4,2])

        # line-segments (bottom 4; top 4; vertical 4)
        bottom_front = bottom and front 
        bottom_right = bottom and right 
        
        top_left = top and left 
        top_back = top and back 

        left_front = left and front 
        right_back = right and back 

        
        return ( bool(left and not(top_left or left_front)) or bool(back and not(top_back or right_back)) or 
                bool(bottom and not(bottom_right or bottom_front)))

    def map(self, x, y):
        """ Mapping the right boundary to left and top to bottom"""
        
        # faces
        right = near(x[0],self.vv[1,0])
        top = near(x[1],self.vv[3,1])
        front = near(x[2],self.vv[4,2])

        # line-segments 
        top_right = top and right 
        top_front = top and front 
        right_front = right and front 
        point_6 = right and front and top 

        if point_6:
            y[0] = x[0] - (self.a1[0] + self.a2[0] + self.a3[0])
            y[1] = x[1] - (self.a1[1] + self.a2[1] + self.a3[1])
            y[2] = x[2] - (self.a1[2] + self.a2[2] + self.a3[2])
        elif top_right:
            y[0] = x[0] - (self.a1[0] + self.a2[0])
            y[1] = x[1] - (self.a1[1] + self.a2[1])
            y[2] = x[2] - (self.a1[2] + self.a2[2])
        elif top_front:
            y[0] = x[0] - (self.a2[0] + self.a3[0])
            y[1] = x[1] - (self.a2[1] + self.a3[1])
            y[2] = x[2] - (self.a2[2] + self.a3[2])
        elif right_front: 
            y[0] = x[0] - (self.a1[0] + self.a3[0])
            y[1] = x[1] - (self.a1[1] + self.a3[1])
            y[2] = x[2] - (self.a1[2] + self.a3[2])
        elif right and not(top_right or right_front):
            y[0] = x[0] - (self.a1[0])
            y[1] = x[1] - (self.a1[1])
            y[2] = x[2] - (self.a1[2])
        elif front and not(right_front or top_front):
            y[0] = x[0] - (self.a3[0])
            y[1] = x[1] - (self.a3[1])
            y[2] = x[2] - (self.a3[2])
        elif top and not(top_right or top_front):
            y[0] = x[0] - (self.a2[0])
            y[1] = x[1] - (self.a2[1])
            y[2] = x[2] - (self.a2[2])
        else: 
            y[0] = -1. 
            y[1] = -1. 
            y[2] = -1. 

class OriginPoint(SubDomain):  # Point 0
    def __init__(self, vertices,tolerance=DOLFIN_EPS):
        SubDomain.__init__(self, tolerance)
        self.vv = vertices

    def inside(self, x,  on_boundary):
        return near(x[0],0.) and near(x[1],0.) and near(x[2],0.)

class bottomright(SubDomain):  # Point 1
    def __init__(self, vertices,tolerance=DOLFIN_EPS):
        SubDomain.__init__(self, tolerance)
        self.vv = vertices

    def inside(self, x,  on_boundary):
        Lx = self.vv[:,0].max()
        return near(x[0],Lx) and near(x[1],0.) and near(x[2],0.) 

class topleft(SubDomain):   # Point 3
    def __init__(self, vertices,tolerance=DOLFIN_EPS):
        SubDomain.__init__(self, tolerance)
        self.vv = vertices

    def inside(self, x,  on_boundary):
        Ly = self.vv[:,1].max()
        return near(x[0], 0.) and near(x[1],Ly) and near(x[2],0.)

class bottomfront(SubDomain):   # Point 3
    def __init__(self, vertices,tolerance=DOLFIN_EPS):
        SubDomain.__init__(self, tolerance)
        self.vv = vertices

    def inside(self, x,  on_boundary):
        Lz = self.vv[:,2].max()
        return near(x[0], 0.) and near(x[1],Ly) and near(x[2],Lz)

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
    Gamm_Voigt = np.zeros((6,))
    Gamm_Voigt[i] = 1.*scale
    return np.array([[Gamm_Voight[0], Gamm_Voight[3], Gamm_Voight[4]],
                     [Gamm_Voight[3], Gamm_Voight[1], Gamm_Voight[5]],
                     [Gamm_Voight[4], Gamm_Voight[5], Gamm_Voight[2]]])

def eps(v):
    return sym(grad(v))

def sigma(v, Eps):
    E, nu = material_parameters   #avoid using global variables 
    lmbda = E*nu/(1+nu)/(1-2*nu)
    mu = E/2/(1+nu)
    return lmbda*tr(eps(v) + Eps)*Identity(3) + 2*mu*(eps(v)+Eps)


L=1.
Lx= np.sqrt(3.)*L
Ly = 2.*L 
Lz = L
vol_of_solid = Lx * Ly * Lz
vertices = np.array([[0.,0,0],   # 0: Origin
                     [Lx,0,0],   # 1: Right
                     [Lx,Ly,0],  # 2
                     [0,Ly,0],   # 3: Top
                     [0,0,Lz],   # 4: Front
                     [Lx,0,Lz],  # 5
                     [Lx,Ly,Lz], # 6
                     [0,Ly,Lz]]) # 7
""" Instantiating the corner-classes """
bot_right = bottomright(vertices)
orgn = OriginPoint(vertices)
top_lft = topleft(vertices)
bot_front = bottomfront(vertices)
"""
`Mesh` and Material parameters:
`Emat` and `nu_mat` for the matrix
`Eh` and `nu_h` for the inclusion """
tol_geom = 1.e-6
deg = 2
nu_mat = 0.46
Emat = 2.2e9 #  2.*mu_m*(1+nu_mat)
material_parameters = [Emat,nu_mat]
nu_h = 0.  #.4 
cdir = '/home/bshrima2/PlatesTrial/'   # name of the binding directory in singularity
Gamm_bar = Constant(((0., 0., 0.), (0., 0., 0.), (0.,0.,0) ))
scl = 1.e-2
Evals = Constant(Emat)
nuvals = Constant(nu_mat)
L_hom = np.zeros((6,6))
fname_Ltil = os.path.join(cdir,'Ltil.csv')
We = VectorElement('CG',msh.ufl_cell(),deg)
Ve = FunctionSpace(msh,We,constrained_domain=PeriodicBoundary(vertices))
du = TestFunction(Ve)
u_ = TrialFunction(Ve)
u = Function(Ve)
bc1 = DirichletBC(Ve,Constant((0.,0.,0)),orgn,method='pointwise')
bc2 = DirichletBC(Ve.sub(1),Constant(0.),bot_right,method='pointwise')
bc3 = DirichletBC(Ve.sub(2),Constant(0.),top_lft,method='pointwise')
bc4 = DirichletBC(Ve.sub(0),Constant(0.),bot_front,method='pointwise')
bcs = [bc1,bc2,bc3,bc4]
a_mu_v = inner(sigma(u_,Gamm_bar),eps(du))*dx 
# # a_mu_v += inner(lamb_w,w_)*dx + inner(lamb_w_,w)*dx 
# # a_mu_v += inner(lamb_thta,thta_)*dx + inner(lamb_thta_,thta)*dx

L_w, f_w = lhs(a_mu_v), rhs(a_mu_v)
y = SpatialCoordinate(msh)
for (j, case) in enumerate(["xx", "yy", 'zz', 'xy', 'xz', "yz"]):
    Gamm_bar.assign(Constant(macro_strain(j,scl)))
    solve(L_w == f_w, u, bcs) 
    y = SpatialCoordinate(msh)
    u_full = u + dot(Gamm_bar,y)
    # solve(L_w_thta == f_thta, w_thta, [], solver_parameters={'linear_solver':'gmres','preconditioner':'petsc_amg'}) # try for parallel
    # w,thta,lamb_w,lamb_thta = w_thta.split(True) 
    sigma_til = np.zeros((6,))
    Eps_til = np.zeros((6,))
    
    for k in range(sigma_til.shape[0]):
        sigma_til[k] = float(assemble(stress2Voigt(sigma(u,Gamm_bar))[k]*dx))/vol_of_solid
        Eps_til[k] = float(assemble(strain2voigt(eps(u)+Gamm_bar)[k]*dx))/vol_of_solid
    L_hom[j, :] = sigma_til.copy()/scl

    fname_cuv = os.path.join(cdir,'Eps_{}.csv'.format(case))
    fname_mom = os.path.join(cdir,'Stil_{}.csv'.format(case))

    np.savetxt(fname_cuv,Eps_til)
    np.savetxt(fname_mom,sigma_til)

    Vt = FunctionSpace(msh,VectorElement('CG',msh.ufl_cell(),deg))
    u_plot = Function(Vt)
    u_plot.assign(project(u_full, Vt))
    # thta_plot.assign(project(thta_full, Vt))

    xdmf_fname = os.path.join(cdir,'data_vals_{}.xdmf'.format(case))

    with XDMFFile(comm,xdmf_fname) as res_fil:
        res_fil.parameters["flush_output"] = True
        res_fil.parameters["functions_share_mesh"] = True
        res_fil.write(u_full,0)
np.savetxt(fname_Ltil,L_hom,delimiter=',')
# Btil = np.loadtxt(fname_Btil,delimiter=',')
# Mhomogenized[:,:,ishp,i_alph,i_fo] = Btil.copy() 