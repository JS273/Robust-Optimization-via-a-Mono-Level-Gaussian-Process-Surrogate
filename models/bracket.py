"""Parametric finite-element analysis of a 2D structural bracket.

The module generates a parameterized bracket geometry and mesh with Gmsh,
solves a plane-stress linear elasticity problem using DOLFINx, computes
structural compliance and von Mises stress, and provides utilities for
exporting and visualizing the results.

Units
-----
Geometry : mm
Force    : N
Stress   : N/mm^2
Density  : kg/mm^3
Time     : s

Author
------
Mustapha Essou, Julius Schultz

Date
----
2026-10-16
"""

from typing import List
import gmsh
import math
from pathlib import Path
from dolfinx import mesh, fem, plot, io, default_scalar_type
from dolfinx.fem import dirichletbc, Function, locate_dofs_topological, FunctionSpace
from dolfinx.io.gmsh import model_to_mesh
from dolfinx.io.utils import XDMFFile
from mpi4py import MPI
import ufl
from dolfinx.fem.petsc import LinearProblem
import numpy as np
import sys, os
from scipy.optimize import fsolve
import pyvista
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

#######################
# Material parameters #
#######################

E = 2.1e5  # Young's modulus [N/mm^2]
nu = 0.3   # Poisson's ratio
model = "plane_stress"
mu = E/2/(1+nu)
lmbda = E*nu/(1+nu)/(1-2*nu)
if model == "plane_stress":
    lmbda = 2*mu*lmbda/(lmbda+2*mu)
g = 9.81e3              # mm/s^2
rho = 7.85e-9           # kg / mm^3

############################
# Define strain and stress #
############################

def epsilon(u):
    return ufl.sym(ufl.grad(u))  # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)
def sigma(u):
    return lmbda * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)
def sigma_dev(u):
    return  sigma(u) - (1 / 3) * ufl.tr(sigma(u)) * ufl.Identity(len(u))
def sigma_vm(u):
    return ufl.sqrt((3 / 2) * ufl.inner(sigma_dev(u), sigma_dev(u)))

#############################
# Objectives and constaints #
#############################

def structural_complience(u): return ufl.inner(sigma(u), epsilon(u)) * ufl.dx

#Define constraints
def vol_const(x,p, circle_radius, max_V): 
    return volume_bracket(x, p, circle_radius) - max_V

def stress_const(sig_v, sig_v0):
    return np.max(sig_v) - sig_v0

def volume_bracket(x, p, circle_radius=10.):
    if x.shape[1] !=5:
        raise ValueError("fist entry must have length 5")
    if len(p)!=8:
        raise ValueError("second entry must have length 8")

    A_outer = (p[2]-p[0])*(p[3]+p[1])/2 + p[0]*p[1]
    # the inner has 2 extra fillets that need to be added to the outer mass
    A_outer += p[4]**2*(1.-np.pi/4)*2
    #np.float64(212371.1846010243)
    
    s824 = p[2]-(x[:,1]+x[:,3])
    s713 = p[1]-(x[:,0]+x[:,2])

    m = (p[0]-p[2])/(p[1]-p[3])
    b = p[2]-m*p[3]
    alpha = np.arctan(m)
    dh = x[:,4]/np.cos(alpha)
    s45x = ((p[2]-x[:,3]) + dh -b)/m
    s51y = m*(p[1] - x[:,0]) + b - dh

    s251 = s51y - x[:,1]
    s455 = s45x - x[:,2]

    A_innter = (s824-s251)*(s455+s713)/2 + s251*s713
    # np.float64(87445.51545889513)
    A_circle = circle_radius**2 * np.pi
    #np.float64(1570.7963267948967)

    return (A_outer - A_innter - A_circle) * p[7]

##########
# Helper #
##########

def assemble(form: ufl.Form|fem.Form)->default_scalar_type:
    compiled_form = fem.form(form)
    local_form = fem.assemble_scalar(compiled_form)
    return compiled_form.mesh.comm.allreduce(local_form, op=MPI.SUM)

def points_for_paralel_lines(model:gmsh.model, coord_p1:np.ndarray, coord_p2:np.ndarray, coord_p3, coord_p4, distance:float, tag_point1=-1, tag_point2=-1):
    #direction vector of outer edge
    r = coord_p2-coord_p1
    #perpendicular direction of r
    rx = np.array([[0,-1,0], 
                   [1,0,0],
                   [0,0,1]])@r
    norm_r = np.linalg.norm(r)

    #new line equation
    x_spm = (coord_p2+coord_p1)/2 + rx/norm_r * distance
    x_ =lambda lmbda:  x_spm + r*lmbda
    
    #find intersection with upper line:
    lmbda_up = (coord_p3[1] - x_spm[1])/r[1]
    # func = lambda l: x_(l[0])- (coord_p3+np.array([1, 0 , 0])*l[1])
    # l = fsolve(func, [0,0,0])
    # print(f"{lmbda_up:.5f}, {l[0]:.5f}")
    #add point
    model.occ.addPoint(*x_(lmbda_up), 5., tag_point1)

    #find intersection with left line:
    lmbda_left = (coord_p4[0] - x_spm[0])/r[0]
    # func = lambda l: x_(l[0]) - (coord_p4+np.array([0, 1, 0])*l[1])
    # l = fsolve(func, [0,0,0])
    # print(f"{lmbda_left:.5f}, {l[0]:.5f}")
    #add point
    model.occ.addPoint(*x_(lmbda_left), 5, tag_point2)
    gmsh.model.occ.synchronize()

def create_mesh(x:List[float], p:List[float], circle_radius=10., h=1., offset:np.ndarray = np.array([0,0,0]), mesh_save_path = None, logger = None):
    if logger is not None: logger.debug(f"     Gmsh: Start geometry and mesh generation")
    if len(x)!=5:
        raise ValueError("fist entry must have length 5")
    if len(p)!=8:
        raise ValueError("second entry must have length 7")
    
    s1,s2,s3,s4,s5 = x
    p1,p2,p3,p4,p5,p6,p7,p8 = p

    # Initialize Gmsh
    gmsh.initialize()
    gmsh.model()
    gmsh.model.add("Bracket")
    gmsh.option.setNumber("General.Terminal", 0)

    # mesh size
    ms = 0

    #gmsh.option.setNumber('Mesh.MeshSizeMax', 1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)
    bracket = gmsh.model.occ

    #define coordinates of the points:
    co_p1 = np.array((0, 0, 0))     + offset
    co_p2 = np.array((p2, 0, 0))    + offset
    co_p3 = np.array((p2, p3, 0))   + offset
    co_p4 = np.array((p2-p4, p3, 0))+ offset
    co_p5 = np.array((0, p1, 0))    + offset

    co_p7 = np.array((s1, s2, 0))   + offset
    co_p8 = np.array((p2-s3, s2, 0))+ offset
    co_p9 = np.array((p2-s3, p3-s4, 0))+ offset

    # outer rectangle
    point1 = bracket.addPoint(*co_p1, ms)
    point2 = bracket.addPoint(*co_p2, ms)
    point3 = bracket.addPoint(*co_p3, ms)
    point4 = bracket.addPoint(*co_p4, ms)
    point5 = bracket.addPoint(*co_p5, ms)

    # inner triangle
    point7 = bracket.addPoint(*co_p7, ms)
    point8 = bracket.addPoint(*co_p8, ms)
    point9 = bracket.addPoint(*co_p9, ms)

    points_for_paralel_lines(gmsh.model(), co_p4, co_p5, co_p9, co_p7, s5, 11,12)

    # Create Lines
    ## outer lines
    bracket.addLine(point1, point2, 1)
    bracket.addLine(point2, point3, 2)
    bracket.addLine(point3, point4, 3)
    bracket.addLine(point4, point5, 4)
    bracket.addLine(point5, point1, 5)

    ## inner lines
    bracket.addLine(point7, point8, 6)
    bracket.addLine(point8, point9, 7)
    bracket.addLine(point9, 11, 8)
    bracket.addLine(11, 12, 9)
    bracket.addLine(12, point7, 10)

    ## add outer fillets
    bracket.fillet2D(3, 4, p5-offset[0],102)
    bracket.fillet2D(4, 5, p5-offset[0],103)
    bracket.fillet2D(5, 1, p5-offset[0],104)

    ## add inner fillets
    bracket.fillet2D(6, 7, p5+offset[0],105)
    bracket.fillet2D(7, 8, p5+offset[0],106)
    bracket.fillet2D(8, 9, p5+offset[0],107)
    bracket.fillet2D(9, 10, p5+offset[0],108)
    bracket.fillet2D(10, 6, p5+offset[0],109)

    outer_lines = [1,2,3,102,4,103,5,104]
    inner_lines = [6,105,7,106,8,107,9,108,10,109]

    # Create a surface with a hole
    ## combine all outer loops
    outer_loop = bracket.addCurveLoop(outer_lines)
    ## create circle and arrange it as loop
    circle_center = np.array([p6, p7, 0]) + offset
    circle = bracket.addCircle(*circle_center, circle_radius)
    circle_loop = bracket.addCurveLoop([circle])
    ## combine all inner loops
    inner_loop = bracket.addCurveLoop(inner_lines)
    bracket.synchronize()

    # Create a surface
    surface = bracket.addPlaneSurface(([outer_loop, inner_loop, circle_loop]))
    bracket.synchronize()

    # Apply Boundary Conditions (Tag boundaries)
    gmsh.model.addPhysicalGroup(1, [110], tag=1)  # Dirichlet boundary
    gmsh.model.addPhysicalGroup(1, np.hstack([[1,111,112,113,114,115,116], [6,118,119,120,121,122,123,124,125, 126]]), tag=2)  # Neumann boundary
    gmsh.model.addPhysicalGroup(1, [117], tag=3) # circular boundary
    gmsh.model.addPhysicalGroup(2, [surface], tag=4)  # Entire domain
    bracket.synchronize()

    # Mesh generation
    # gmsh.model.mesh.setRecombine(2, surface)
    bracket.synchronize()
    gmsh.model.mesh.generate(2)

    if mesh_save_path is not None: #Export the mesh
        Path(mesh_save_path).parent.mkdir(parents=True, exist_ok=True)
        gmsh.write(mesh_save_path)

    gdim = 2  # domain geometry dimension
    msh_data = model_to_mesh(gmsh.model(), MPI.COMM_WORLD, rank=0, gdim=gdim)
    gmsh.finalize()
    if logger is not None: logger.debug(f"     Gmsh: Finished")
    
    return msh_data

def fem_simulation(msh_data, t, f_mag, f_angle, circle_radius, logger = None):

    if logger is not None: logger.debug(f"     Fenicsx: Started")
    domain = msh_data.mesh
    V = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim, )))
    gdim = 2
    fdim = domain.topology.dim - 1
    boundary_facets = msh_data.facet_tags.find(1)
    db_nodes = locate_dofs_topological(V, domain.topology.dim - 1, boundary_facets)

    # Apply zero displacement
    u_D = np.array([0,]*gdim, dtype=default_scalar_type)
    bc = fem.dirichletbc(u_D, db_nodes, V)

    #Apply traction force T=F/A
    force_vector = np.array([f_mag*np.cos(f_angle), f_mag*np.sin(f_angle)], dtype=default_scalar_type)/(t*2*circle_radius*np.pi)
    T = fem.Constant(domain, force_vector)

    ds = ufl.Measure("ds", domain=domain, subdomain_data=msh_data.facet_tags)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    f = fem.Constant(domain, default_scalar_type((0, -rho * g)))
    a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
    L = ufl.dot(f, v) * ufl.dx + ufl.dot(T, v) * ds(3)

    # assamble problem and solve
    if logger is not None: logger.debug(f"     Fenicsx: Solving")
    problem = LinearProblem(
        a, L,
        bcs=[bc],
        petsc_options_prefix="my_problem",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
    )
    uh = problem.solve()

    if logger is not None: logger.debug(f"     Fenicsx: Finished")

    return uh, domain

def compute_stress(domain, uh):
    V0 = fem.functionspace(domain, ("DG", 0, (4,)))
    sig_exp = fem.Expression(sigma(uh), V0.element.interpolation_points)
    sig = fem.Function(V0, name="Stress")
    sig.interpolate(sig_exp)

    V0 = fem.functionspace(domain, ("DG", 0, (1,)))
    sig_vm = fem.Expression(sigma_vm(uh), V0.element.interpolation_points)
    sig_v = fem.Function(V0, name="von Mises Stress")
    sig_v.interpolate(sig_vm)

    return sig,sig_v

def export_paraview(output_path, domain, uh, sig, sig_v):
        # Save to file for ParaView if requested
    if output_path is not None:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with io.XDMFFile(domain.comm, output_path, "w") as xdmf:
            xdmf.write_mesh(domain)
            xdmf.write_function(uh)
            xdmf.write_function(sig)
            xdmf.write_function(sig_v)

############
# Callable #
############

def sim_bracket_qoi(x:List[float], p:List[float], xi:List[float], circle_radius=10., h=1., offset:np.ndarray = np.array([0,0,0]), mesh_save_path=None, logger = None):

    msh_data = create_mesh(x=x, p=p, circle_radius=circle_radius, h=h, offset=offset, mesh_save_path = mesh_save_path, logger = logger)
    uh, _ = fem_simulation(msh_data=msh_data, t=p[-1], f_mag=xi[0], f_angle=xi[1], circle_radius=circle_radius, logger=logger)

    if logger is not None: logger.debug(f"     Fenicsx: Compute compliance")
    complience = assemble(structural_complience(uh))
    if logger is not None: logger.debug(f"     Fenicsx: Finished \n")
    return complience

#####################
# Plotting routines #
#####################

def plot_field_solution(uh, scale = 1):

    V = uh.function_space
    gdim = uh.function_space.mesh.geometry.dim
    # Create plotter and pyvista grid
    p = pyvista.Plotter()
    topology, cell_types, geometry = plot.vtk_mesh(V)
    u3D = np.hstack([uh.x.array.reshape(-1,gdim),np.zeros([geometry.shape[0],1])])
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry+u3D*scale)

    # Attach vector values to grid and warp grid by vector
    grid["u"] = u3D
    actor_0 = p.add_mesh(grid, scalars="u", show_edges=True)
    p.view_xy()
    p.show_grid()

    if not pyvista.OFF_SCREEN:
        p.show()

    return p

def plot_von_mises_matplotlib(field, domain, fig_ax=None, vmin=None, vmax=None):
    topology, cell_types, geometry = plot.vtk_mesh(domain)

    x = geometry[:, 0]
    y = geometry[:, 1]

    cells = topology.reshape(-1, 4)[:, 1:]
    triang = mtri.Triangulation(x, y, cells)

    if fig_ax is None:
        fig, ax = plt.subplots(figsize=[8, 8])
    else:
        fig, ax = fig_ax

    tpc = ax.tripcolor(
        triang,
        facecolors=field,
        edgecolors='face',
        cmap='coolwarm',
        vmin=vmin,
        vmax=vmax
    )

    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    data = {"triang": triang,
            "field": field,
            "vmin": vmin,
            "vmax": vmax}

    return fig, ax, tpc, data