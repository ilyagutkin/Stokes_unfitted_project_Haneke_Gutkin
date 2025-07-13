"""
convergence_study.py

This script performs a convergence study for the analytical Stokes problem:

    -Δu + ∇p = f   in Ω
     div(u) = 0   in Ω

with the known exact solution:
    u1 = -4*y * (1 - x**2 - y**2)
    u2 = 4*x*(1 - x**2 - y**2)
    u_exact = CoefficientFunction((u1, u2))
    p_exact = sin(x) * cos(y)

The right-hand side f is constructed accordingly:

    Δu1 = -32*y
    Δu2 = 32*x
    ∇p = (dpdx, dpdy), with:
        dpdx = cos(x) * cos(y)
        dpdy = -sin(x) * sin(y)

Thus,
    f = (-Δu1 - dpdx,
         -Δu2 - dpdy)

This study is performed in the unfitted CutFEM setting. The script supports different solver variants through the `Stokes_Solver` class, including:

- Taylor–Hood elements of arbitrary order
- P1–P1 elements
- P1–P0 elements
- H(div)-conforming methods

You can extend the `Stokes_Solver` class with additional solvers and use this script to numerically evaluate the convergence rates of your method.
"""


from netgen.geom2d import SplineGeometry
from ngsolve import *
from ngsolve.internal import *
import ngsolve
from xfem import *
from xfem.lsetcurv import *
from stokes_solver import *
from helper_functions.vizualization import *

def convergence_study(solver, levelset,f, ud, uexact, pexact,geo = None,maxh= [0.5, 0.25, 0.125, 0.0625], order = None):
    l2erroru = []
    l2errorp = []
    print("Convergence study is working ...")
    for maxh in maxh:
        if geo== None:
            square = SplineGeometry()
            square.AddRectangle((-1.25, -1.25), (1.25, 1.25), bc=1)
            ngmesh = square.GenerateMesh(maxh=maxh)
            mesh = Mesh(ngmesh)
        else:
            ngmesh = geo.GenerateMesh(maxh=maxh)
            mesh = Mesh(ngmesh)

        gfu1, l2erroru_1 , l2errorp_1= solver(mesh, levelset=levelset, f=f, ud=uexact, uexact=uexact , pexact=pexact,order=order)
        l2erroru.append((maxh, order, l2erroru_1))
        l2errorp.append((maxh, order, l2errorp_1))
    return l2erroru, l2errorp
        


if __name__ == "__main__":
    u1 = -4*y * (1 - x**2 - y**2)
    u2 = 4*x*(1 - x**2 - y**2)
    uexact = CoefficientFunction((u1, u2))
    pexact = sin(x)*cos(y)
    # Laplace-Anteil
    lapu1 = 32*y
    lapu2 = -32*x
    # Gradient von p
    dpdx = cos(x) * cos(y)
    dpdy = -sin(x) * sin(y)
    f = CoefficientFunction((
        -lapu1 - dpdx,
        -lapu2 - dpdy
    ))

    #levelsetfunction 
    levelset = x**2 + y**2 - 1

    l2erroru ,l2errorp = convergence_study(P1_P1,levelset,f,ud=uexact,uexact=uexact,pexact=pexact)
    plot_convergence(l2erroru,error_index=2, h_index=0, label="L2 error of velocity")

    l2erroru_taylor_hood = []
    l2errorp_taylor_hood = []


    ### EXAMPLE USAGE ###
    # This example shows how to use the stokes_Taylor_Hood function to solve the Stokes problem
    # with Taylor-Hood elements of different orders and evaluate the convergence rates.
    
    
    for order in [2,3,4]:
        l2erroru ,l2errorp = convergence_study(stokes_Taylor_Hood,levelset,f,ud=uexact,uexact=uexact,pexact=pexact,order=order)
        plot_convergence(l2erroru,error_index=2, h_index=0, label="L2 error of velocity")
        l2erroru_taylor_hood = l2erroru_taylor_hood + l2erroru
        l2errorp_taylor_hood = l2errorp_taylor_hood + l2errorp

    #Visualization of the errors in a table format
    print("L2 errors velocity:", triplet_table(l2erroru_taylor_hood, 0,1))
    print("L2 errors pressure:", triplet_table(l2errorp_taylor_hood, 0,1))
    print("finish")

