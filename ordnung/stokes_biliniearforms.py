from netgen.geom2d import SplineGeometry
from ngsolve import *
from ngsolve.internal import *
import ngsolve
from xfem import *
import numpy as np

# Some helper functions
def jump(f):
    return f - f.Other()
def jumpn(f,n):
    return Grad(f)*n - Grad(f).Other()*n


def stokes_Taylor_Hood(fes, h , n, dw_interface, dx , ds, f=CF((0,0)), ud=CF((0,0)), order= 2, nu=0.01):
    if order == 1:
        raise ValueError("Order 1 Taylor-Hood elements are not stable. Use order 2 or higher.")
    # Stabilization parameter for ghost-penalty
    gamma_stab = 100
    beta2 = 10 * order**2
    beta0 = 10 * order**2
    # Stabilization parameter for Nitsche
    lambda_nitsche = 10 * order * order
    (u,p,z),(v,q,z1) = fes.TnT()

    stokes = BilinearForm(fes)
    stokes += InnerProduct(Grad(u), Grad(v))*dx + div(u)*q*dx -q*n * u *ds \
        + div(v)*p*dx -p*n * v * ds
    stokes += -(Grad(u)*n * v + Grad(v)*n * u) * ds + gamma_stab/ h * u * v * ds
    stokes += beta2* InnerProduct(jumpn(u,n), jumpn(v,n)) * dw_interface #velocity ghost penalty
    stokes += -beta0 * InnerProduct(jump(p), jump(q)) * dw_interface #velocity ghost penalty
    stokes += p*z1 *dx + q *z*dx #lagrange multiplier to fix the pressure
    stokes.Assemble()

    rhs = LinearForm(fes)
    rhs += InnerProduct(f, v) * dx 
    #rhs += - Grad(v) * n * ud * ds
    #rhs += lambda_nitsche/h * ud * v * ds
    #rhs += q*n * ud * ds
    rhs.Assemble()

    return stokes,rhs