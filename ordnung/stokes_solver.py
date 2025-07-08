from netgen.geom2d import SplineGeometry
from ngsolve import *
from ngsolve.internal import *
import ngsolve
from xfem import *
from xfem.lsetcurv import *
import numpy as np
from ngsolve import *


def stokes_Taylor_Hood(mesh,levelset , order, f=CF((0,0)), ud=CF((0,0)), nu=1, uexact =None):
    if order == 1:
        raise ValueError("Order 1 Taylor-Hood elements are not stable. Use order 2 or higher.")
    gamma_stab = 100
    beta0 = 10 * order**2
    beta2 = 10 * order**2

    lsetmeshadap = LevelSetMeshAdaptation(mesh, order=order, threshold=0.1,discontinuous_qn=True)# Higher order level set approximation 
    deformation = lsetmeshadap.CalcDeformation(levelset)

    lsetp1 = GridFunction(H1(mesh,order=1,autoupdate=True),autoupdate=True)
    InterpolateToP1(levelset,lsetp1)# Element, facet and dof marking w.r.t. boundary approximation with lsetp1:
    
    ci = CutInfo(mesh, lsetp1)
    hasneg = ci.GetElementsOfType(HASNEG)
    neg = ci.GetElementsOfType(NEG)
    hasif = ci.GetElementsOfType(IF)
    haspos = ci.GetElementsOfType(HASPOS)
    ba_facets = GetFacetsWithNeighborTypes(mesh, a=haspos, b=any) 


    interior_facets = GetFacetsWithNeighborTypes(mesh, a=neg, b=neg)
    interface_facet_set = GetFacetsWithNeighborTypes(mesh, a=hasif, b=hasneg)
        

        
    h = specialcf.mesh_size
    n = Normalize(grad(lsetp1))

        # integration domains:
    dx = dCut(lsetp1, NEG, definedonelements=hasneg,deformation=deformation)
    ds = dCut(lsetp1, IF, definedonelements=hasif,deformation=deformation)

    dw_interface = dFacetPatch(definedonelements=interface_facet_set, deformation=deformation)
    V = VectorH1(mesh, order=order,dgjumps=True)
    V = Compress(V, GetDofsOfElements(V,hasneg))
    Q = H1(mesh, order=order-1)
    Q = Compress(Q, GetDofsOfElements(Q,hasneg))
    Z = NumberSpace(mesh)
    X = V*Q*Z
    (u,p,z),(v,q,z1) = X.TnT()
    gfu = GridFunction(X)

    def jump(f):
        return f - f.Other()
    def jumpn(f):
        n = Normalize(grad(lsetp1))
        return Grad(f)*n - Grad(f).Other()*n

    a = BilinearForm(X)
    stokes = InnerProduct(Grad(u), Grad(v))*dx + div(u)*q*dx + div(v)*p*dx
    a += stokes
    a += -(Grad(u)*n * v + Grad(v)*n * u) * ds + gamma_stab / h * u * v * ds #nitzshe stabilization
    a += -(q*n * u + p*n * v) * ds
    a += p*z1 *dx + q *z*dx
    a += beta2* InnerProduct(jump(Grad(u)), jump(Grad(v))) * dw_interface #velocity ghost penalty
    a += -beta0 * InnerProduct(jump(p), jump(q)) * dw_interface #pressure ghost penalty
    a.Assemble()

    rhs = LinearForm(X)  # oder f*v*dx mit f gegeben
    rhs += InnerProduct(f, v)* dx
    #rhs += -(Grad(v)*n * uexact) * ds + gamma_stab / h * uexact * v * ds+q*n*uexact *ds
    rhs.Assemble()
    gfu.vec.data = a.mat.Inverse(X.FreeDofs()) * rhs.vec

    if uexact is not None:
        error_u = sqrt(Integrate( (gfu.components[0] - uexact) ** 2*dx, mesh ))
    return gfu , error_u