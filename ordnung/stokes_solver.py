from netgen.geom2d import SplineGeometry
from ngsolve import *
from ngsolve.internal import *
import ngsolve
from xfem import *
from xfem.lsetcurv import *
import numpy as np
from ngsolve import *
from helper_functions.conditionnumber import compute_condition_number


def stokes_Taylor_Hood(mesh,levelset , f=CF((0,0)), ud=CF((0,0)), nu=1, uexact =None, pexact=None, order=2, conditionnumber = False):
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
    rhs += -(Grad(v)*n * uexact) * ds + gamma_stab / h * uexact * v * ds+q*n*uexact *ds
    rhs.Assemble()
    gfu.vec.data = a.mat.Inverse(X.FreeDofs()) * rhs.vec

    if conditionnumber == True:
        condi = compute_condition_number(a)
    else:
        condi = None

    if uexact is not None:
        error_u = sqrt(Integrate( (gfu.components[0] - uexact) ** 2*dx, mesh ))
    if pexact is not None:
        error_p = sqrt(Integrate( (gfu.components[1] - pexact) ** 2*dx, mesh ))
    if condi == None:
        return gfu , error_u ,error_p
    else:
        return gfu , error_u, error_p, condi

def P1_P1(mesh, levelset, f=CF((0,0)), ud=CF((0,0)), nu=1, uexact=None, pexact=None, order = None, conditionnumber = False):
    gamma_stab = 100
    beta0 = 1
    beta1 = 1
    beta2 = 1

    lsetp1 = GridFunction(H1(mesh,order=1,autoupdate=True),autoupdate=True)
    InterpolateToP1(levelset,lsetp1)# Element, facet and dof marking w.r.t. boundary approximation with lsetp1:
    
    ci = CutInfo(mesh, lsetp1)
    hasneg = ci.GetElementsOfType(HASNEG)
    neg = ci.GetElementsOfType(NEG)
    hasif = ci.GetElementsOfType(IF)
    haspos = ci.GetElementsOfType(HASPOS)
    GP_facets = GetFacetsWithNeighborTypes(mesh, a=hasif, b=hasneg, use_and=True)

    h = specialcf.mesh_size
    n = Normalize(grad(lsetp1))
    nF = specialcf.normal(mesh.dim)

        # integration domains:
    d_stable_GP = dx(definedonelements = GP_facets, skeleton = True)
    dX = dCut(lsetp1, NEG, definedonelements=hasneg)
    ds = dCut(lsetp1, IF, definedonelements=hasif)

    #dw_interface = dFacetPatch(definedonelements=interface_facet_set)
    V = VectorH1(mesh, order=1,dgjumps=True)
    V = Compress(V, GetDofsOfElements(V,hasneg))
    Q = H1(mesh, order=1)
    Q = Compress(Q, GetDofsOfElements(Q,hasneg))
    Z = NumberSpace(mesh)
    X = V*Q*Z
    (u,p,z),(v,q,z1) = X.TnT()
    gfu = GridFunction(X)

    def jump(f):
        return f - f.Other()
    def jumpn(f):
        return Grad(f)*nF - Grad(f).Other()*nF

    a = BilinearForm(X)
    stokes = InnerProduct(Grad(u), Grad(v))*dX + div(u)*q*dx + div(v)*p*dX
    a += stokes
    a += -(Grad(u)*n * v + Grad(v)*n * u) * ds + gamma_stab / h * u * v * ds #nitzshe stabilization
    a += -(q*n * u + p*n * v) * ds
    a += p*z1 *dx + q *z*dX
    a += -beta1* h**2 * grad(p) * grad(q) * dX  #pressure stabilization
    a += beta2* h* InnerProduct(jumpn(u), jumpn(v)) * d_stable_GP#ds_inner_facets #velocity ghost penalty
    a += -beta0 * h**3*InnerProduct(jumpn(p), jumpn(q)) * d_stable_GP#ds_inner_facets #pressure ghost penalty
    a.Assemble()

    rhs = LinearForm(X)  
    rhs += InnerProduct(f, v)* dX
    rhs += -(Grad(v)*n * uexact) * ds + gamma_stab / h * uexact * v * ds+q*n*uexact *ds
    rhs += -beta1 * h**2 * InnerProduct(f , grad(q))* dX  # pressure stabilization term
    rhs.Assemble()
    gfu.vec.data = a.mat.Inverse(X.FreeDofs()) * rhs.vec

    if conditionnumber == True:
        condi = compute_condition_number(a)
    else:
        condi = None


    if uexact is not None:
        error_u = sqrt(Integrate( (gfu.components[0] - uexact) ** 2*dX, mesh ))
    if pexact is not None:
        error_p = sqrt(Integrate( (gfu.components[1] - pexact) ** 2*dX, mesh ))
    if condi == None:
        return gfu , error_u ,error_p
    else:
        return gfu , error_u, error_p, condi


def P1_P0(mesh, levelset, f=CF((0,0)), ud=CF((0,0)), nu=1, uexact=None, pexact=None, order=None, conditionnumber = False):
    gamma_stab = 100
    beta0 =1
    beta1 = 1
    beta2 = 1

    lsetp1 = GridFunction(H1(mesh,order=1,autoupdate=True),autoupdate=True)
    InterpolateToP1(levelset,lsetp1)# Element, facet and dof marking w.r.t. boundary approximation with lsetp1:
    
    ci = CutInfo(mesh, lsetp1)
    hasneg = ci.GetElementsOfType(HASNEG)
    neg = ci.GetElementsOfType(NEG)
    hasif = ci.GetElementsOfType(IF)
    haspos = ci.GetElementsOfType(HASPOS)


    interior_facets = GetFacetsWithNeighborTypes(mesh, a=neg, b=neg)
    interface_facet_set = GetFacetsWithNeighborTypes(mesh, a=hasif, b=hasneg)

        
    h = specialcf.mesh_size
    n = Normalize(grad(lsetp1))
    nF = specialcf.normal(mesh.dim)

        # integration domains:
    dX = dCut(lsetp1, NEG, definedonelements=hasneg)
    ds = dCut(lsetp1, IF, definedonelements=hasif)
    ds_inner_facets = dx(definedonelements = interface_facet_set, skeleton = True)
    ds_stable_facets = dCut(lsetp1, NEG, definedonelements=interior_facets, skeleton = True)

    #dw_interface = dFacetPatch(definedonelements=interface_facet_set)
    V = VectorH1(mesh, order=1,dgjumps=True)
    V = Compress(V, GetDofsOfElements(V,hasneg))
    Q = L2(mesh, order=0, dgjumps=True)
    Q = Compress(Q, GetDofsOfElements(Q,hasneg))
    Z = NumberSpace(mesh)
    X = V*Q*Z
    (u,p,z),(v,q,z1) = X.TnT()
    gfu = GridFunction(X)

    def jump(f):
        return f - f.Other()
    def jumpn(f):
        return Grad(f)*nF - Grad(f).Other()*nF

    a = BilinearForm(X)
    stokes = InnerProduct(Grad(u), Grad(v))*dX + div(u)*q*dX + div(v)*p*dX
    a += stokes
    a += -(Grad(u)*n * v + Grad(v)*n * u) * ds + gamma_stab / h * u * v * ds #nitzshe stabilization
    a += -(q*n * u + p*n * v) * ds
    a += p*z1 *dX + q *z*dX
    a += -beta0 * h* jump(p)  * jump(q) * ds_stable_facets #pressure stabilization
    a += beta2* h* InnerProduct(jumpn(u), jumpn(v)) * ds_inner_facets #velocity ghost penalty
    a += -beta0 * h*InnerProduct(jump(p), jump(q)) * ds_inner_facets #pressure ghost penalty
    a.Assemble()

    rhs = LinearForm(X)  
    rhs += InnerProduct(f, v)* dX
    rhs += -(Grad(v)*n * uexact) * ds + gamma_stab / h * uexact * v * ds+q*n*uexact *ds
    rhs.Assemble()
    gfu.vec.data = a.mat.Inverse(X.FreeDofs()) * rhs.vec

    if conditionnumber == True:
        condi = compute_condition_number(a)
    else:
        condi = None


    if uexact is not None:
        error_u = sqrt(Integrate( (gfu.components[0] - uexact) ** 2*dX, mesh ))
    if pexact is not None:
        error_p = sqrt(Integrate( (gfu.components[1] - pexact) ** 2*dX, mesh ))
    if condi == None:
        return gfu , error_u ,error_p
    else:
        return gfu , error_u, error_p, condi


def Divergence_free(mesh, levelset, f=CF((0,0)), ud=CF((0,0)), nu=1, uexact=None, pexact=None, order=3, conditionnumber = False):
    gamma_GP = 0.2*order**2
    gamma_Nitsche = 20
    gamma_IP = 20

    gamma_stab = 20
    gamma_GP = 0.2*order**2
    alpha = 20

    gamma_stab_N_fac = lambda k: 0.01 #* (k+1)**2#0.25 * (k+1)**2
    gamma_stab_N_vol = lambda k: 0.01 #* (k+1)**2

    lsetmeshadap = LevelSetMeshAdaptation(mesh, order=order, threshold=0.1,discontinuous_qn=True)# Higher order level set approximation 
    deformation = lsetmeshadap.CalcDeformation(levelset)

    lsetp1 = GridFunction(H1(mesh,order=1,autoupdate=True),autoupdate=True)
    InterpolateToP1(levelset,lsetp1)# Element, facet and dof marking w.r.t. boundary approximation with lsetp1:

    ci = CutInfo(mesh, lsetp1)
    hasneg = ci.GetElementsOfType(HASNEG)
    hasif = ci.GetElementsOfType(IF)
    neg = ci.GetElementsOfType(NEG)
    cut_facets = GetFacetsWithNeighborTypes(mesh, a=hasif, b=hasif)
    stab_facets = GetFacetsWithNeighborTypes(mesh, a=hasif, b=hasneg)
    hasneg_facets = GetFacetsWithNeighborTypes(mesh, a=hasneg, b=hasneg, use_and=True)

    h = specialcf.mesh_size
    n = Normalize(grad(lsetp1))
    nF = specialcf.normal(mesh.dim)

        #Integration domains
    dX = dCut(lsetp1, NEG, definedonelements=hasneg,deformation=deformation)
    ds = dCut(lsetp1, IF, definedonelements=hasif,deformation=deformation)
    dw = dFacetPatch(definedonelements=stab_facets,deformation=deformation)
    dz = dFacetPatch(definedonelements=cut_facets, deformation=deformation)
    dxbar = dx(definedonelements=hasneg, deformation=deformation)
    dxcut = dx(definedonelements=hasif, deformation=deformation)
    dxinner = dx(definedonelements=neg, deformation=deformation)
    dcutskel = dCut(lsetp1, NEG, skeleton=True, definedonelements=hasneg_facets, deformation=deformation)

    #dw_interface = dFacetPatch(definedonelements=interface_facet_set)
    Shsbase = HDiv(mesh, order=order, dirichlet=[], dgjumps=True)
    Vhbase = L2(mesh, order=order-1, dirichlet=[], dgjumps=True)
    Vh = Restrict(Vhbase, hasneg)
    Shs = Restrict(Shsbase, hasneg)
    Fhbase = H1(mesh, order=order, dirichlet=[], dgjumps=True)
    Fh = Restrict(Fhbase, hasif) 
    Nh = NumberSpace(mesh)
    X = Shs*Vh*Fh*Nh
    
    (u,p,lam,r),(v,q,mu,s) = X.TnT()
    gfu = GridFunction(X)
    
    def jump(f):
        return f - f.Other()
    def avgdnF(f):
        return 0.5 * (Grad(f)*nF + Grad(f.Other())*nF)
    def jumpn(f): 
        return InnerProduct(f, nF) - InnerProduct(f.Other(), nF)

    #Bilinearform
    a = BilinearForm(X, symmetric=False)
    a += nu* InnerProduct(Grad(u), Grad(v))*dX + nu*(-Grad(u)*n * v -Grad(v)*n * u) * ds + nu * gamma_Nitsche / h * u * v * ds # a terms
    a += div(u) * q * dxbar + div(v) * p * dxbar #b terms
    a += nu*(-avgdnF(u) * jump(v) + -avgdnF(v) * jump(u) + gamma_IP / h * jump(u) * jump(v)) * dcutskel # interior penalty terms
    a += gamma_GP/h**2 * nu * InnerProduct(jump(u), jump(v)) * dw #velocity ghost penalty

    # Lagrange multiplier term to fix the normal velocity at the boundary
    a += (-u*n * mu - v*n * lam) * ds 
    a += -gamma_stab_N_vol(order)*h*(grad(lam)*n) * (grad(mu)*n)*dxcut #to many dofs -> constant extension in normaldirection 
    a += -gamma_stab_N_fac(order)*1/h*jump(lam)*jump(mu)* dz #GP-for Lagrange multiplier 
    a += (p*s + q*r)*dX # unique pressure term

    # regularisation (to help the linear solver)
    a += -1e-8 *r*s * dX - 1e-8 * p*q*dxbar

    rhs = LinearForm(X) 
    rhs += InnerProduct(f, v)* dX
    rhs += -uexact * n * mu * ds
    rhs += uexact * nu * ( gamma_Nitsche / h * v - Grad(v)*n) * ds
    rhs += pexact * s * dX

    a.Assemble()
    rhs.Assemble()

    gfu.vec.data = a.mat.Inverse(X.FreeDofs(), inverse="sparsecholesky") * rhs.vec

    if conditionnumber == True:
        condi = compute_condition_number(a)
    else:
        condi = None

    
    if uexact is not None:
        error_u = sqrt(Integrate( (gfu.components[0] - uexact) ** 2*dX, mesh ))
    if pexact is not None:
        error_p = sqrt(Integrate( (gfu.components[1] - pexact) ** 2*dxinner, mesh ))
    if condi == None:
        return gfu , error_u ,error_p
    else:
        return gfu , error_u, error_p, condi