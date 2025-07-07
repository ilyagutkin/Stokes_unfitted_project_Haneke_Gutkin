from netgen.geom2d import SplineGeometry
from ngsolve import *
from ngsolve.internal import *
import ngsolve
from xfem import *
from xfem.lsetcurv import *
from stokes_biliniearforms import stokes_Taylor_Hood
from spielerei.vizualization import triplet_table

print("done")
l2erroru = []
l2errorp = []
r = sqrt(x**2 + y**2)
levelset = r-1

# Analytische testfunktion
uexact = CoefficientFunction((
            sin(pi*x) * cos(pi*y),   # u(x,y)
            -cos(pi*x) * sin(pi*y)   # v(x,y)
        ))
pexact = sin(pi*x) * sin(pi*y)
# Laplace von uexact (Komponentenweise)
lapu_x = -pi**2 * sin(pi*x) * cos(pi*y) * 2
lapu_y = -pi**2 * (-cos(pi*x) * sin(pi*y)) * 2  

# Gradient von p
dp_dx = pi * cos(pi*x) * sin(pi*y)
dp_dy = pi * sin(pi*x) * cos(pi*y)
# Gesamte rechte Seite f = -Δu + ∇p
f = CoefficientFunction((
            -lapu_x + dp_dx,
            -lapu_y + dp_dy
        ))
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

#pexact = 0
#dpdx = 0
#dpdy = 0
f = CoefficientFunction((
    -lapu1 - dpdx,
    -lapu2 - dpdy
))
for maxh in [0.5, 0.25,0.125]:
    square = SplineGeometry()
    square.AddRectangle((-1.25, -1.25), (1.25, 1.25), bc=1)
    ngmesh = square.GenerateMesh(maxh=maxh)
    mesh = Mesh(ngmesh)

    for order in [2, 3, 4]:
        lsetmeshadap = LevelSetMeshAdaptation(mesh, order=order, threshold=0.1,discontinuous_qn=True)# Higher order level set approximation 
        deformation = lsetmeshadap.CalcDeformation(levelset)
        #lsetp1 = lsetmeshadap.lset_p1

        lsetp1 = GridFunction(H1(mesh,order=1,autoupdate=True),autoupdate=True)
        InterpolateToP1(levelset,lsetp1)# Element, facet and dof marking w.r.t. boundary approximation with lsetp1:
        ci = CutInfo(mesh, lsetp1)
        hasneg = ci.GetElementsOfType(HASNEG)
        neg = ci.GetElementsOfType(NEG)
        hasif = ci.GetElementsOfType(IF)

        interface_facet_set = GetFacetsWithNeighborTypes(mesh, a=hasif, b=hasneg)
        h = specialcf.mesh_size
        n = Normalize(grad(lsetp1))

        # integration domains:
        dx = dCut(lsetp1, NEG, definedonelements=hasneg,deformation=deformation)
        ds = dCut(lsetp1, IF, definedonelements=hasif, deformation=deformation)
        dw_interface = dFacetPatch(definedonelements=interface_facet_set, deformation=deformation)
        V = VectorH1(mesh, order=order,dgjumps=True)
        V = Compress(V, GetDofsOfElements(V,hasneg))
        Q = H1(mesh, order=order-1)
        Q = Compress(Q, GetDofsOfElements(Q,hasneg))
        Z = NumberSpace(mesh)
        X = V*Q*Z

        gfu = GridFunction(X)

        A , F = stokes_Taylor_Hood(X, h , n, dw_interface, dx , ds, f=f ,ud=uexact, order= order, nu=1)

        gfu.vec.data = A.mat.Inverse(X.FreeDofs()) * F.vec

        error_u = sqrt(Integrate( (gfu.components[0] - uexact) ** 2*dx, mesh ))
        #        r = sqrt(x**2 + y**2)print("L2 error u:", error_u, "maxh:", maxh, "order:", order)
        error_p = sqrt(Integrate( (gfu.components[1] - pexact) ** 2*dx, mesh ))
        #print("L2 error p:", error_p)
        l2erroru.append((maxh, order, error_u))
        l2errorp.append((maxh, order, error_p))

print("L2 errors velocity:", triplet_table(l2erroru, 0,1))
print("L2 errors pressure:", triplet_table(l2errorp, 0,1))