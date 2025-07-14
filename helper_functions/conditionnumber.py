import numpy as np
from ngsolve import *
from scipy.sparse.linalg import svds
#from ordnung.konvergenzstudie import *
from ngsolve.la import EigenValues_Preconditioner



def compute_condition_number(a, use_sparse_svd=False):
    """
    Berechne die Konditionszahl einer NGSolve-Matrix.

    Parameter:
        matrix: ngsolve.BaseMatrix oder ngsolve.SparseMatrix
        use_sparse_svd: ob statt Vollmatrix eine sparse SVD verwendet werden soll

    Rückgabe:
        cond: geschätzte Konditionszahl (float)
    """
    lams=EigenValues_Preconditioner(a.mat, IdentityMatrix(a.space.ndof))
    lams_sorted = sorted(lams, key=lambda x: abs(x))
    cond = abs(lams_sorted[-1]) / abs(lams_sorted[0])

    return cond

if __name__ == "__main__":
    import sys
    sys.path.append('.')
    from ordnung.stokes_solver import P1_P1, P1_P0
    from ordnung.konvergenzstudie import *
    print("Start condition number study ...")
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

    square = SplineGeometry()
    square.AddRectangle((-1.25, -1.25), (1.25, 1.25), bc=1)
    ngmesh = square.GenerateMesh(maxh=0.2)
    mesh = Mesh(ngmesh)
    gfu1, l2erroru_1 , l2errorp_1, condi= P1_P1(mesh, levelset=levelset, f=f, ud=uexact, uexact=uexact , pexact=pexact,order=3 , conditionnumber=True)
    print(condi)
    print(convergence_study(P1_P0, levelset,f, uexact, uexact, pexact,geo = None,maxh= [0.5, 0.25, 0.125, 0.0625], order = None , condition_number= True))

