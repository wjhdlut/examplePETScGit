#include<petscdmda.h>
#include<petscdm.h>
#include<petscksp.h>
#include"applicationExamples.h"

applicationExamples::applicationExamples()
{
    m_aEP  = this;
}

applicationExamples::applicationExamples(int rank, int size)
{
    m_rank = rank;
    m_size = size;
    m_aEP  = this;
}

applicationExamples::~applicationExamples()
{

}

/* ------------------ KSP example ex_13 ------------------ */
PetscErrorCode applicationExamples::SolPoissonProblemKSP()
{
    PetscReal      enorm;

    /* The next two lines are for testing only; these allow the user to
       decide the grid size at runtime. */
    m_m = 6;
    m_n = 7;
    ierr = PetscOptionsGetInt(NULL, NULL, "-m", &m_m, NULL); CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL, NULL, "-n", &m_n, NULL); CHKERRQ(ierr);
    m_N    = m_m * m_n;

    /* Create the empty sparse matrix and linear solver data structures */
    ierr = InitializeLinearSolver(&m_userCtx); CHKERRQ(ierr);

    /* Allocate arrays to hold the solution to the linear system.
       This is not normally done in PETSc programs, but in this case,
       since we are calling these routines from a non-PETSc program, we
       would like to reuse the data structures from another code. So in
       the context of a larger application these would be provided by
       other (non-PETSc) parts of the application code. */
    PetscScalar *userx    = new PetscScalar[m_N];
    PetscScalar *userb    = new PetscScalar[m_N];
    PetscScalar *solution = new PetscScalar[m_N];

    /* Allocate an array to hold the coefficients in the elliptic operator */
    PetscScalar *rho = new PetscScalar[m_N];

    /* Fill up the array rho[] with the function rho(x,y) = x; fill the
       right-hand-side b[] and the solution with a known problem for testing. */
    PetscScalar hx = 1.0/(m_m+1);
    PetscScalar hy = 1.0/(m_n+1);
    PetscScalar y  = hy;
    PetscInt    Ii = 0;

    for (PetscInt j=0; j<m_n; j++) {
        PetscScalar x = hx;
        for (PetscInt i=0; i<m_m; i++) {
            rho[Ii]      = x;
            solution[Ii] = PetscSinScalar(2. * PETSC_PI * x) * PetscSinScalar(2. * PETSC_PI * y);
            userb[Ii]    = -2. * PETSC_PI * PetscCosScalar(2. *PETSC_PI * x) * PetscSinScalar(+2. * PETSC_PI * y)
                    + 8. * PETSC_PI * PETSC_PI * x * PetscSinScalar(2. * PETSC_PI * x) * PetscSinScalar(2. * PETSC_PI * y);
            x += hx;
            Ii++;
        }
        y += hy;
    }

    /* Loop over a bunch of timesteps, setting up and solver the linear
       system for each time-step.
       Note this is somewhat artificial. It is intended to demonstrate how
       one may reuse the linear solver stuff in each time-step. */
    PetscInt      tmax = 2;
    PetscInt      J;
    PetscScalar   v;

    for (PetscInt t=0; t<tmax; t++) {
        Ii = 0;
        for (PetscInt j=0; j<m_n; j++) {
            for (PetscInt i=0; i<m_m; i++) {
                if (j>0) {
                    J    = Ii - m_m;
                    v    = -0.5 * (rho[Ii] + rho[J]) * m_userCtx.hy2;
                    ierr = MatSetValue(m_userCtx.A, Ii, J, v, INSERT_VALUES); CHKERRQ(ierr);
                }
                if (j<m_n-1) {
                    J    = Ii + m_m;
                    v    = -0.5 * (rho[Ii] + rho[J]) * m_userCtx.hy2;
                    ierr = MatSetValue(m_userCtx.A, Ii, J, v, INSERT_VALUES); CHKERRQ(ierr);
                }
                if (i>0) {
                    J    = Ii - 1;
                    v    = -0.5 * (rho[Ii] + rho[J]) * m_userCtx.hx2;
                    ierr = MatSetValue(m_userCtx.A, Ii, J, v, INSERT_VALUES); CHKERRQ(ierr);
                }
                if (i<m_m-1) {
                    J    = Ii + 1;
                    v    = -0.5 * (rho[Ii] + rho[J]) * m_userCtx.hx2;
                    ierr = MatSetValue(m_userCtx.A, Ii, J, v, INSERT_VALUES); CHKERRQ(ierr);
                }
                v    = 2.0 * rho[Ii] * (m_userCtx.hx2 + m_userCtx.hy2);
                ierr = MatSetValue(m_userCtx.A, Ii, Ii, v, INSERT_VALUES); CHKERRQ(ierr);
                Ii++;
            }
        }

        /* Assemble matrix */
        ierr = MatAssemblyBegin(m_userCtx.A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
        ierr = MatAssemblyEnd(m_userCtx.A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

        /* Set operators. Here the matrix that defines the linear system
           also serves as the preconditioning matrix. Since all the matrices
           will have the same nonzero pattern here, we indicate this so the
           linear solvers can take advantage of this. */
        ierr = KSPSetOperators(m_userCtx.ksp, m_userCtx.A, m_userCtx.A); CHKERRQ(ierr);

        /* Set linear solver defaults for this problem (optional).
           - Here we set it to use direct LU factorization for the solution */
        ierr = KSPGetPC(m_userCtx.ksp, &m_userCtx.pc); CHKERRQ(ierr);
        ierr = PCSetType(m_userCtx.pc, PCLU); CHKERRQ(ierr);

        /* Set runtime options, e.g.,
              -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
           These options will override those specified above as long as
           KSPSetFromOptions() is called _after_ any other customization
           routines.
           Run the program with the option -help to see all the possible
           linear solver options. */
        ierr = KSPSetFromOptions(m_userCtx.ksp); CHKERRQ(ierr);

        /* This allows the PETSc linear solvers to compute the solution
           directly in the user's array rather than in the PETSc vector.
           This is essentially a hack and not highly recommend unless you
           are quite comfortable with using PETSc. In general, users should
           write their entire application using PETSc vectors rather than
           arrays. */
        ierr = VecPlaceArray(m_userCtx.x, userx); CHKERRQ(ierr);
        ierr = VecPlaceArray(m_userCtx.b, userb); CHKERRQ(ierr);

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                            Solve the linear system
           - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        ierr = KSPSolve(m_userCtx.ksp, m_userCtx.b, m_userCtx.x); CHKERRQ(ierr);

        /* Put back the PETSc array that belongs in the vector xuserctx->x */
        ierr = VecResetArray(m_userCtx.x); CHKERRQ(ierr);
        ierr = VecResetArray(m_userCtx.b); CHKERRQ(ierr);

        /* Compute error: Note that this could (and usually should) all be done
          using the PETSc vector operations. Here we demonstrate using more
          standard programming practices to show how they may be mixed with
          PETSc. */
        enorm = 0.0;
        for (PetscInt i=0; i<m_N; i++){
            enorm += PetscConj(solution[i] - userx[i]) * (solution[i] - userx[i]);
        }
        enorm *= PetscRealPart(hx*hy);
        ierr   = PetscPrintf(PETSC_COMM_WORLD, "m %D n %D error norm %g\n",
                             m_m, m_n, (double)enorm); CHKERRQ(ierr);
    }

    /* We are all finished solving linear systems, so we clean up the
       data structures. */
    ierr = FinalizeLinearSolver(&m_userCtx); CHKERRQ(ierr);
    delete[] userx;
    delete[] userb;
    delete[] solution;
    delete[] rho;
    return ierr;
}

PetscErrorCode applicationExamples::InitializeLinearSolver(UserCtx *userCtx)
{
    userCtx->hx2 = (m_m + 1) * (m_m + 1);
    userCtx->hy2 = (m_n + 1) * (m_n + 1);

    /* Create the sparse matrix. Preallocate 5 nonzeros per row. */
    ierr = MatCreateSeqAIJ(PETSC_COMM_SELF, m_N, m_N, 5, 0, &userCtx->A); CHKERRQ(ierr);

    /* Create vectors. Here we create vectors with no memory allocated.
       This way, we can use the data structures already in the program
       by using VecPlaceArray() subroutine at a later stage. */
    ierr = VecCreateSeqWithArray(PETSC_COMM_SELF, 1, m_N, NULL, &userCtx->b); CHKERRQ(ierr);
    ierr = VecDuplicate(userCtx->b, &userCtx->x); CHKERRQ(ierr);

    /* Create linear solver context. This will be used repeatedly for all
       the linear solves needed. */
    ierr = KSPCreate(PETSC_COMM_SELF, &userCtx->ksp); CHKERRQ(ierr);

    return ierr;
}

PetscErrorCode applicationExamples::FinalizeLinearSolver(UserCtx *userCtx)
{
    ierr = MatDestroy(&userCtx->A); CHKERRQ(ierr);
    ierr = VecDestroy(&userCtx->b); CHKERRQ(ierr);
    ierr = VecDestroy(&userCtx->x); CHKERRQ(ierr);
    ierr = KSPDestroy(&userCtx->ksp); CHKERRQ(ierr);

    return ierr;
}

/* ------------------ KSP example ex_25 ------------------ */
/* Solves 1D variable coefficient Laplacian using multigrid. */

/*
d  |(1 + e*sine(2*pi*k*x)) d u | = 1, 0 < x < 1,
-- |                       --- |
dx |                       dx  |
*/
PetscErrorCode applicationExamples::SolPartialDiffEqu()
{
    PetscErrorCode ierr;
    KSP            ksp;
    DM             da;
    AppCtx         user;
    Mat            A;
    Vec            b, b2;
    Vec            x;
    PetscReal      nrm;

    user.k = 1;
    ierr   = PetscOptionsGetInt(NULL, 0, "-k", &user.k, 0); CHKERRQ(ierr);
    user.e = .99;
    ierr   = PetscOptionsGetScalar(NULL, 0, "-e", &user.e, 0); CHKERRQ(ierr);

    ierr = KSPCreate(PETSC_COMM_WORLD, &ksp); CHKERRQ(ierr);
    ierr = DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, 128, 1, 1, 0, &da); CHKERRQ(ierr);
    ierr = DMSetFromOptions(da); CHKERRQ(ierr);
    ierr = DMSetUp(da); CHKERRQ(ierr);
    ierr = KSPSetDM(ksp,da); CHKERRQ(ierr);

    ierr = KSPSetComputeRHS(ksp, CompRHS, &user); CHKERRQ(ierr);
    ierr = KSPSetComputeOperators(ksp, CompStiffMatrix, &user); CHKERRQ(ierr);

    ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);
    ierr = KSPSolve(ksp, NULL, NULL); CHKERRQ(ierr);

    ierr = KSPGetOperators(ksp, &A, NULL); CHKERRQ(ierr);
    ierr = KSPGetSolution(ksp, &x); CHKERRQ(ierr);
    ierr = KSPGetRhs(ksp, &b); CHKERRQ(ierr);
    ierr = VecDuplicate(b, &b2); CHKERRQ(ierr);
    ierr = MatMult(A, x, b2); CHKERRQ(ierr);
    ierr = VecAXPY(b2, -1.0, b); CHKERRQ(ierr);
    ierr = VecNorm(b2, NORM_MAX, &nrm); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Residual norm %g\n", (double)nrm); CHKERRQ(ierr);

    ierr = VecDestroy(&b2); CHKERRQ(ierr);
    ierr = KSPDestroy(&ksp); CHKERRQ(ierr);
    ierr = DMDestroy(&da); CHKERRQ(ierr);

    return ierr;
}

applicationExamples* applicationExamples::m_aEP = NULL;

PetscErrorCode applicationExamples::CompRHS(KSP ksp, Vec b, void *ctx)
{
    PetscInt       mx;
    PetscScalar    h;
    DM             da;
    PetscErrorCode ierr;
    PetscInt       *idx = new PetscInt[2];
    PetscScalar    *v   = new PetscScalar[2];

    PetscFunctionBeginUser;
    ierr   = KSPGetDM(ksp, &da); CHKERRQ(ierr);
    ierr   = DMDAGetInfo(da, 0, &mx, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0); CHKERRQ(ierr);
    std::cout << "RANK[" << m_aEP->m_rank << "], " << "mx = " << mx << std::endl;

    h      = 1.0/((mx-1));
    ierr   = VecSet(b, h); CHKERRQ(ierr);
    idx[0] = 0;
    idx[1] = mx -1;
    v[0]   = 0.0;
    v[1]   = 0.0;
    ierr   = VecSetValues(b, 2, idx, v, INSERT_VALUES); CHKERRQ(ierr);
    ierr   = VecAssemblyBegin(b); CHKERRQ(ierr);
    ierr   = VecAssemblyEnd(b); CHKERRQ(ierr);
    PetscFunctionReturn(0);

    delete[] idx;
    delete[] v;
}

PetscErrorCode applicationExamples::CompStiffMatrix(KSP ksp, Mat J, Mat jac, void *ctx)
{
    AppCtx         *user = (AppCtx*)ctx;
    PetscInt       mx,xm,xs;
    PetscScalar    v[3],h,xlow,xhigh;
    MatStencil     row,col[3];
    DM             da;
    PetscErrorCode ierr;

    PetscFunctionBeginUser;
    ierr = KSPGetDM(ksp, &da); CHKERRQ(ierr);
    ierr = DMDAGetInfo(da, 0, &mx, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0); CHKERRQ(ierr);
    ierr = DMDAGetCorners(da, &xs, 0, 0, &xm, 0, 0); CHKERRQ(ierr);
    std::cout << "RANK[" << m_aEP->m_rank << "], " << "xs = " << xs << ", xm = " << xm << std::endl;

    h    = 1.0/(mx-1);
    std::cout << "RANK[" << m_aEP->m_rank << "], h = " << h << std::endl;

    for (auto i=xs; i<xs+xm; i++) {
        row.i = i;
        if (i==0 || i==mx-1) {
            v[0] = 2.0/h;
            ierr = MatSetValuesStencil(jac, 1, &row, 1, &row, v, INSERT_VALUES); CHKERRQ(ierr);
        } else {
            xlow  = h*(PetscReal)i - .5*h;
            xhigh = xlow + h;
            v[0]  = (-1.0 - user->e*PetscSinScalar(2.0*PETSC_PI*user->k*xlow))/h;
            col[0].i = i-1;
            v[1]  = (2.0 + user->e*PetscSinScalar(2.0*PETSC_PI*user->k*xlow)
                     + user->e*PetscSinScalar(2.0*PETSC_PI*user->k*xhigh))/h;
            col[1].i = row.i;
            v[2]  = (-1.0 - user->e*PetscSinScalar(2.0*PETSC_PI*user->k*xhigh))/h;
            col[2].i = i+1;
            ierr  = MatSetValuesStencil(jac, 1, &row, 3, col, v, INSERT_VALUES); CHKERRQ(ierr);
        }
    }
    ierr = MatAssemblyBegin(jac, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(jac, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    PetscInt gn;
    ierr = MatGetSize(jac, NULL, &gn); CHKERRQ(ierr);
    std::cout << "RANK[" << m_aEP->m_rank << "], gn = " << gn << std::endl;
    PetscFunctionReturn(0);
    return ierr;
}

/* -------- the oerloading version of KSP example ex_25 -------- */
PetscErrorCode applicationExamples::SolPartialDiffEqu(int temp)
{
    AppCtx user;
    user.k = 1;
    ierr   = PetscOptionsGetInt(NULL, 0, "-k", &user.k, 0); CHKERRQ(ierr);
    user.e = .99;
    ierr   = PetscOptionsGetScalar(NULL, 0, "-e", &user.e, 0); CHKERRQ(ierr);

    m_N = 128;
    ierr = PetscOptionsGetInt(NULL, NULL, "-N", &m_N, NULL); CHKERRQ(ierr);

    Mat A;
    Vec x, b;
    ierr = MatCreate(PETSC_COMM_WORLD, &A); CHKERRQ(ierr);
    ierr = MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, m_N, m_N); CHKERRQ(ierr);
    ierr = MatSetFromOptions(A); CHKERRQ(ierr);
    ierr = MatSetUp(A); CHKERRQ(ierr);

    ierr = MatCreateVecs(A, &x, NULL); CHKERRQ(ierr);
    ierr = VecDuplicate(x, &b); CHKERRQ(ierr);

    /* Set the right-hand-side vector */
    PetscScalar h = 1.0/(m_N - 1.);
    RANK_COUT << "h = " << h << std::endl;
    ierr = VecSet(b, h); CHKERRQ(ierr);

    PetscInt *idx = new PetscInt[3]();
    PetscScalar *v = new PetscScalar[3]();
    idx[0] = 0;
    idx[1] = m_N -1;
    v[0]   = 0.0;
    v[1]   = 0.0;
    ierr = VecSetValues(b, 2, idx, v, INSERT_VALUES); CHKERRQ(ierr);
    ierr = VecAssemblyBegin(b); CHKERRQ(ierr);
    ierr = VecAssemblyEnd(b); CHKERRQ(ierr);

    /* Compute the stiffness matrix */
    PetscInt Istart, Iend;
    ierr = MatGetOwnershipRange(A, &Istart, &Iend); CHKERRQ(ierr);
    RANK_COUT << "Istart = " << Istart << ", Iend = " << Iend << std::endl;

    for(auto i = Istart; i < Iend; i++)
    {
        if (i==0 || i==m_N-1) {
            v[0] = 2.0/h;
            ierr = MatSetValue(A, i, i, v[0], INSERT_VALUES); CHKERRQ(ierr);
        }
        else {
            PetscScalar xlow  = h*(PetscReal)i - .5*h;
            PetscScalar xhigh = xlow + h;
            v[0]   = (-1.0 - user.e*PetscSinScalar(2.0*PETSC_PI*user.k*xlow))/h;
            idx[0] = i - 1;
            v[1]   = (2.0 + user.e*PetscSinScalar(2.0*PETSC_PI*user.k*xlow)
                     + user.e*PetscSinScalar(2.0*PETSC_PI*user.k*xhigh))/h;
            idx[1] = i;
            v[2]  = (-1.0 - user.e*PetscSinScalar(2.0*PETSC_PI*user.k*xhigh))/h;
            idx[2] = i + 1;
            ierr  = MatSetValues(A, 1, &i, 3, idx, v, INSERT_VALUES); CHKERRQ(ierr);
        }
    }
    ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

    /* create the ksp context */
    KSP ksp;
    ierr = KSPCreate(PETSC_COMM_WORLD, &ksp); CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp, A, A); CHKERRQ(ierr);
    ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);
    ierr = KSPSolve(ksp, b, x); CHKERRQ(ierr);

    /* check error */
    PetscInt its;
    Vec b2;
    PetscScalar norm;
    ierr = VecDuplicate(b, &b2); CHKERRQ(ierr);
    ierr = MatMult(A, x, b2); CHKERRQ(ierr);
    ierr = VecAXPY(b2, -1.0, b); CHKERRQ(ierr);
    ierr = VecNorm(b2, NORM_MAX, &norm); CHKERRQ(ierr);
    ierr = KSPGetIterationNumber(ksp, &its); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Norm of error %g iterations %D\n",
                       (double)norm, its); CHKERRQ(ierr);

    ierr = MatDestroy(&A); CHKERRQ(ierr);
    ierr = VecDestroy(&x); CHKERRQ(ierr);
    ierr = VecDestroy(&b); CHKERRQ(ierr);
    ierr = VecDestroy(&b2); CHKERRQ(ierr);
    ierr = KSPDestroy(&ksp); CHKERRQ(ierr);
    delete[] idx;
    delete[] v;

    return ierr;
}

/* ------------------ KSP example ex_28 ------------------ */
/* Solves 1D wave equation using multigrid. */
PetscErrorCode applicationExamples::Sol1DWaveEqu()
{
    PetscInt       i;
    KSP            ksp;
    DM             da;
    Vec            x;

    ierr = KSPCreate(PETSC_COMM_WORLD, &ksp); CHKERRQ(ierr);
    ierr = DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC, 3, 2, 1, 0, &da); CHKERRQ(ierr);
    ierr = DMSetFromOptions(da); CHKERRQ(ierr);
    ierr = DMSetUp(da); CHKERRQ(ierr);
    ierr = KSPSetDM(ksp, da); CHKERRQ(ierr);

    ierr = KSPSetComputeRHS(ksp, CompRHSEx28, NULL); CHKERRQ(ierr);
    ierr = KSPSetComputeOperators(ksp, CompStiffMatrixEx28, NULL); CHKERRQ(ierr);

    ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);
    ierr = DMCreateGlobalVector(da, &x); CHKERRQ(ierr);

    ierr = CompInitialSolution(da, x); CHKERRQ(ierr);

    ierr = DMSetApplicationContext(da, x); CHKERRQ(ierr);
    ierr = KSPSetUp(ksp); CHKERRQ(ierr);
    ierr = VecView(x, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
    for (i=0; i<10; i++) {
        ierr = KSPSolve(ksp, NULL, x); CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD, "Print x vector\n"); CHKERRQ(ierr);
        ierr = VecView(x, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
    }
    ierr = VecDestroy(&x); CHKERRQ(ierr);
    ierr = KSPDestroy(&ksp); CHKERRQ(ierr);
    ierr = DMDestroy(&da); CHKERRQ(ierr);

    return ierr;
}

PetscErrorCode applicationExamples::CompRHSEx28(KSP ksp,Vec b,void *ctx)
{
    PetscErrorCode ierr;
    PetscInt       mx;
    PetscScalar    h;
    Vec            x;
    DM             da;

    PetscFunctionBeginUser;
    ierr = KSPGetDM(ksp, &da); CHKERRQ(ierr);
    ierr = DMDAGetInfo(da, 0, &mx, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0); CHKERRQ(ierr);
    ierr = DMGetApplicationContext(da, &x);CHKERRQ(ierr);
    h    = 2.0*PETSC_PI/((mx));
    ierr = VecCopy(x, b); CHKERRQ(ierr);
    ierr = VecScale(b, h); CHKERRQ(ierr);

    ierr = VecView(b, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
    PetscFunctionReturn(ierr);
}

PetscErrorCode applicationExamples::CompStiffMatrixEx28(KSP ksp, Mat J, Mat jac, void *ctx)
{
    PetscErrorCode ierr;
    PetscInt       i, mx, xm, xs;
    PetscScalar    v[7], Hx;
    MatStencil     row;
    PetscScalar    lambda;
    DM             da;
    MatStencil     *col = new MatStencil[7]();

    PetscFunctionBeginUser;
    ierr   = KSPGetDM(ksp, &da); CHKERRQ(ierr);
    ierr   = DMDAGetInfo(da, 0, &mx, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0); CHKERRQ(ierr);
    Hx     = 2.0*PETSC_PI / (PetscReal)(mx);
    ierr   = DMDAGetCorners(da, &xs, 0, 0, &xm, 0, 0); CHKERRQ(ierr);

    std::cout << "RANK[" << m_aEP->m_rank << "], " << "xs = " << xs << ", xm = " << xm << std::endl;

    lambda = 2.0*Hx;
    for (i=xs; i<xs+xm; i++) {
      row.i = i;
      row.j = 0;
      row.k = 0;
      row.c = 0;

      v[0]  = Hx;
      col[0].i = i;
      col[0].c = 0;

      v[1]  = lambda;
      col[1].i = i-1;
      col[1].c = 1;

      v[2]  = -lambda;
      col[2].i = i+1;
      col[2].c = 1;
      ierr  = MatSetValuesStencil(jac, 1, &row, 3, col, v, INSERT_VALUES); CHKERRQ(ierr);

      row.i = i;
      row.j = 0;
      row.k = 0;
      row.c = 1;

      v[0]  = lambda;
      col[0].i = i-1;
      col[0].c = 0;

      v[1]  = Hx;
      col[1].i = i;
      col[1].c = 1;

      v[2]  = -lambda;
      col[2].i = i+1;
      col[2].c = 0;
      ierr  = MatSetValuesStencil(jac, 1, &row, 3, col, v, INSERT_VALUES); CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(jac, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(jac, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

    ierr = MatView(jac, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
    delete[] col;
    PetscFunctionReturn(ierr);
}

PetscErrorCode applicationExamples::CompInitialSolution(DM da, Vec x)
{
    PetscInt       mx, col[2], xs, xm, i;
    PetscScalar    Hx, val[2];

    PetscFunctionBeginUser;
    ierr = DMDAGetInfo(da, 0, &mx, 0, 0, 0, 0, 0, 0, 0,0,0,0,0); CHKERRQ(ierr);

    ierr = PetscPrintf(PETSC_COMM_WORLD, "CompInitialSolution!\n"); CHKERRQ(ierr);
    std::cout << "RANK[" << m_aEP->m_rank << "], " << "mx = " << mx << std::endl;

    Hx   = 2.0*PETSC_PI / (PetscReal)(mx);
    ierr = DMDAGetCorners(da, &xs, 0, 0, &xm, 0, 0); CHKERRQ(ierr);

    for (i=xs; i<xs+xm; i++) {
      col[0] = 2*i;
      col[1] = 2*i + 1;
      val[0] = val[1] = PetscSinScalar(((PetscScalar)i)*Hx);
      ierr   = VecSetValues(x, 2, col, val, INSERT_VALUES); CHKERRQ(ierr);
    }
    ierr = VecAssemblyBegin(x); CHKERRQ(ierr);
    ierr = VecAssemblyEnd(x); CHKERRQ(ierr);

    PetscInt xsize;
    ierr = VecGetSize(x, &xsize); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "the global dimension of vec x %d\n", xsize); CHKERRQ(ierr);

    PetscFunctionReturn(ierr);
}


/* ------------------ KSP example ex_29 ------------------ */
PetscErrorCode applicationExamples::SolInhomoLapl2D()
{
    KSP            ksp;
    DM             da;
    UserContext    user;
    PetscInt       bc;
    Vec            b, x;
    PetscBool      testsolver = PETSC_FALSE;
    const char     *bcTypes[2] = {"dirichlet","neumann"};

    ierr = KSPCreate(PETSC_COMM_WORLD, &ksp); CHKERRQ(ierr);
    ierr = DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, 3, 3,
                        PETSC_DECIDE, PETSC_DECIDE, 1, 1, 0, 0, &da); CHKERRQ(ierr);
    ierr = DMSetFromOptions(da); CHKERRQ(ierr);
    ierr = DMSetUp(da); CHKERRQ(ierr);

    ierr = DMDASetUniformCoordinates(da, 0, 1, 0, 1, 0, 0); CHKERRQ(ierr);

    /* User can specify the basic parameter at runtime */
    ierr = DMDASetFieldName(da, 0, "Pressure"); CHKERRQ(ierr);

    ierr        = PetscOptionsBegin(PETSC_COMM_WORLD, "", "Options for the inhomogeneous Poisson equation",
                                    "DMqq"); CHKERRQ(ierr);
    user.rho    = 1.0;
    ierr        = PetscOptionsReal("-rho", "The conductivity", "ex29.c", user.rho,
                                   &user.rho, NULL); CHKERRQ(ierr);
    ierr        = PetscPrintf(PETSC_COMM_WORLD, " rho = %g\n", user.rho); CHKERRQ(ierr);

    user.nu     = 0.1;
    ierr        = PetscOptionsReal("-nu", "The width of the Gaussian source", "ex29.c", user.nu,
                                   &user.nu, NULL); CHKERRQ(ierr);
    ierr        = PetscPrintf(PETSC_COMM_WORLD, " nu = %g \n", user.nu); CHKERRQ(ierr);

    bc          = (PetscInt)DIRICHLET;
    ierr        = PetscOptionsEList("-bc_type","Type of boundary condition","ex29.c",
                                    bcTypes, 2, bcTypes[0], &bc, NULL); CHKERRQ(ierr);

    user.bcType = (BCType)bc;
    ierr        = PetscOptionsBool("-testsolver", "Run solver multiple times, useful for performance studies of solver",
                                   "ex29.c", testsolver, &testsolver, NULL); CHKERRQ(ierr);
    ierr        = PetscOptionsEnd(); CHKERRQ(ierr);

    /* Compute the right-hand-side vector and stiffness matrix */
    ierr = KSPSetComputeRHS(ksp, CompRHSEx29, &user); CHKERRQ(ierr);
    ierr = KSPSetComputeOperators(ksp, CompStiffMatrixEx29, &user); CHKERRQ(ierr);

    /* set up the ksp context */
    ierr = KSPSetDM(ksp, da); CHKERRQ(ierr);
    ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);
    ierr = KSPSetUp(ksp); CHKERRQ(ierr);

    /* solve the equation */
    ierr = DMCreateGlobalVector(da, &x); CHKERRQ(ierr);
    ierr = KSPSolve(ksp, NULL, x); CHKERRQ(ierr);

    if (testsolver) {
        ierr = KSPGetSolution(ksp, &x); CHKERRQ(ierr);
        ierr = KSPGetRhs(ksp, &b); CHKERRQ(ierr);
        KSPSetDMActive(ksp, PETSC_FALSE);
        ierr = KSPSolve(ksp, b, x); CHKERRQ(ierr);
        {
            PetscInt n = 20;

            for (auto i=0; i<n; i++) {
                ierr = KSPSolve(ksp, b, x); CHKERRQ(ierr);
            }
        }
    }

    ierr = VecView(x, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

    ierr = DMDestroy(&da); CHKERRQ(ierr);
    ierr = KSPDestroy(&ksp); CHKERRQ(ierr);

    return ierr;
}

PetscErrorCode applicationExamples::CompRHSEx29(KSP ksp, Vec b, void *ctx)
{
    UserContext    *user = (UserContext*)ctx;
    PetscErrorCode ierr;
    PetscInt       i, j, mx, my, xm, ym, xs, ys;
    PetscScalar    Hx, Hy;
    PetscScalar    **array;
    DM             da;

    PetscFunctionBeginUser;
    ierr = KSPGetDM(ksp, &da); CHKERRQ(ierr);
    ierr = DMDAGetInfo(da, 0, &mx, &my, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0); CHKERRQ(ierr);
    Hx   = 1.0 / (PetscReal)(mx-1);
    Hy   = 1.0 / (PetscReal)(my-1);
    ierr = DMDAGetCorners(da, &xs, &ys, 0, &xm, &ym, 0); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, b, &array); CHKERRQ(ierr);
    for (j=ys; j<ys+ym; j++) {
        for (i=xs; i<xs+xm; i++) {
            array[j][i] = PetscExpScalar(-((PetscReal)i*Hx)*((PetscReal)i*Hx)/user->nu)
                    *PetscExpScalar(-((PetscReal)j*Hy)*((PetscReal)j*Hy)/user->nu)*Hx*Hy;
        }
    }
    ierr = DMDAVecRestoreArray(da, b, &array); CHKERRQ(ierr);
    ierr = VecAssemblyBegin(b); CHKERRQ(ierr);
    ierr = VecAssemblyEnd(b); CHKERRQ(ierr);

    /* force right hand side to be consistent for singular matrix */
    /* note this is really a hack, normally the model would provide you with a consistent right handside */
    if (user->bcType == NEUMANN) {
        MatNullSpace nullspace;

        ierr = MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, 0, &nullspace); CHKERRQ(ierr);
        ierr = MatNullSpaceRemove(nullspace, b); CHKERRQ(ierr);
        ierr = MatNullSpaceDestroy(&nullspace); CHKERRQ(ierr);
    }
    PetscFunctionReturn(ierr);
}

PetscErrorCode applicationExamples::CompStiffMatrixEx29(KSP ksp, Mat J, Mat jac, void *ctx)
{
    UserContext    *user = (UserContext*)ctx;
    PetscReal      centerRho;
    PetscErrorCode ierr;
    PetscInt       mx, my, xm, ym, xs, ys;
    PetscScalar    v[5];
    PetscReal      Hx, Hy, HydHx, HxdHy, rho;
    MatStencil     row, col[5];
    DM             da;
    PetscBool      check_matis = PETSC_FALSE;

    PetscFunctionBeginUser;
    ierr      = KSPGetDM(ksp, &da); CHKERRQ(ierr);
    centerRho = user->rho;
    ierr      = DMDAGetInfo(da, 0, &mx, &my, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0); CHKERRQ(ierr);
    Hx        = 1.0 / (PetscReal)(mx-1);
    Hy        = 1.0 / (PetscReal)(my-1);
    HxdHy     = Hx/Hy;
    HydHx     = Hy/Hx;
    ierr      = DMDAGetCorners(da, &xs, &ys, 0, &xm, &ym, 0); CHKERRQ(ierr);
    for (auto j=ys; j<ys+ym; j++) {
        for (auto i=xs; i<xs+xm; i++) {
            row.i = i;
            row.j = j;
            /* Compute the Rho */
            if ((i > mx/3.0) && (i < 2.0*mx/3.0) && (j > my/3.0) && (j < 2.0*my/3.0)) {
                rho = centerRho;
            }
            else {
                rho = 1.0;
            }

            if (i==0 || j==0 || i==mx-1 || j==my-1) {
                if (user->bcType == DIRICHLET) {
                    v[0] = 2.0*rho*(HxdHy + HydHx);
                    ierr = MatSetValuesStencil(jac, 1, &row, 1, &row, v, INSERT_VALUES); CHKERRQ(ierr);
                }
                else if (user->bcType == NEUMANN) {
                    PetscInt numx = 0, numy = 0, num = 0;
                    if (j!=0) {
                        v[num] = -rho*HxdHy;
                        col[num].i = i;
                        col[num].j = j-1;
                        numy++;
                        num++;
                    }

                    if (i!=0) {
                        v[num] = -rho*HydHx;
                        col[num].i = i-1;
                        col[num].j = j;
                        numx++;
                        num++;
                    }

                    if (i!=mx-1) {
                        v[num] = -rho*HydHx;
                        col[num].i = i+1;
                        col[num].j = j;
                        numx++;
                        num++;
                    }

                    if (j!=my-1) {
                        v[num] = -rho*HxdHy;
                        col[num].i = i;
                        col[num].j = j+1;
                        numy++;
                        num++;
                    }

                    v[num] = numx*rho*HydHx + numy*rho*HxdHy;
                    col[num].i = i;
                    col[num].j = j;
                    num++;
                    ierr = MatSetValuesStencil(jac, 1, &row, num, col, v, INSERT_VALUES); CHKERRQ(ierr);
                }
            }
            else {
                v[0] = -rho*HxdHy;
                col[0].i = i;
                col[0].j = j-1;

                v[1] = -rho*HydHx;
                col[1].i = i-1;
                col[1].j = j;

                v[2] = 2.0*rho*(HxdHy + HydHx);
                col[2].i = i;
                col[2].j = j;

                v[3] = -rho*HydHx;
                col[3].i = i+1;
                col[3].j = j;

                v[4] = -rho*HxdHy;
                col[4].i = i;
                col[4].j = j+1;
                ierr = MatSetValuesStencil(jac, 1, &row, 5, col, v, INSERT_VALUES); CHKERRQ(ierr);
            }
        }
    }
    ierr = MatAssemblyBegin(jac, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(jac, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

    ierr = MatViewFromOptions(jac, NULL, "-view_mat"); CHKERRQ(ierr);

    /* print the Row and Line of matrix */
    PetscInt numRow, numLine;
    ierr = MatGetSize(jac, &numRow, &numLine); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "the size of stiffness matrix: numRow = %d, numLine = %d \n",
                       numRow, numLine); CHKERRQ(ierr);

    /* print the Row and Line of matrix in each process */
    PetscInt lNumRow, lNumLine;
    ierr = MatGetLocalSize(jac, &lNumRow, &lNumLine); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF, "RANK[%d], local size of stiffness matrix: numRow = %d, numLine = %d\n",
                       m_aEP->m_rank, lNumRow, lNumLine); CHKERRQ(ierr);

    PetscInt Istart, Iend;
    ierr = MatGetOwnershipRange(jac, &Istart, &Iend); CHKERRQ(ierr);
    std::cout << "RANK[" << m_aEP->m_rank << "], Istart = " << Istart << ", Iend = " << Iend << std::endl;

    ierr = MatView(jac, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
    ierr = PetscOptionsGetBool(NULL, NULL, "-check_matis", &check_matis, NULL); CHKERRQ(ierr);
    if (check_matis) {
        void      (*f)(void);
        Mat       J2;
        MatType   jtype;
        PetscReal nrm;

        ierr = MatGetType(jac, &jtype); CHKERRQ(ierr);
        ierr = MatConvert(jac, MATIS, MAT_INITIAL_MATRIX, &J2); CHKERRQ(ierr);
        ierr = MatViewFromOptions(J2, NULL, "-view_conv"); CHKERRQ(ierr);
        ierr = MatConvert(J2, jtype, MAT_INPLACE_MATRIX, &J2); CHKERRQ(ierr);
        ierr = MatGetOperation(jac, MATOP_VIEW, &f); CHKERRQ(ierr);
        ierr = MatSetOperation(J2, MATOP_VIEW, f); CHKERRQ(ierr);
        ierr = MatSetDM(J2, da); CHKERRQ(ierr);
        ierr = MatViewFromOptions(J2, NULL, "-view_conv_assembled"); CHKERRQ(ierr);
        ierr = MatAXPY(J2, -1., jac, DIFFERENT_NONZERO_PATTERN); CHKERRQ(ierr);
        ierr = MatNorm(J2, NORM_FROBENIUS, &nrm); CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD, "Error MATIS %g\n", (double)nrm); CHKERRQ(ierr);
        ierr = MatViewFromOptions(J2, NULL, "-view_conv_err"); CHKERRQ(ierr);
        ierr = MatDestroy(&J2); CHKERRQ(ierr);
    }
    if (user->bcType == NEUMANN) {
        MatNullSpace nullspace;

        ierr = MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, 0, &nullspace); CHKERRQ(ierr);
        ierr = MatSetNullSpace(J, nullspace); CHKERRQ(ierr);
        ierr = MatNullSpaceDestroy(&nullspace); CHKERRQ(ierr);
    }
    PetscFunctionReturn(ierr);
}

/* ------------------ KSP example ex_32 ------------------ */
/*
Laplacian in 2D. Modeled by the partial differential equation

   div  grad u = f,  0 < x,y < 1,

with forcing function

   f = e^{-(1 - x)^2/\nu} e^{-(1 - y)^2/\nu}

with pure Neumann boundary conditions */
PetscErrorCode applicationExamples::SolInhomoLapl2DEx32()
{
    KSP            ksp;
    DM             da;
    UserContext    user;
    PetscInt       bc;
    const char     *bcTypes[2] = {"dirichlet","neumann"};

    ierr = KSPCreate(PETSC_COMM_WORLD, &ksp); CHKERRQ(ierr);
    ierr = DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR,
                        12, 12, PETSC_DECIDE, PETSC_DECIDE, 1, 1, 0, 0, &da); CHKERRQ(ierr);
    ierr = DMSetFromOptions(da); CHKERRQ(ierr);
    ierr = DMSetUp(da); CHKERRQ(ierr);
    ierr = DMDASetInterpolationType(da, DMDA_Q0); CHKERRQ(ierr);

    ierr = KSPSetDM(ksp, da); CHKERRQ(ierr);

    ierr        = PetscOptionsBegin(PETSC_COMM_WORLD, "", "Options for the inhomogeneous Poisson equation",
                                    "DM"); CHKERRQ(ierr);

    user.nu     = 0.1;
    ierr        = PetscOptionsScalar("-nu", "The width of the Gaussian source", "ex29.c",
                                     user.nu, &user.nu, NULL); CHKERRQ(ierr);
    bc          = (PetscInt)NEUMANN;
    ierr        = PetscOptionsEList("-bc_type", "Type of boundary condition", "ex29.c",
                                    bcTypes, 2, bcTypes[0], &bc, NULL); CHKERRQ(ierr);

    user.bcType = (BCType)bc;
    ierr        = PetscOptionsEnd(); CHKERRQ(ierr);

    /* Compute the right-hand-side vector and stiffness matrix */
    ierr = KSPSetComputeRHS(ksp, CompRHSEx32, &user);CHKERRQ(ierr);
    ierr = KSPSetComputeOperators(ksp, CompStiffMatrixEx32, &user); CHKERRQ(ierr);

    /* Solve the equation using ksp */
    ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);
    ierr = KSPSolve(ksp, NULL, NULL); CHKERRQ(ierr);


    ierr = KSPDestroy(&ksp); CHKERRQ(ierr);
    ierr = DMDestroy(&da); CHKERRQ(ierr);

    return ierr;
}

PetscErrorCode applicationExamples::CompRHSEx32(KSP ksp, Vec b, void *ctx)
{
    UserContext    *user = (UserContext*)ctx;
    PetscErrorCode ierr;
    DM             da;

    PetscFunctionBeginUser;
    ierr = KSPGetDM(ksp, &da); CHKERRQ(ierr);

    PetscInt mx, my;
    ierr = DMDAGetInfo(da, 0, &mx, &my, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0); CHKERRQ(ierr);

    PetscInt xs, xm;
    PetscInt ys, ym;
    PetscScalar **array;
    ierr = DMDAGetCorners(da, &xs, &ys, 0, &xm, &ym, 0); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, b, &array); CHKERRQ(ierr);

    PetscScalar Hx   = 1.0 / (PetscReal)(mx);
    PetscScalar Hy   = 1.0 / (PetscReal)(my);
    for (PetscInt j=ys; j<ys+ym; j++) {
        for (PetscInt i=xs; i<xs+xm; i++) {
            array[j][i] = PetscExpScalar(-(((PetscReal)i+0.5)*Hx)*(((PetscReal)i+0.5)*Hx)/user->nu)
                    *PetscExpScalar(-(((PetscReal)j+0.5)*Hy)*(((PetscReal)j+0.5)*Hy)/user->nu)*Hx*Hy;
        }
    }

    ierr = DMDAVecRestoreArray(da, b, &array); CHKERRQ(ierr);
    ierr = VecAssemblyBegin(b); CHKERRQ(ierr);
    ierr = VecAssemblyEnd(b); CHKERRQ(ierr);

    /* force right hand side to be consistent for singular matrix */
    /* note this is really a hack, normally the model would provide you with a consistent right handside */
    if (user->bcType == NEUMANN) {
        MatNullSpace nullspace;

        ierr = MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, 0, &nullspace); CHKERRQ(ierr);
        ierr = MatNullSpaceRemove(nullspace, b); CHKERRQ(ierr);
        ierr = MatNullSpaceDestroy(&nullspace); CHKERRQ(ierr);
    }
    PetscFunctionReturn(ierr);
}

PetscErrorCode applicationExamples::CompStiffMatrixEx32(KSP ksp, Mat J, Mat jac, void *ctx)
{
    UserContext    *user = (UserContext*)ctx;
    PetscErrorCode ierr;
    PetscInt       i,j,mx,my,xm,ym,xs,ys,num, numi, numj;
    PetscScalar    v[5],Hx,Hy,HydHx,HxdHy;
    MatStencil     row, col[5];
    DM             da;

    PetscFunctionBeginUser;
    ierr  = KSPGetDM(ksp,&da);CHKERRQ(ierr);
    ierr  = DMDAGetInfo(da,0,&mx,&my,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
    Hx    = 1.0 / (PetscReal)(mx);
    Hy    = 1.0 / (PetscReal)(my);
    HxdHy = Hx/Hy;
    HydHx = Hy/Hx;
    ierr  = DMDAGetCorners(da,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);
    for (j=ys; j<ys+ym; j++) {
        for (i=xs; i<xs+xm; i++) {
            row.i = i; row.j = j;
            if (i==0 || j==0 || i==mx-1 || j==my-1) {
                if (user->bcType == DIRICHLET) {
                    v[0] = 2.0*(HxdHy + HydHx);
                    ierr = MatSetValuesStencil(jac, 1, &row, 1, &row, v, INSERT_VALUES); CHKERRQ(ierr);
                    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "Dirichlet boundary conditions not supported !\n");
                }
                else if (user->bcType == NEUMANN) {
                    num = 0; numi=0; numj=0;
                    if (j!=0) {
                        v[num] = -HxdHy;
                        col[num].i = i;
                        col[num].j = j-1;
                        num++; numj++;
                    }
                    if (i!=0) {
                        v[num]     = -HydHx;
                        col[num].i = i-1;
                        col[num].j = j;
                        num++; numi++;
                    }
                    if (i!=mx-1) {
                        v[num]     = -HydHx;
                        col[num].i = i+1;
                        col[num].j = j;
                        num++; numi++;
                    }
                    if (j!=my-1) {
                        v[num]     = -HxdHy;
                        col[num].i = i;
                        col[num].j = j+1;
                        num++; numj++;
                    }
                    v[num] = (PetscReal)(numj)*HxdHy + (PetscReal)(numi)*HydHx; col[num].i = i;   col[num].j = j;
                    num++;
                    ierr = MatSetValuesStencil(jac,1,&row,num,col,v,INSERT_VALUES);CHKERRQ(ierr);
                }
            } else {
                v[0] = -HxdHy;              col[0].i = i;   col[0].j = j-1;
                v[1] = -HydHx;              col[1].i = i-1; col[1].j = j;
                v[2] = 2.0*(HxdHy + HydHx); col[2].i = i;   col[2].j = j;
                v[3] = -HydHx;              col[3].i = i+1; col[3].j = j;
                v[4] = -HxdHy;              col[4].i = i;   col[4].j = j+1;
                ierr = MatSetValuesStencil(jac,1,&row,5,col,v,INSERT_VALUES);CHKERRQ(ierr);
            }
        }
    }
    ierr = MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    if (user->bcType == NEUMANN) {
        MatNullSpace nullspace;

        ierr = MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,0,0,&nullspace);CHKERRQ(ierr);
        ierr = MatSetNullSpace(J,nullspace);CHKERRQ(ierr);
        ierr = MatNullSpaceDestroy(&nullspace);CHKERRQ(ierr);
    }
    PetscFunctionReturn(ierr);
}
