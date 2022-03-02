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

    ierr = VecCreate(PETSC_COMM_WORLD, &x); CHKERRQ(ierr);
    ierr = VecSetSizes(x, PETSC_DECIDE, m_N); CHKERRQ(ierr);
    ierr = VecSetFromOptions(x); CHKERRQ(ierr);
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
