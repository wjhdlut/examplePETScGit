#include"applicationExamples.h"

applicationExamples::applicationExamples()
{

}

applicationExamples::applicationExamples(int rank, int size)
{
    m_rank = rank;
    m_size = size;
}

applicationExamples::~applicationExamples()
{

}

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
            solution[Ii] = (2. * PETSC_PI * x) * (2. * PETSC_PI * y);
            userb[Ii]    = -2. * PETSC_PI * (2. * PETSC_PI * x) * (2. * PETSC_PI * y)
                          + 8. * pow(PETSC_PI, 2.) * x * (2. * PETSC_PI * x) * (2. * PETSC_PI * y);
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
        ierr = KSPSetOperators(m_userCtx.ksp, m_userCtx.A, m_userCtx.A);CHKERRQ(ierr);

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
}

