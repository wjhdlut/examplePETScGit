#include"testPETScSNES.h"

testPETScSNES::testPETScSNES()
{}

testPETScSNES::testPETScSNES(int rank, int size)
{
    m_rank = rank;
    m_size = size;


}

testPETScSNES::~testPETScSNES()
{

}

/* ------------------ SNES example ex_1 ------------------ */
/* "Newton's method for a two-variable system, sequential.
 f[0] = sin(3*x[0])+x[0];
 f[1] = x[1]      */
// glosize = 2;
PetscErrorCode testPETScSNES::testSNES_Sol2VarSysSeq()
{
    PetscScalar  pfive = .5;
    PetscScalar  *xx;

    if (m_size > 1){
        SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "Example is only for sequential runs");
    }

    /* - - - - -  - - - - Create nonlinear solver context - - - - - - - - - - */
    ierr = SNESCreate(PETSC_COMM_WORLD, &snes); CHKERRQ(ierr);

    /* - - - - - Create vectors for solution and nonlinear function - - - - - */
    ierr = VecCreate(PETSC_COMM_WORLD, &x); CHKERRQ(ierr);
    glosize = 2;
    ierr = PetscOptionsGetInt(NULL, "The global size of problem", "-glosize", &glosize, NULL); CHKERRQ(ierr);

    ierr = VecSetSizes(x, PETSC_DECIDE, glosize); CHKERRQ(ierr);
    ierr = VecSetFromOptions(x); CHKERRQ(ierr);
    ierr = VecDuplicate(x, &r); CHKERRQ(ierr);

    /* - - - - -  - - - - Create Jacobian matrix data structure - - - - - */
    ierr = MatCreate(PETSC_COMM_WORLD, &J); CHKERRQ(ierr);
    ierr = MatSetSizes(J, PETSC_DECIDE, PETSC_DECIDE, glosize, glosize); CHKERRQ(ierr);
    ierr = MatSetFromOptions(J); CHKERRQ(ierr);
    ierr = MatSetUp(J); CHKERRQ(ierr);

    PetscBool flg;
    ierr = PetscOptionsHasName(NULL, NULL, "-hard", &flg); CHKERRQ(ierr);

    if (!flg) {
        /* Set function evaluation routine and vector. */
        ierr = SNESSetFunction(snes, r, FormFunctionEx1_1, NULL); CHKERRQ(ierr);

        /* Set Jacobian matrix data structure and Jacobian evaluation routine */
        ierr = SNESSetJacobian(snes, J, J, FormJacobianEx1_1, NULL); CHKERRQ(ierr);
    }
    else {
        ierr = SNESSetFunction(snes, r, FormFunctionEx1_2, NULL); CHKERRQ(ierr);
        ierr = SNESSetJacobian(snes, J, J, FormJacobianEx1_2, NULL); CHKERRQ(ierr);
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
              Customize nonlinear solver; set runtime options
       Set linear solver defaults for this problem. By extracting the
       KSP and PC contexts from the SNES context, we can then
       directly call any KSP and PC routines to set various options.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    ierr = SNESGetKSP(snes, &ksp); CHKERRQ(ierr);
    ierr = KSPGetPC(ksp, &pc); CHKERRQ(ierr);
    ierr = PCSetType(pc, PCNONE); CHKERRQ(ierr);
    ierr = KSPSetTolerances(ksp, 1.e-4, PETSC_DEFAULT, PETSC_DEFAULT, 20); CHKERRQ(ierr);

    /* Set SNES/KSP/KSP/PC runtime options, e.g.,
           -snes_view -snes_monitor -ksp_type <ksp> -pc_type <pc>
       These options will override those specified above as long as
       SNESSetFromOptions() is called _after_ any other customization
       routines. */
    ierr = SNESSetFromOptions(snes); CHKERRQ(ierr);

    /* - - -  Evaluate initial guess; then solve nonlinear system - - - - - */
    if (!flg) {
        ierr = VecSet(x, pfive); CHKERRQ(ierr);
    }
    else {
        ierr  = VecGetArray(x, &xx); CHKERRQ(ierr);
        xx[0] = 2.0;
        xx[1] = 3.0;
        ierr  = VecRestoreArray(x, &xx); CHKERRQ(ierr);
    }

    /* Note: The user should initialize the vector, x, with the initial guess
       for the nonlinear solver prior to calling SNESSolve().  In particular,
       to employ an initial guess of zero, the user should explicitly set
       this vector to zero by calling VecSet(). */
    ierr = SNESSolve(snes, NULL, x); CHKERRQ(ierr);
    if (flg) {
        Vec f;
        ierr = PetscPrintf(PETSC_COMM_WORLD, "Print the solution vector x:\n"); CHKERRQ(ierr);
        ierr = VecView(x, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
        ierr = SNESGetFunction(snes, &f, 0, 0); CHKERRQ(ierr);

        ierr = PetscPrintf(PETSC_COMM_WORLD, "Print the function vector f:\n"); CHKERRQ(ierr);
        ierr = VecView(f, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

        ierr = PetscPrintf(PETSC_COMM_WORLD, "Print the residual vector r:\n"); CHKERRQ(ierr);
        ierr = VecView(r, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
    }

    ierr = VecDestroy(&x); CHKERRQ(ierr);
    ierr = VecDestroy(&r); CHKERRQ(ierr);
    ierr = MatDestroy(&J); CHKERRQ(ierr);
    ierr = SNESDestroy(&snes); CHKERRQ(ierr);

    return ierr;
}

PetscErrorCode testPETScSNES::FormFunctionEx1_1(SNES snes,Vec x,Vec f,void *ctx)
{
    const PetscScalar *xx;
    PetscErrorCode    ierr;
    PetscScalar       *ff;

    /* Get pointers to vector data.
        - For default PETSc vectors, VecGetArray() returns a pointer to
          the data array.  Otherwise, the routine is implementation dependent.
        - You MUST call VecRestoreArray() when you no longer need access to
          the array. */
    ierr = VecGetArrayRead(x, &xx); CHKERRQ(ierr);
    ierr = VecGetArray(f, &ff); CHKERRQ(ierr);

    /* Compute function */
    ff[0] = xx[0]*xx[0] + xx[0]*xx[1] - 3.0;
    ff[1] = xx[0]*xx[1] + xx[1]*xx[1] - 6.0;

    /* Restore vectors */
    ierr = VecRestoreArrayRead(x, &xx); CHKERRQ(ierr);
    ierr = VecRestoreArray(f, &ff); CHKERRQ(ierr);
    return ierr;
}

PetscErrorCode testPETScSNES::FormJacobianEx1_1(SNES snes, Vec x, Mat jac, Mat B, void *dummy)
{
    const PetscScalar *xx;
    PetscScalar       A[4];
    PetscErrorCode    ierr;
    PetscInt          idx[2] = {0,1};

    /* Get pointer to vector data */
    ierr = VecGetArrayRead(x, &xx); CHKERRQ(ierr);

    /* Compute Jacobian entries and insert into matrix.
        - Since this is such a small problem, we set all entries for
          the matrix at once. */
    A[0] = 2.0*xx[0] + xx[1];
    A[1] = xx[0];
    A[2] = xx[1];
    A[3] = xx[0] + 2.0*xx[1];
    ierr = MatSetValues(B, 2, idx, 2, idx, A, INSERT_VALUES); CHKERRQ(ierr);

    /* Restore vector */
    ierr = VecRestoreArrayRead(x, &xx); CHKERRQ(ierr);

    /* Assemble matrix */
    ierr = MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

    if (jac != B) {
        ierr = MatAssemblyBegin(jac, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
        ierr = MatAssemblyEnd(jac, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    }

    return ierr;
}

PetscErrorCode testPETScSNES::FormFunctionEx1_2(SNES snes, Vec x, Vec f, void *dummy)
{
    const PetscScalar *xx;
    PetscErrorCode    ierr;
    PetscScalar       *ff;

    /* Get pointers to vector data.
         - For default PETSc vectors, VecGetArray() returns a pointer to
           the data array.  Otherwise, the routine is implementation dependent.
         - You MUST call VecRestoreArray() when you no longer need access to
           the array. */
    ierr = VecGetArrayRead(x, &xx); CHKERRQ(ierr);
    ierr = VecGetArray(f, &ff); CHKERRQ(ierr);

    /* Compute function */
    ff[0] = PetscSinScalar(3.0*xx[0]) + xx[0];
    ff[1] = xx[1];

    /* Restore vectors */
    ierr = VecRestoreArrayRead(x, &xx); CHKERRQ(ierr);
    ierr = VecRestoreArray(f, &ff); CHKERRQ(ierr);

    return ierr;
}

PetscErrorCode testPETScSNES::FormJacobianEx1_2(SNES snes, Vec x, Mat jac, Mat B, void *dummy)
{
    const PetscScalar *xx;
    PetscScalar       A[4];
    PetscErrorCode    ierr;
    PetscInt          idx[2] = {0, 1};

    /* Get pointer to vector data */
    ierr = VecGetArrayRead(x,&xx);CHKERRQ(ierr);

    /* Compute Jacobian entries and insert into matrix.
        - Since this is such a small problem, we set all entries for
          the matrix at once. */
    A[0] = 3.0*PetscCosScalar(3.0*xx[0]) + 1.0;
    A[1] = 0.0;
    A[2] = 0.0;
    A[3] = 1.0;
    ierr = MatSetValues(B, 2, idx, 2, idx, A, INSERT_VALUES); CHKERRQ(ierr);

    /* Restore vector */
    ierr = VecRestoreArrayRead(x, &xx); CHKERRQ(ierr);

    /* Assemble matrix */
    ierr = MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

    if (jac != B) {
        ierr = MatAssemblyBegin(jac, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
        ierr = MatAssemblyEnd(jac, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    }

    return ierr;
}

/* ------------------ SNES example ex_2 ------------------ */
/* the form of nonlinear function is u'' + u^{2} = f */
PetscErrorCode testPETScSNES::testSNES_SolNewtonMethSeq()
{
    if (m_size != 1){
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "This is a uniprocessor example only!");
    }
    PetscInt n = 5;
    ierr = PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL); CHKERRQ(ierr);
    PetscScalar h    = 1.0/(n-1);

    /* - - - - - - - Create nonlinear solver context - - - - - - - - - - - */
    ierr = SNESCreate(PETSC_COMM_WORLD, &snes); CHKERRQ(ierr);

    /* - - Create vector data structures; set function evaluation routine - - */
    ierr = VecCreate(PETSC_COMM_WORLD, &x); CHKERRQ(ierr);
    ierr = VecSetSizes(x, PETSC_DECIDE, n); CHKERRQ(ierr);
    ierr = VecSetFromOptions(x); CHKERRQ(ierr);

    /* - - - - - - - - - - - - duplicate as needed. - - - - - - - - - - - - */
    Vec F, U;
    ierr = VecDuplicate(x, &r);CHKERRQ(ierr);
    ierr = VecDuplicate(x, &F);CHKERRQ(ierr);
    ierr = VecDuplicate(x, &U);CHKERRQ(ierr);

    /* Set function evaluation routine and vector */
    ierr = SNESSetFunction(snes, r, FormFunctionEx2, (void*)F); CHKERRQ(ierr);

    /* - - Create matrix data structure; set Jacobian evaluation routine - - */
    ierr = MatCreate(PETSC_COMM_WORLD, &J); CHKERRQ(ierr);
    ierr = MatSetSizes(J, PETSC_DECIDE, PETSC_DECIDE, n, n); CHKERRQ(ierr);
    ierr = MatSetFromOptions(J); CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(J, 3, NULL); CHKERRQ(ierr);

    /* Set Jacobian matrix data structure and default Jacobian evaluation
       routine. User can override with:
       -snes_fd : default finite differencing approximation of Jacobian
       -snes_mf : matrix-free Newton-Krylov method with no preconditioning
                  (unless user explicitly sets preconditioner)
       -snes_mf_operator : form preconditioning matrix as set by the user,
                           but use matrix-free approx for Jacobian-vector
                           products within Newton-Krylov method */
    ierr = SNESSetJacobian(snes, J, J, FormJacobianEx2, NULL); CHKERRQ(ierr);

    /* - - - - Customize nonlinear solver; set runtime options - - - - - */
    MonitorCtx monP;
    ierr = PetscViewerDrawOpen(PETSC_COMM_WORLD, 0, 0, 0, 0, 400, 400, &monP.viewer); CHKERRQ(ierr);
    ierr = SNESMonitorSet(snes, Monitor, &monP, 0); CHKERRQ(ierr);

    /* Set names for some vectors to facilitate monitoring (optional) */
    ierr = PetscObjectSetName((PetscObject)x, "Approximate Solution"); CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)U, "Exact Solution"); CHKERRQ(ierr);

    /* Set SNES/KSP/KSP/PC runtime options, e.g.,
           -snes_view -snes_monitor -ksp_type <ksp> -pc_type <pc> */
    ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

    /* Print parameters used for convergence testing (optional) ... just
       to demonstrate this routine; this information is also printed with
       the option -snes_view */
    PetscInt    maxit, maxf;
    PetscScalar abstol, rtol, stol;
    ierr = SNESGetTolerances(snes, &abstol, &rtol, &stol, &maxit, &maxf); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "atol=%g, rtol=%g, stol=%g, maxit=%D, maxf=%D\n",
                       (double)abstol, (double)rtol, (double)stol, maxit, maxf); CHKERRQ(ierr);

    /* Initialize application: Store right-hand-side of PDE and exact solution */

    PetscScalar v, xp = 0.0;
    for (auto i=0; i<n; i++) {
        /* +1.e-12 is to prevent 0^6 */
        v    = 6.0*xp + PetscPowScalar(xp+1.e-12,6.0);
        ierr = VecSetValues(F, 1, &i, &v, INSERT_VALUES); CHKERRQ(ierr);
        v    = xp*xp*xp;
        ierr = VecSetValues(U, 1, &i, &v, INSERT_VALUES); CHKERRQ(ierr);
        xp  += h;
    }

    /* - - - - Evaluate initial guess; then solve nonlinear system - - - - */
    /* Note: The user should initialize the vector, x, with the initial guess
       for the nonlinear solver prior to calling SNESSolve().  In particular,
       to employ an initial guess of zero, the user should explicitly set
       this vector to zero by calling VecSet(). */
    PetscInt    its;
    PetscScalar pfive = 0.5;
    ierr = VecSet(x, pfive); CHKERRQ(ierr);
    ierr = SNESSolve(snes, NULL, x); CHKERRQ(ierr);
    ierr = SNESGetIterationNumber(snes, &its);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "number of SNES iterations = %D\n\n", its); CHKERRQ(ierr);

    /* - - - - - - - - - Check solution and clean up - - - - - - - - - - - */
    PetscScalar norm, none = -1.0;
    ierr = VecAXPY(x, none, U);CHKERRQ(ierr);
    ierr = VecNorm(x, NORM_2, &norm); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Norm of error %g, Iterations %D\n", (double)norm, its); CHKERRQ(ierr);

    /* Free work space.  All PETSc objects should be destroyed when they
       are no longer needed. */
    ierr = VecDestroy(&x); CHKERRQ(ierr);
    ierr = VecDestroy(&r); CHKERRQ(ierr);
    ierr = VecDestroy(&U); CHKERRQ(ierr);
    ierr = VecDestroy(&F); CHKERRQ(ierr);
    ierr = MatDestroy(&J); CHKERRQ(ierr);
    ierr = SNESDestroy(&snes); CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&monP.viewer); CHKERRQ(ierr);

    return ierr;
}

PetscErrorCode testPETScSNES::FormFunctionEx2(SNES snes, Vec x, Vec f, void *ctx)
{
    PetscErrorCode    ierr;

    /* Get pointers to vector data.
         - For default PETSc vectors, VecGetArray() returns a pointer to
           the data array.  Otherwise, the routine is implementation dependent.
         - You MUST call VecRestoreArray() when you no longer need access to
           the array. */
    const PetscScalar *xx;
    const PetscScalar *gg;
    PetscScalar *ff;

    Vec g = (Vec)ctx;
    ierr = VecGetArrayRead(x, &xx);CHKERRQ(ierr);
    ierr = VecGetArray(f, &ff);CHKERRQ(ierr);
    ierr = VecGetArrayRead(g, &gg);CHKERRQ(ierr);

    /* Compute function */
    PetscInt n;
    ierr  = VecGetSize(x,&n);CHKERRQ(ierr);

    PetscScalar d = (PetscReal)(n - 1);
    d = d*d;

    ff[0] = xx[0];
    for (auto i=1; i<n-1; i++){
        ff[i] = d*(xx[i-1] - 2.0*xx[i] + xx[i+1]) + xx[i]*xx[i] - gg[i];
    }
    ff[n-1] = xx[n-1] - 1.0;

    /* Restore vectors */
    ierr = VecRestoreArrayRead(x, &xx); CHKERRQ(ierr);
    ierr = VecRestoreArray(f, &ff); CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(g, &gg); CHKERRQ(ierr);

    return ierr;
}

PetscErrorCode testPETScSNES::FormJacobianEx2(SNES snes, Vec x, Mat jac, Mat B, void *dummy)
{
    PetscErrorCode    ierr;

    /* Get pointer to vector data */
    const PetscScalar *xx;
    ierr = VecGetArrayRead(x, &xx); CHKERRQ(ierr);

    /* Compute Jacobian entries and insert into matrix.
        - Note that in this case we set all elements for a particular
          row at once. */
    PetscInt n;
    ierr = VecGetSize(x, &n); CHKERRQ(ierr);

    PetscScalar d = (PetscReal)(n - 1);
    d    = d*d;

    /* Interior grid points */
    PetscInt *j = new PetscInt[3];
    PetscScalar *A = new PetscScalar[3];
    for (auto i=1; i<n-1; i++) {
        j[0] = i - 1;
        j[1] = i;
        j[2] = i + 1;
        A[0] = A[2] = d;
        A[1] = -2.0*d + 2.0*xx[i];
        ierr = MatSetValues(B, 1, &i, 3, j, A, INSERT_VALUES); CHKERRQ(ierr);
    }
    delete[] j;

    /* Boundary points */
    auto i = 0;
    A[0] = 1.0;
    ierr = MatSetValues(B, 1, &i, 1, &i, A, INSERT_VALUES); CHKERRQ(ierr);

    i = n - 1;
    A[0] = 1.0;

    ierr = MatSetValues(B, 1, &i, 1, &i, A, INSERT_VALUES); CHKERRQ(ierr);

    /* Restore vector */
    ierr = VecRestoreArrayRead(x, &xx); CHKERRQ(ierr);

    /* Assemble matrix */
    ierr = MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

    if (jac != B) {
        ierr = MatAssemblyBegin(jac, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
        ierr = MatAssemblyEnd(jac, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    }
    delete[] A;

    return ierr;
}

PetscErrorCode testPETScSNES::Monitor(SNES snes, PetscInt its, PetscReal fnorm, void *ctx)
{
    PetscErrorCode ierr;
    MonitorCtx     *monP = (MonitorCtx*) ctx;
    Vec            x;

    ierr = PetscPrintf(PETSC_COMM_WORLD, "iter = %D, SNES Function norm %g\n", its, (double)fnorm); CHKERRQ(ierr);
    ierr = SNESGetSolution(snes, &x); CHKERRQ(ierr);
    ierr = VecView(x, monP->viewer); CHKERRQ(ierr);
    //ierr = PetscSleep(10.); CHKERRQ(ierr);

    return ierr;
}
