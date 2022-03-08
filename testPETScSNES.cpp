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
/* "Newton's method for a two-variable system, sequential. */
PetscErrorCode testPETScSNES::testSNES_Sol2VarSysSeq()
{
    PetscScalar    pfive = .5, *xx;

    if (m_size > 1){
        SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "Example is only for sequential runs");
    }

    /* - - - - -  - - - - Create nonlinear solver context - - - - - - - - - - */
    ierr = SNESCreate(PETSC_COMM_WORLD, &snes); CHKERRQ(ierr);

    /* - - - - - Create vectors for solution and nonlinear function - - - - - */
    ierr = VecCreate(PETSC_COMM_WORLD, &x); CHKERRQ(ierr);
    ierr = VecSetSizes(x, PETSC_DECIDE, 2); CHKERRQ(ierr);
    ierr = VecSetFromOptions(x); CHKERRQ(ierr);
    ierr = VecDuplicate(x, &r); CHKERRQ(ierr);

    /* - - - - -  - - - - Create Jacobian matrix data structure - - - - - */
    ierr = MatCreate(PETSC_COMM_WORLD, &J); CHKERRQ(ierr);
    ierr = MatSetSizes(J, PETSC_DECIDE, PETSC_DECIDE, 2, 2); CHKERRQ(ierr);
    ierr = MatSetFromOptions(J); CHKERRQ(ierr);
    ierr = MatSetUp(J); CHKERRQ(ierr);

    PetscBool      flg;
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
      ierr = VecSet(x,pfive); CHKERRQ(ierr);
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
      ierr = VecView(x, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
      ierr = SNESGetFunction(snes, &f, 0, 0); CHKERRQ(ierr);
      ierr = VecView(r, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Free work space.  All PETSc objects should be destroyed when they
       are no longer needed.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    ierr = VecDestroy(&x); CHKERRQ(ierr);
    ierr = VecDestroy(&r); CHKERRQ(ierr);
    ierr = MatDestroy(&J); CHKERRQ(ierr);
    ierr = SNESDestroy(&snes); CHKERRQ(ierr);

    return ierr;
}

PetscErrorCode testPETScSNES::FormFunctionEx1_1(SNES snes,Vec x,Vec f,void *ctx)
{
    PetscErrorCode    ierr;
    const PetscScalar *xx;
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
    ierr = VecGetArrayRead(x,&xx);CHKERRQ(ierr);

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
    PetscErrorCode    ierr;
    const PetscScalar *xx;
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
    PetscInt          idx[2] = {0,1};

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
