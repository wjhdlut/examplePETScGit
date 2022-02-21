#include<iostream>
#include"testPETScKSP.h"

testPETScKSP::testPETScKSP()
{

}

testPETScKSP::testPETScKSP(int rank, int size)
{
    m_rank = rank;
    m_size = size;
}

testPETScKSP::~testPETScKSP()
{

}

/* ------------------ KSP example ex_1 ------------------ */
PetscErrorCode testPETScKSP::testKSP_SolTridiagonalLinearSysSeq()
{
    PetscInt       i, n = 10, col[3], its;
    PetscScalar    value[3];

    if (m_size != 1)
    {
        SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE,
                "This is a uniprocessor example only!");
    }
    ierr = PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL); CHKERRQ(ierr);

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
           Compute the matrix and right-hand-side vector that define
           the linear system, Ax = b.
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    /* Create vectors.  Note that we form 1 vector from scratch and
       then duplicate as needed. */
    ierr = VecCreate(PETSC_COMM_WORLD, &x); CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) x, "Solution"); CHKERRQ(ierr);
    ierr = VecSetSizes(x, PETSC_DECIDE, n); CHKERRQ(ierr);
    ierr = VecSetFromOptions(x); CHKERRQ(ierr);
    ierr = VecDuplicate(x, &b); CHKERRQ(ierr);
    ierr = VecDuplicate(x, &u); CHKERRQ(ierr);

    /* Create matrix.  When using MatCreate(), the matrix format can
       be specified at runtime.
       Performance tuning note:  For problems of substantial size,
       preallocation of matrix memory is crucial for attaining good
       performance. See the matrix chapter of the users manual for details. */
    ierr = MatCreate(PETSC_COMM_WORLD, &A); CHKERRQ(ierr);
    ierr = MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n); CHKERRQ(ierr);
    ierr = MatSetFromOptions(A); CHKERRQ(ierr);
    ierr = MatSetUp(A); CHKERRQ(ierr);

    /* Assemble matrix */
    value[0] = -1.0; value[1] = 2.0; value[2] = -1.0;
    for (i=1; i<n-1; i++)
    {
        col[0] = i-1;
        col[1] = i;
        col[2] = i+1;
        ierr   = MatSetValues(A, 1, &i, 3, col, value, INSERT_VALUES); CHKERRQ(ierr);
    }

    i    = n - 1;
    col[0] = n - 2;
    col[1] = n - 1;
    ierr = MatSetValues(A, 1, &i, 2, col, value, INSERT_VALUES); CHKERRQ(ierr);

    i    = 0;
    col[0] = 0;
    col[1] = 1;
    value[0] = 2.0;
    value[1] = -1.0;
    ierr = MatSetValues(A, 1, &i, 2, col, value, INSERT_VALUES); CHKERRQ(ierr);

    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

    /* Set exact solution; then compute right-hand-side vector. */
    ierr = VecSet(u, 1.0); CHKERRQ(ierr);
    ierr = MatMult(A, u, b); CHKERRQ(ierr);

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                  Create the linear solver and set various options
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    ierr = KSPCreate(PETSC_COMM_WORLD, &ksp); CHKERRQ(ierr);

    /* Set operators. Here the matrix that defines the linear system
       also serves as the matrix that defines the preconditioner. */
    ierr = KSPSetOperators(ksp, A, A); CHKERRQ(ierr);

    /* Set linear solver defaults for this problem (optional).
       - By extracting the KSP and PC contexts from the KSP context,
         we can then directly call any KSP and PC routines to set
         various options.
       - The following four statements are optional; all of these
         parameters could alternatively be specified at runtime via
         KSPSetFromOptions(); */
    ierr = KSPGetPC(ksp, &pc); CHKERRQ(ierr);
    ierr = PCSetType(pc, PCJACOBI); CHKERRQ(ierr);
    ierr = KSPSetTolerances(ksp, 1.e-5, PETSC_DEFAULT, PETSC_DEFAULT,
                            PETSC_DEFAULT); CHKERRQ(ierr);

    /* Set runtime options, e.g.,
          -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
      These options will override those specified above as long as
      KSPSetFromOptions() is called _after_ any other customization
      routines. */
    ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                        Solve the linear system
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    ierr = KSPSolve(ksp, b, x); CHKERRQ(ierr);

    /* View solver info; we could instead use the option -ksp_view to
       print this info to the screen at the conclusion of KSPSolve(). */
    ierr = KSPView(ksp, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                        Check the solution and clean up
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    ierr = CheckError(x, u, ksp); CHKERRQ(ierr);

    /* Free work space.  All PETSc objects should be destroyed when they
       are no longer needed. */
    ierr = VecDestroy(&x); CHKERRQ(ierr);
    ierr = VecDestroy(&u); CHKERRQ(ierr);
    ierr = VecDestroy(&b); CHKERRQ(ierr);
    ierr = MatDestroy(&A); CHKERRQ(ierr);
    ierr = KSPDestroy(&ksp); CHKERRQ(ierr);

    return ierr;
}

/* ------------------ KSP example ex_2 ------------------ */
PetscErrorCode testPETScKSP::testKSP_SolTridiagonalLinearSysPar()
{
    PetscBool      flg;
    PetscScalar    v1 = -1.0, v2 = 4.0;

    m = 8, n = 7;
    ierr = PetscOptionsGetInt(NULL, NULL, "-m", &m, NULL); CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL); CHKERRQ(ierr);
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
           Compute the matrix and right-hand-side vector that define
           the linear system, Ax = b.
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    /* Create parallel matrix, specifying only its global dimensions.
       When using MatCreate(), the matrix format can be specified at
       runtime. Also, the parallel partitioning of the matrix is
       determined by PETSc at runtime.
       Performance tuning note:  For problems of substantial size,
       preallocation of matrix memory is crucial for attaining good
       performance. See the matrix chapter of the users manual for details. */
    ierr = MatCreate(PETSC_COMM_WORLD, &A); CHKERRQ(ierr);
    ierr = MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, m*n, m*n); CHKERRQ(ierr);
    ierr = MatSetFromOptions(A); CHKERRQ(ierr);

    ierr = MatMPIAIJSetPreallocation(A, 5, NULL, 5, NULL); CHKERRQ(ierr);
    //ierr = MatSeqAIJSetPreallocation(A, 5, NULL); CHKERRQ(ierr);

    //ierr = MatSeqSBAIJSetPreallocation(A, 1, 5, NULL); CHKERRQ(ierr);
    ierr = MatMPISBAIJSetPreallocation(A, 1, 5, NULL, 5, NULL); CHKERRQ(ierr);

    //ierr = MatMPISELLSetPreallocation(A, 5, NULL, 5, NULL); CHKERRQ(ierr);
    ierr = MatSeqSELLSetPreallocation(A, 5, NULL); CHKERRQ(ierr);

    /* Currently, all PETSc parallel matrix formats are partitioned by
       contiguous chunks of rows across the processors.  Determine which
       rows of the matrix are locally owned. */
    ierr = MatGetOwnershipRange(A, &Istart, &Iend);CHKERRQ(ierr);

    /* Set matrix elements for the 2-D, five-point stencil in parallel.
        - Each processor needs to insert only elements that it owns
          locally (but any non-local elements will be sent to the
          appropriate processor during matrix assembly).
        - Always specify global rows and columns of matrix entries.

       Note: this uses the less common natural ordering that orders first
       all the unknowns for x = h then for x = 2h etc; Hence you see J = Ii +- n
       instead of J = I +- m as you might expect. The more standard ordering
       would first do all variables for y = h, then y = 2h etc. */
    ierr = FormMatrixA(v1, v2, ADD_VALUES); CHKERRQ(ierr);

    /* Assemble matrix, using the 2-step process:
         MatAssemblyBegin(), MatAssemblyEnd()
       Computations can be done while messages are in transition
       by placing code between these two statements. */
    ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

    ierr = MatView(A, PETSC_VIEWER_STDERR_WORLD); CHKERRQ(ierr);

    /* A is symmetric. Set symmetric flag to enable ICC/Cholesky preconditioner */
    ierr = MatSetOption(A, MAT_SYMMETRIC, PETSC_TRUE); CHKERRQ(ierr);

    /* Create parallel vectors.
        - We form 1 vector from scratch and then duplicate as needed.
        - When using VecCreate(), VecSetSizes and VecSetFromOptions()
          in this example, we specify only the
          vector's global dimension; the parallel partitioning is determined
          at runtime.
        - When solving a linear system, the vectors and matrices MUST
          be partitioned accordingly.  PETSc automatically generates
          appropriately partitioned matrices and vectors when MatCreate()
          and VecCreate() are used with the same communicator.
        - The user can alternatively specify the local vector and matrix
          dimensions when more sophisticated partitioning is needed
          (replacing the PETSC_DECIDE argument in the VecSetSizes() statement
          below). */
    ierr = VecCreate(PETSC_COMM_WORLD, &u); CHKERRQ(ierr);
    ierr = VecSetSizes(u, PETSC_DECIDE, m*n); CHKERRQ(ierr);
    ierr = VecSetFromOptions(u); CHKERRQ(ierr);
    ierr = VecDuplicate(u, &b); CHKERRQ(ierr);
    ierr = VecDuplicate(b, &x); CHKERRQ(ierr);

    /* Set exact solution; then compute right-hand-side vector.
       By default we use an exact solution of a vector with all
       elements of 1.0; */
    ierr = VecSet(u, 1.0); CHKERRQ(ierr);
    ierr = MatMult(A, u, b); CHKERRQ(ierr);

    /* View the exact solution vector if desired */
    flg  = PETSC_FALSE;
    ierr = PetscOptionsGetBool(NULL, NULL, "-view_exact_sol", &flg, NULL); CHKERRQ(ierr);
    if (flg){
        ierr = VecView(u, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                  Create the linear solver and set various options
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    ierr = KSPCreate(PETSC_COMM_WORLD, &ksp); CHKERRQ(ierr);

    /* Set operators. Here the matrix that defines the linear system
       also serves as the preconditioning matrix. */
    ierr = KSPSetOperators(ksp, A, A); CHKERRQ(ierr);

    /* Set linear solver defaults for this problem (optional).
       - By extracting the KSP and PC contexts from the KSP context,
         we can then directly call any KSP and PC routines to set
         various options.
       - The following two statements are optional; all of these
         parameters could alternatively be specified at runtime via
         KSPSetFromOptions().  All of these defaults can be
         overridden at runtime, as indicated below. */
    ierr = KSPSetTolerances(ksp, 1.e-2/((m+1)*(n+1)), 1.e-50, PETSC_DEFAULT, PETSC_DEFAULT); CHKERRQ(ierr);

    /* Set runtime options, e.g.,
          -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
      These options will override those specified above as long as
      KSPSetFromOptions() is called _after_ any other customization
      routines. */
    ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                        Solve the linear system
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    ierr = KSPSolve(ksp, b, x); CHKERRQ(ierr);
    ierr = KSPView(ksp, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                        Check the solution and clean up
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    ierr = CheckError(x, u, ksp); CHKERRQ(ierr);

    /* Free work space.  All PETSc objects should be destroyed when they
       are no longer needed. */
    ierr = KSPDestroy(&ksp); CHKERRQ(ierr);
    ierr = VecDestroy(&u); CHKERRQ(ierr);
    ierr = VecDestroy(&x); CHKERRQ(ierr);
    ierr = VecDestroy(&b); CHKERRQ(ierr);
    ierr = MatDestroy(&A); CHKERRQ(ierr);

    return ierr;
}

/* ------------------ KSP example ex_3 ------------------ */
PetscErrorCode testPETScKSP::testKSP_Laplacian()
{
    PetscInt       N;           /* dimension of system (global) */
    PetscInt       M;           /* number of elements (global) */
    PetscScalar    Ke[16];      /* element matrix */
    PetscScalar    r[4];        /* element vector */
    PetscReal      h;           /* mesh width */
    PetscScalar    xx, yy;
    PetscInt       idx[4], count, i, m = 5;

    ierr = PetscOptionsGetInt(NULL, NULL, "-m", &m, NULL); CHKERRQ(ierr);
    N    = (m+1)*(m+1);
    M    = m*m;
    h    = 1.0/m;

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
           Compute the matrix and right-hand-side vector that define
           the linear system, Ax = b.
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    /* Create stiffness matrix */
    ierr  = MatCreate(PETSC_COMM_WORLD, &A); CHKERRQ(ierr);
    ierr  = MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, N, N); CHKERRQ(ierr);
    ierr  = MatSetFromOptions(A); CHKERRQ(ierr);

    ierr  = MatSeqAIJSetPreallocation(A, 9, NULL); CHKERRQ(ierr);
    ierr  = MatMPIAIJSetPreallocation(A, 9, NULL, 8, NULL); CHKERRQ(ierr);

    Istart = m_rank*(M/m_size) + ((M%m_size) < m_rank ? (M%m_size) : m_rank);
    Iend   = Istart + M/m_size + ((M%m_size) > m_rank);

    /* Assemble matrix */
    ierr = FormElementStiffness(h*h, Ke); CHKERRQ(ierr);

    for (i=Istart; i<Iend; i++) {
        /* node numbers for the four corners of element */
        idx[0] = (m+1)*(i/m) + (i % m);
        idx[1] = idx[0]+1;
        idx[2] = idx[1] + m + 1;
        idx[3] = idx[2] - 1;
        ierr   = MatSetValues(A, 4, idx, 4, idx, Ke, ADD_VALUES); CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

    /* Create right-hand-side and solution vectors */
    ierr = VecCreate(PETSC_COMM_WORLD, &x); CHKERRQ(ierr);
    ierr = VecSetSizes(x, PETSC_DECIDE, N); CHKERRQ(ierr);
    ierr = VecSetFromOptions(x); CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)x,"Approx. Solution"); CHKERRQ(ierr);

    ierr = VecDuplicate(x, &b); CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)b, "Right hand side"); CHKERRQ(ierr);

    ierr = VecDuplicate(b, &u);CHKERRQ(ierr);
    ierr = VecSet(x, 0.0); CHKERRQ(ierr);
    ierr = VecSet(b, 0.0); CHKERRQ(ierr);

    /* Assemble right-hand-side vector */
    for (i=Istart; i<Iend; i++) {
        /* location of lower left corner of element */
        xx = h*(i % m);
        yy = h*(i/m);
        /* node numbers for the four corners of element */
        idx[0] = (m+1)*(i/m) + (i % m);
        idx[1] = idx[0]+1;
        idx[2] = idx[1] + m + 1;
        idx[3] = idx[2] - 1;
        ierr   = FormElementRhs(xx, yy, h*h, r); CHKERRQ(ierr);
        ierr   = VecSetValues(b, 4, idx, r, ADD_VALUES); CHKERRQ(ierr);
    }
    ierr = VecAssemblyBegin(b); CHKERRQ(ierr);
    ierr = VecAssemblyEnd(b); CHKERRQ(ierr);

    /* Modify matrix and right-hand-side for Dirichlet boundary conditions */
    PetscInt *rows = new PetscInt[4*m]();
    for (i=0; i<m+1; i++) {
        rows[i] = i; /* bottom */
        rows[3*m - 1 +i] = m*(m+1) + i; /* top */
    }
    count = m+1; /* left side */
    for (i=m+1; i<m*(m+1); i+= m+1){
        rows[count++] = i;
    }
    count = 2*m; /* left side */
    for (i=2*m+1; i<m*(m+1); i+= m+1){
        rows[count++] = i;
    }
    for (i=0; i<4*m; i++) {
        yy = h*(rows[i]/(m+1));
        ierr = VecSetValues(x, 1, &rows[i], &yy, INSERT_VALUES); CHKERRQ(ierr);
        ierr = VecSetValues(b, 1, &rows[i], &yy, INSERT_VALUES); CHKERRQ(ierr);
    }
    ierr = MatZeroRows(A, 4*m, rows, 1.0, 0,0); CHKERRQ(ierr);
    delete[] rows;

    ierr = VecAssemblyBegin(x);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(x);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(b);CHKERRQ(ierr);

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                  Create the linear solver and set various options
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    ierr = KSPCreate(PETSC_COMM_WORLD, &ksp); CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp, A, A); CHKERRQ(ierr);
    ierr = KSPSetInitialGuessNonzero(ksp, PETSC_TRUE); CHKERRQ(ierr);
    ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                        Solve the linear system
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    ierr = KSPSolve(ksp, b, x);CHKERRQ(ierr);

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                        Check solution and clean up
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    /* Check error */
    ierr = VecGetOwnershipRange(u, &Istart, &Iend); CHKERRQ(ierr);
    for (i=Istart; i<Iend; i++) {
        yy = h*(i/(m+1));
        ierr = VecSetValues(u, 1, &i, &yy, INSERT_VALUES); CHKERRQ(ierr);
    }
    ierr = VecAssemblyBegin(u); CHKERRQ(ierr);
    ierr = VecAssemblyEnd(u); CHKERRQ(ierr);

    ierr = CheckError(x, u, ksp); CHKERRQ(ierr);

    /* Free work space.  All PETSc objects should be destroyed when they
       are no longer needed. */
    ierr = KSPDestroy(&ksp); CHKERRQ(ierr);
    ierr = VecDestroy(&x); CHKERRQ(ierr);
    ierr = VecDestroy(&u); CHKERRQ(ierr);
    ierr = VecDestroy(&b); CHKERRQ(ierr);
    ierr = MatDestroy(&A); CHKERRQ(ierr);

    return ierr;
}

PetscErrorCode testPETScKSP::FormElementStiffness(PetscReal H,PetscScalar *Ke)
{
    PetscFunctionBeginUser;
    Ke[0]  = H/6.0;
    Ke[1]  = -.125*H;
    Ke[2]  = H/12.0;
    Ke[3]  = -.125*H;
    Ke[4]  = -.125*H;
    Ke[5]  = H/6.0;
    Ke[6]  = -.125*H;
    Ke[7]  = H/12.0;
    Ke[8]  = H/12.0;
    Ke[9]  = -.125*H;
    Ke[10] = H/6.0;
    Ke[11] = -.125*H;
    Ke[12] = -.125*H;
    Ke[13] = H/12.0;
    Ke[14] = -.125*H;
    Ke[15] = H/6.0;
    PetscFunctionReturn(0);
}

PetscErrorCode testPETScKSP::FormElementRhs(PetscScalar x,PetscScalar y,PetscReal H,PetscScalar *r)
{
    PetscFunctionBeginUser;
    r[0] = 0.;
    r[1] = 0.;
    r[2] = 0.;
    r[3] = 0.0;
    PetscFunctionReturn(0);
}

/* ------------------ KSP example ex_4 ------------------ */
PetscErrorCode testPETScKSP::testKSP_PCHMG()
{
    PetscBool      flg;
    PetscBool      test=PETSC_FALSE, reuse=PETSC_FALSE, viewexpl=PETSC_FALSE;
    PetscScalar    v;
    PetscScalar    v1 = -1.0, v2 = 4.0;

    bs = 1;
    ierr = PetscOptionsGetInt(NULL, NULL, "-m", &m, NULL); CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL); CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL, NULL, "-bs", &bs, NULL); CHKERRQ(ierr);

    ierr = PetscOptionsGetBool(NULL, NULL, "-test_hmg_interface", &test, NULL); CHKERRQ(ierr);
    ierr = PetscOptionsGetBool(NULL, NULL, "-test_reuse_interpolation", &reuse, NULL); CHKERRQ(ierr);
    ierr = PetscOptionsGetBool(NULL, NULL, "-view_explicit_mat", &viewexpl, NULL); CHKERRQ(ierr);

    ierr = MatCreate(PETSC_COMM_WORLD, &A); CHKERRQ(ierr);
    ierr = MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, m*n*bs, m*n*bs); CHKERRQ(ierr);
    ierr = MatSetBlockSize(A, bs); CHKERRQ(ierr);
    ierr = MatSetFromOptions(A); CHKERRQ(ierr);

    ierr = MatMPIAIJSetPreallocation(A, 5, NULL, 5, NULL); CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(A, 5, NULL); CHKERRQ(ierr);

    ierr = MatGetOwnershipRange(A, &Istart, &Iend); CHKERRQ(ierr);

    RANK_COUT << " Istart = " << Istart << ", Iend = " << Iend << std::endl;

    ierr = FormBlockMatrixA(v1, v2, ADD_VALUES); CHKERRQ(ierr);

    ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

    //ierr = MatView(A, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

    if (viewexpl) {
        Mat E;
        ierr = MatComputeOperator(A, MATAIJ, &E); CHKERRQ(ierr);
        ierr = MatView(E, NULL); CHKERRQ(ierr);
        ierr = MatDestroy(&E); CHKERRQ(ierr);
    }

    ierr = MatCreateVecs(A, &u, NULL); CHKERRQ(ierr);
    ierr = VecSetFromOptions(u); CHKERRQ(ierr);
    ierr = VecDuplicate(u, &b); CHKERRQ(ierr);
    ierr = VecDuplicate(b, &x); CHKERRQ(ierr);

    ierr = VecSet(u, 1.0); CHKERRQ(ierr);
    ierr = MatMult(A, u, b); CHKERRQ(ierr);

    flg  = PETSC_FALSE;
    ierr = PetscOptionsGetBool(NULL, NULL, "-view_exact_sol", &flg, NULL); CHKERRQ(ierr);
    if (flg){
        ierr = VecView(u, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
    }

    ierr = KSPCreate(PETSC_COMM_WORLD, &ksp); CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp, A, A); CHKERRQ(ierr);
    ierr = KSPSetTolerances(ksp, 1.e-2/((m+1)*(n+1)), 1.e-50,
                            PETSC_DEFAULT, PETSC_DEFAULT); CHKERRQ(ierr);

    if (test) {
        ierr = KSPGetPC(ksp, &pc); CHKERRQ(ierr);
        ierr = PCSetType(pc, PCHMG); CHKERRQ(ierr);

        ierr = PCHMGSetInnerPCType(pc, PCGAMG); CHKERRQ(ierr);
        ierr = PCHMGSetReuseInterpolation(pc, PETSC_TRUE); CHKERRQ(ierr);
        ierr = PCHMGSetUseSubspaceCoarsening(pc, PETSC_TRUE); CHKERRQ(ierr);
        ierr = PCHMGUseMatMAIJ(pc, PETSC_FALSE); CHKERRQ(ierr);
        ierr = PCHMGSetCoarseningComponent(pc, 0); CHKERRQ(ierr);
    }

    ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);
    ierr = KSPSolve(ksp, b, x); CHKERRQ(ierr);

    if (reuse) {
        ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
        ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
        ierr = KSPSolve(ksp, b, x); CHKERRQ(ierr);

        ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
        ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
        ierr = KSPSolve(ksp, b, x); CHKERRQ(ierr);

        /* Make sparsity pattern different and reuse interpolation */
        ierr = MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE); CHKERRQ(ierr);
        ierr = MatSetOption(A, MAT_IGNORE_ZERO_ENTRIES, PETSC_FALSE); CHKERRQ(ierr);
        ierr = MatGetSize(A, &m, NULL); CHKERRQ(ierr);
        n = 0;
        v = 0;
        m--;

        /* Connect the last element to the first element */
        ierr = MatSetValue(A, m, n, v, ADD_VALUES); CHKERRQ(ierr);
        ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
        ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
        ierr = KSPSolve(ksp, b, x); CHKERRQ(ierr);
    }
    ierr = CheckError(x, u, ksp); CHKERRQ(ierr);

    ierr = KSPDestroy(&ksp); CHKERRQ(ierr);
    ierr = VecDestroy(&u); CHKERRQ(ierr);
    ierr = VecDestroy(&x); CHKERRQ(ierr);
    ierr = VecDestroy(&b); CHKERRQ(ierr);
    ierr = MatDestroy(&A); CHKERRQ(ierr);

    return ierr;
}

/* ------------------ KSP example ex_5 ------------------ */
PetscErrorCode testPETScKSP::testKSP_SolTwoLinearSys()
{
    PetscInt       Ii, J, ldim, low, high, iglobal;
    PetscInt       i, j;
    PetscBool      mat_nonsymmetric = PETSC_FALSE;
    PetscBool      testnewC=PETSC_FALSE, testscaledMat=PETSC_FALSE;
    PetscScalar    v1 = -1.0, v2 = 4.0, v3 = 6.0;

    m = 3, n = 2;
    ierr = PetscOptionsGetInt(NULL, NULL, "-m", &m, NULL); CHKERRQ(ierr);
    n    = 2*m_size;

    /* Set flag if we are doing a nonsymmetric problem; the default is symmetric. */
    ierr = PetscOptionsGetBool(NULL, NULL, "-mat_nonsym", &mat_nonsymmetric, NULL); CHKERRQ(ierr);
    ierr = PetscOptionsGetBool(NULL, NULL, "-test_scaledMat", &testscaledMat, NULL); CHKERRQ(ierr);

    /* -------------- Stage 0: Solve Original System ---------------------- */
    /* Create parallel matrix, specifying only its global dimensions.
       When using MatCreate(), the matrix format can be specified at
       runtime. Also, the parallel partitioning of the matrix is
       determined by PETSc at runtime. */
    ierr = MatCreate(PETSC_COMM_WORLD, &A); CHKERRQ(ierr);
    ierr = MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, m*n, m*n); CHKERRQ(ierr);
    ierr = MatSetFromOptions(A); CHKERRQ(ierr);
    ierr = MatSetUp(A); CHKERRQ(ierr);

    /* Currently, all PETSc parallel matrix formats are partitioned by
       contiguous chunks of rows across the processors.  Determine which
       rows of the matrix are locally owned. */
    ierr = MatGetOwnershipRange(A, &Istart, &Iend); CHKERRQ(ierr);

    RANK_COUT << " Istart = " << Istart << ", Iend = " << Iend << std::endl;

    /* Set matrix entries matrix in parallel.
        - Each processor needs to insert only elements that it owns
          locally (but any non-local elements will be sent to the
          appropriate processor during matrix assembly).
        - Always specify global row and columns of matrix entries. */
    ierr = FormMatrixA(v1, v2, ADD_VALUES); CHKERRQ(ierr);

    /* Make the matrix nonsymmetric if desired */
    PetscScalar v;
    if (mat_nonsymmetric) {
        for (Ii=Istart; Ii<Iend; Ii++) {
            v = -1.5;
            i = Ii/n;
            if (i>1){
                J = Ii-n-1;
                ierr = MatSetValues(A, 1, &Ii, 1, &J, &v, ADD_VALUES); CHKERRQ(ierr);
            }
        }
    } else {
        ierr = MatSetOption(A, MAT_SYMMETRIC, PETSC_TRUE); CHKERRQ(ierr);
        ierr = MatSetOption(A, MAT_SYMMETRY_ETERNAL, PETSC_TRUE); CHKERRQ(ierr);
    }

    /* Assemble matrix, using the 2-step process:
         MatAssemblyBegin(), MatAssemblyEnd()
       Computations can be done while messages are in transition
       by placing code between these two statements. */
    ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

    ierr = MatView(A, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

    /* Create parallel vectors.
        - When using VecSetSizes(), we specify only the vector's global
          dimension; the parallel partitioning is determined at runtime.
        - Note: We form 1 vector from scratch and then duplicate as needed. */
    ierr = VecCreate(PETSC_COMM_WORLD, &u); CHKERRQ(ierr);
    ierr = VecSetSizes(u, PETSC_DECIDE, m*n); CHKERRQ(ierr);
    ierr = VecSetFromOptions(u); CHKERRQ(ierr);
    ierr = VecDuplicate(u, &b); CHKERRQ(ierr);
    ierr = VecDuplicate(b, &x); CHKERRQ(ierr);

    /* Currently, all parallel PETSc vectors are partitioned by
       contiguous chunks across the processors.  Determine which
       range of entries are locally owned. */
    ierr = VecGetOwnershipRange(x, &low, &high);CHKERRQ(ierr);
    RANK_COUT << "low = " << low << ", high = " << high << std::endl;

    /* Set elements within the exact solution vector in parallel.
       - Each processor needs to insert only elements that it owns
         locally (but any non-local entries will be sent to the
         appropriate processor during vector assembly).
       - Always specify global locations of vector entries. */
    ierr = VecGetLocalSize(x, &ldim); CHKERRQ(ierr);
    RANK_COUT << "local size of Vector x is " << ldim << std::endl;

    for (i=0; i<ldim; i++) {
        iglobal = i + low;
        v       = (PetscScalar)(i + 100*m_rank);
        ierr    = VecSetValues(u, 1, &iglobal, &v, INSERT_VALUES); CHKERRQ(ierr);
    }

    /* Assemble vector, using the 2-step process:
         VecAssemblyBegin(), VecAssemblyEnd()
       Computations can be done while messages are in transition,
       by placing code between these two statements. */
    ierr = VecAssemblyBegin(u); CHKERRQ(ierr);
    ierr = VecAssemblyEnd(u); CHKERRQ(ierr);

    ierr = VecView(u, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

    /* Compute right-hand-side vector */
    ierr = MatMult(A, u, b); CHKERRQ(ierr);

    /* Create linear solver context */
    ierr = KSPCreate(PETSC_COMM_WORLD, &ksp); CHKERRQ(ierr);

    /* Set operators. Here the matrix that defines the linear system
       also serves as the preconditioning matrix. */
    ierr = KSPSetOperators(ksp, A, A); CHKERRQ(ierr);

    /* Set runtime options (e.g., -ksp_type <type> -pc_type <type>) */
    ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);

    /* Solve linear system.  Here we explicitly call KSPSetUp() for more
       detailed performance monitoring of certain preconditioners, such
       as ICC and ILU.  This call is optional, as KSPSetUp() will
       automatically be called within KSPSolve() if it hasn't been
       called already. */
    ierr = KSPSetUp(ksp); CHKERRQ(ierr);
    ierr = KSPSolve(ksp, b, x); CHKERRQ(ierr);

    /* Check the error */
    ierr = CheckError(x, u, ksp); CHKERRQ(ierr);

    /* -------------- Stage 1: Solve Second System ---------------------- */
    /* Solve another linear system with the same method.  We reuse the KSP
       context, matrix and vector data structures, and hence save the
       overhead of creating new ones.

    /* Initialize all matrix entries to zero.  MatZeroEntries() retains the
       nonzero structure of the matrix for sparse formats. */
    ierr = MatZeroEntries(A); CHKERRQ(ierr);

    /* Assemble matrix again.  Note that we retain the same matrix data
       structure and the same nonzero pattern; we just change the values
       of the matrix entries. */
    for (i=0; i<m; i++) {
        for (j=2*m_rank; j<2*m_rank+2; j++) {
            v = v1;
            Ii = j + n*i;
            if (i>0){
                J = Ii - n;
                ierr = MatSetValues(A, 1, &Ii, 1, &J, &v, ADD_VALUES); CHKERRQ(ierr);
            }

            if (i<m-1){
                J = Ii + n;
                ierr = MatSetValues(A, 1, &Ii, 1, &J, &v, ADD_VALUES); CHKERRQ(ierr);
            }

            if (j>0){
                J = Ii - 1;
                ierr = MatSetValues(A, 1, &Ii, 1, &J, &v, ADD_VALUES); CHKERRQ(ierr);
            }

            if (j<n-1){
                J = Ii + 1;
                ierr = MatSetValues(A, 1, &Ii, 1, &J, &v, ADD_VALUES); CHKERRQ(ierr);
            }
            v = v3;
            ierr = MatSetValues(A, 1, &Ii, 1, &Ii, &v, ADD_VALUES); CHKERRQ(ierr);
        }
    }

    if (mat_nonsymmetric){
        for (Ii=Istart; Ii<Iend; Ii++){
            v = -1.5;
            i = Ii/n;
            if (i>1){
                J = Ii-n-1;
                ierr = MatSetValues(A, 1, &Ii, 1, &J, &v, ADD_VALUES); CHKERRQ(ierr);
            }
            //            else if (i<m-2)
            //            {
            //                J = Ii+n+1;
            //                RANK_COUT << "Ii = " << Ii << ", J = " << J << std::endl;
            //                ierr = MatSetValues(A, 1, &Ii, 1, &J, &v, ADD_VALUES); CHKERRQ(ierr);
            //            }
        }
    }

    ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

    ierr = MatView(A, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

    if (testscaledMat) {
        PetscRandom    rctx;

        /* Scale a(0,0) and a(M-1,M-1) */
        if (m_rank == 0) {
            v = 6.0*0.00001;
            Ii = 0; J = 0;
            ierr = MatSetValues(A, 1, &Ii, 1, &J, &v, INSERT_VALUES); CHKERRQ(ierr);
        } else if (m_rank == m_size -1) {
            v = 6.0*0.00001;
            Ii = m*n-1;
            J = m*n-1;
            ierr = MatSetValues(A, 1, &Ii, 1, &J, &v, INSERT_VALUES); CHKERRQ(ierr);
        }
        ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
        ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

        /* Compute a new right-hand-side vector */
        ierr = VecDestroy(&u); CHKERRQ(ierr);
        ierr = VecCreate(PETSC_COMM_WORLD, &u); CHKERRQ(ierr);
        ierr = VecSetSizes(u, PETSC_DECIDE, m*n); CHKERRQ(ierr);
        ierr = VecSetFromOptions(u); CHKERRQ(ierr);

        ierr = PetscRandomCreate(PETSC_COMM_WORLD, &rctx); CHKERRQ(ierr);
        ierr = PetscRandomSetFromOptions(rctx); CHKERRQ(ierr);
        ierr = VecSetRandom(u, rctx); CHKERRQ(ierr);
        ierr = PetscRandomDestroy(&rctx); CHKERRQ(ierr);

        ierr = VecAssemblyBegin(u); CHKERRQ(ierr);
        ierr = VecAssemblyEnd(u); CHKERRQ(ierr);
    }

    ierr = PetscOptionsGetBool(NULL, NULL, "-test_newMat", &testnewC, NULL); CHKERRQ(ierr);

    if (testnewC) {
        /* User may use a new matrix C with same nonzero pattern, e.g.
            ./ex5 -ksp_monitor -mat_type sbaij -pc_type cholesky
                  -pc_factor_mat_solver_type mumps -test_newMat */
        Mat Ctmp;
        ierr = MatDuplicate(A, MAT_COPY_VALUES, &Ctmp); CHKERRQ(ierr);
        ierr = MatDestroy(&A); CHKERRQ(ierr);
        ierr = MatDuplicate(Ctmp, MAT_COPY_VALUES, &A); CHKERRQ(ierr);
        ierr = MatDestroy(&Ctmp); CHKERRQ(ierr);
    }

    ierr = MatMult(A, u, b); CHKERRQ(ierr);

    /* Set operators. Here the matrix that defines the linear system
       also serves as the preconditioning matrix. */
    ierr = KSPSetOperators(ksp, A, A); CHKERRQ(ierr);

    /* Solve linear system */
    ierr = KSPSetUp(ksp); CHKERRQ(ierr);
    ierr = KSPSolve(ksp, b, x); CHKERRQ(ierr);

    /* Check the error */
    ierr = CheckError(x, u, ksp); CHKERRQ(ierr);

    /* Free work space.  All PETSc objects should be destroyed when they
       are no longer needed. */
    ierr = KSPDestroy(&ksp); CHKERRQ(ierr);
    ierr = VecDestroy(&u); CHKERRQ(ierr);
    ierr = VecDestroy(&x); CHKERRQ(ierr);
    ierr = VecDestroy(&b); CHKERRQ(ierr);
    ierr = MatDestroy(&A); CHKERRQ(ierr);

    return ierr;
}

/* ------------------ KSP example ex_6 ------------------ */
PetscErrorCode testPETScKSP::testKSP_SolTriLinearKSP()
{
    PetscInt       i, col[3], its, N=10, num_numfac;
    PetscScalar    value[3];

    ierr = PetscOptionsGetInt(NULL, NULL, "-N", &N, NULL); CHKERRQ(ierr);

    /* Create and assemble matrix. */
    ierr = MatCreate(PETSC_COMM_WORLD, &A); CHKERRQ(ierr);
    ierr = MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, N, N); CHKERRQ(ierr);
    ierr = MatSetFromOptions(A); CHKERRQ(ierr);
    ierr = MatSetUp(A); CHKERRQ(ierr);

    ierr = MatGetOwnershipRange(A, &Istart, &Iend); CHKERRQ(ierr);

    RANK_COUT << "Istart = " << Istart << ", Iend = " << Iend << std::endl;

    value[0] = -1.0;
    value[1] = 2.0;
    value[2] = -1.0;

    for (i=Istart; i<Iend; i++) {
        col[0] = i-1;
        col[1] = i;
        col[2] = i+1;

        if (i == 0) {
            ierr = MatSetValues(A, 1, &i, 2, col+1, value+1, INSERT_VALUES); CHKERRQ(ierr);
        }
        else if (i == N-1) {
            ierr = MatSetValues(A, 1, &i, 2, col, value, INSERT_VALUES); CHKERRQ(ierr);
        }
        else {
            ierr   = MatSetValues(A, 1, &i, 3, col, value, INSERT_VALUES); CHKERRQ(ierr);
        }
    }

    ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatView(A, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

    ierr = MatSetOption(A, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE); CHKERRQ(ierr);

    /* Create vectors */
    ierr = MatCreateVecs(A, &x, &b); CHKERRQ(ierr);
    ierr = VecDuplicate(x, &u); CHKERRQ(ierr);

    /* Set exact solution; then compute right-hand-side vector. */
    ierr = VecSet(u, 1.0); CHKERRQ(ierr);
    ierr = MatMult(A, u, b); CHKERRQ(ierr);

    /* Create the linear solver and set various options. */
    ierr = KSPCreate(PETSC_COMM_WORLD, &ksp); CHKERRQ(ierr);
    ierr = KSPGetPC(ksp, &pc); CHKERRQ(ierr);
    ierr = PCSetType(pc, PCJACOBI); CHKERRQ(ierr);
    ierr = KSPSetTolerances(ksp, 1.e-5, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT); CHKERRQ(ierr);

    ierr = KSPSetOperators(ksp, A, A); CHKERRQ(ierr);
    ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);

    num_numfac = 1;
    ierr = PetscOptionsGetInt(NULL, NULL, "-num_numfac", &num_numfac, NULL); CHKERRQ(ierr);
    while (num_numfac--) {
        /* An example on how to update matrix A for repeated numerical factorization and solve. */
        RANK_COUT << "num_numfac = " << num_numfac << std::endl;

        PetscScalar one = 1.0;
        PetscInt    i   = 0;
        ierr = MatSetValues(A, 1, &i, 1, &i, &one, ADD_VALUES); CHKERRQ(ierr);

        ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
        ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

        ierr = MatView(A, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

        /* Update b */
        ierr = MatMult(A, u, b); CHKERRQ(ierr);

        /* Solve the linear system */
        ierr = KSPSolve(ksp, b, x); CHKERRQ(ierr);

        /* Check the solution and clean up */
        ierr = CheckError(x, u, ksp); CHKERRQ(ierr);
    }

    /* Free work space. */
    ierr = VecDestroy(&x); CHKERRQ(ierr);
    ierr = VecDestroy(&u); CHKERRQ(ierr);
    ierr = VecDestroy(&b); CHKERRQ(ierr);
    ierr = MatDestroy(&A); CHKERRQ(ierr);
    ierr = KSPDestroy(&ksp); CHKERRQ(ierr);

    return ierr;
}

/* ------------------ KSP example ex_7 ------------------ */
PetscErrorCode testPETScKSP::testKSP_BlockJacPC()
{
    KSP            *subksp;     /* array of local KSP contexts on this processor */
    PC             subpc;       /* PC context for subdomain */
    PetscInt       i;
    PetscInt       nlocal, first;
    PetscScalar    v1 = -1.0, v2 = 4.0;
    PetscScalar    one = 1.0, none = -1.0;
    PetscBool      isbjacobi;

    m = 4;
    ierr = PetscOptionsGetInt(NULL, NULL, "-m", &m, NULL); CHKERRQ(ierr);
    n    = m+2;

    /* -------------------------------------------------------------------
           Compute the matrix and right-hand-side vector that define
           the linear system, Ax = b.
       ------------------------------------------------------------------- */
    /* Create and assemble parallel matrix */
    ierr = MatCreate(PETSC_COMM_WORLD, &A); CHKERRQ(ierr);
    ierr = MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, m*n, m*n); CHKERRQ(ierr);
    ierr = MatSetFromOptions(A); CHKERRQ(ierr);

    ierr = MatMPIAIJSetPreallocation(A, 5, NULL, 5, NULL); CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(A, 5, NULL); CHKERRQ(ierr);
    ierr = MatGetOwnershipRange(A, &Istart, &Iend); CHKERRQ(ierr);

    //RANK_COUT << "Istart = " << Istart << ", Iend = " << Iend << std::endl;
    ierr = FormMatrixA(v1, v2, ADD_VALUES); CHKERRQ(ierr);
    ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

    ierr = MatSetOption(A, MAT_SYMMETRIC, PETSC_TRUE); CHKERRQ(ierr);

    /* Create parallel vectors */
    ierr = MatCreateVecs(A, &u, &b); CHKERRQ(ierr);
    ierr = VecDuplicate(u, &x); CHKERRQ(ierr);

    /* Set exact solution; then compute right-hand-side vector. */
    ierr = VecSet(u, one); CHKERRQ(ierr);
    ierr = MatMult(A, u, b); CHKERRQ(ierr);

    /* Create linear solver context */
    ierr = KSPCreate(PETSC_COMM_WORLD, &ksp); CHKERRQ(ierr);

    /* Set operators. Here the matrix that defines the linear system
       also serves as the preconditioning matrix. */
    ierr = KSPSetOperators(ksp, A, A); CHKERRQ(ierr);

    /* Set default preconditioner for this program to be block Jacobi.
       This choice can be overridden at runtime with the option
          -pc_type <type>
       IMPORTANT NOTE: Since the inners solves below are constructed to use
       iterative methods (such as KSPGMRES) the outer Krylov method should
       be set to use KSPFGMRES since it is the only Krylov method (plus KSPFCG)
       that allows the preconditioners to be nonlinear (that is have iterative methods
       inside them). The reason these examples work is because the number of
       iterations on the inner solves is left at the default (which is 10,000)
       and the tolerance on the inner solves is set to be a tight value of around 10^-6. */
    ierr = KSPGetPC(ksp, &pc); CHKERRQ(ierr);
    ierr = PCSetType(pc, PCBJACOBI); CHKERRQ(ierr);

    /* -------------------------------------------------------------------
                     Define the problem decomposition
       ------------------------------------------------------------------- */
    /* Call PCBJacobiSetTotalBlocks() to set individually the size of
       each block in the preconditioner.  This could also be done with
       the runtime option
           -pc_bjacobi_blocks <blocks>
       Also, see the command PCBJacobiSetLocalBlocks() to set the
       local blocks.
        Note: The default decomposition is 1 block per processor. */
    PetscInt *blks = new PetscInt[m];
    for (i=0; i<m; i++){
        blks[i] = n;
        RANK_COUT << "blks[" << i << "] = " << blks[i] << std::endl;
    }
    blks[0] = 8;
    blks[1] = 4;
    ierr = PCBJacobiSetTotalBlocks(pc, m, blks); CHKERRQ(ierr);
    delete[] blks;

    /* Set runtime options */
    ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);

    /* -------------------------------------------------------------------
                 Set the linear solvers for the subblocks
         Basic method, should be sufficient for the needs of most users.
       ------------------------------------------------------------------- */
    /* By default, the block Jacobi method uses the same solver on each
       block of the problem.  To set the same solver options on all blocks,
       use the prefix -sub before the usual PC and KSP options, e.g.,
            -sub_pc_type <pc> -sub_ksp_type <ksp> -sub_ksp_rtol 1.e-4 */
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
          Advanced method, setting different solvers for various blocks.
        Note that each block's KSP context is completely independent of
        the others, and the full range of uniprocessor KSP options is
        available for each block. The following section of code is intended
        to be a simple illustration of setting different linear solvers for
        the individual blocks.  These choices are obviously not recommended
        for solving this particular problem. */
    ierr = PetscObjectTypeCompare((PetscObject)pc, PCBJACOBI, &isbjacobi); CHKERRQ(ierr);

    RANK_COUT << "isbjacobi = " << isbjacobi << std::endl;
    if (isbjacobi) {
        /* Call KSPSetUp() to set the block Jacobi data structures (including
            creation of an internal KSP context for each block).
            Note: KSPSetUp() MUST be called before PCBJacobiGetSubKSP(). */
        ierr = KSPSetUp(ksp); CHKERRQ(ierr);

        /* Extract the array of KSP contexts for the local blocks */
        ierr = PCBJacobiGetSubKSP(pc, &nlocal, &first, &subksp); CHKERRQ(ierr);

        RANK_COUT << "nlocal = " << nlocal << ", first = " << first << std::endl;

        /* Loop over the local blocks, setting various KSP options
         for each block. */
        for (i=0; i<nlocal; i++) {
            ierr = KSPGetPC(subksp[i], &subpc); CHKERRQ(ierr);

            if (m_rank == 0) {
                if (i%2) {
                    ierr = PCSetType(subpc, PCILU); CHKERRQ(ierr);
                }
                else {
                    ierr = PCSetType(subpc, PCNONE); CHKERRQ(ierr);
                    ierr = KSPSetType(subksp[i], KSPBCGS); CHKERRQ(ierr);
                    ierr = KSPSetTolerances(subksp[i], 1.e-6, PETSC_DEFAULT, PETSC_DEFAULT,
                                            PETSC_DEFAULT); CHKERRQ(ierr);
                }
            }
            else {
                ierr = PCSetType(subpc, PCJACOBI); CHKERRQ(ierr);
                ierr = KSPSetType(subksp[i], KSPGMRES); CHKERRQ(ierr);
                ierr = KSPSetTolerances(subksp[i], 1.e-6, PETSC_DEFAULT, PETSC_DEFAULT,
                                        PETSC_DEFAULT); CHKERRQ(ierr);
            }
        }
    }

    /* -------------------------------------------------------------------
                        Solve the linear system
       ------------------------------------------------------------------- */
    ierr = KSPSolve(ksp, b, x); CHKERRQ(ierr);

    /* -------------------------------------------------------------------
                        Check solution and clean up
       ------------------------------------------------------------------- */
    ierr = CheckError(x, u, ksp); CHKERRQ(ierr);

    ierr = KSPDestroy(&ksp); CHKERRQ(ierr);
    ierr = VecDestroy(&u); CHKERRQ(ierr);
    ierr = VecDestroy(&x); CHKERRQ(ierr);
    ierr = VecDestroy(&b); CHKERRQ(ierr);
    ierr = MatDestroy(&A); CHKERRQ(ierr);

    return ierr;
}

/* ------------------ KSP example ex_8 ------------------ */
PetscErrorCode testPETScKSP::testKSP_PCASM()
{
    PetscInt       overlap = 1;             /* width of subdomain overlap */
    PetscInt       M = 2, N = 1;            /* number of subdomains in x- and y- directions */
    PetscInt       Nsub;                    /* number of subdomains */
    IS             *is, *is_local;          /* array of index sets that define the subdomains */
    PetscInt       i;
    PetscBool      flg;
    PetscBool      user_subdomains = PETSC_FALSE;
    PetscScalar    one = 1.0;
    PetscReal      e;
    PetscScalar    v1 = -1.0, v2 = 4.0;

    /* set the value of some variables at runtimes */
    m = 15;
    n = 17;
    ierr = PetscOptionsGetInt(NULL, NULL, "-m", &m, NULL); CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL); CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL, NULL, "-Mdomains", &M, NULL); CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL, NULL, "-Ndomains", &N, NULL); CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL, NULL, "-overlap", &overlap, NULL); CHKERRQ(ierr);
    ierr = PetscOptionsGetBool(NULL, NULL, "-user_set_subdomains", &user_subdomains, NULL); CHKERRQ(ierr);

    /* -------------------------------------------------------------------
           Compute the matrix and right-hand-side vector that define
           the linear system, Ax = b.
       ------------------------------------------------------------------- */
    /* Assemble the matrix for the five point stencil, YET AGAIN */
    ierr = MatCreate(PETSC_COMM_WORLD, &A); CHKERRQ(ierr);
    ierr = MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, m*n, m*n); CHKERRQ(ierr);
    ierr = MatSetFromOptions(A); CHKERRQ(ierr);
    ierr = MatSetUp(A); CHKERRQ(ierr);
    ierr = MatGetOwnershipRange(A, &Istart, &Iend); CHKERRQ(ierr);

    RANK_COUT << " Istart = " << Istart << ", Iend =" << Iend << std::endl;

    ierr = FormMatrixA(v1, v2, INSERT_VALUES); CHKERRQ(ierr);
    ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

    /* Create and set vectors */
    ierr = MatCreateVecs(A, &u, &b); CHKERRQ(ierr);
    ierr = VecDuplicate(u, &x); CHKERRQ(ierr);
    ierr = VecSet(u, one); CHKERRQ(ierr);
    ierr = MatMult(A, u, b); CHKERRQ(ierr);

    /* Create linear solver context */
    ierr = KSPCreate(PETSC_COMM_WORLD, &ksp); CHKERRQ(ierr);

    /* Set operators. Here the matrix that defines the linear system
       also serves as the preconditioning matrix. */
    ierr = KSPSetOperators(ksp, A, A); CHKERRQ(ierr);

    /* Set the default preconditioner for this program to be ASM */
    ierr = KSPGetPC(ksp, &pc); CHKERRQ(ierr);
    ierr = PCSetType(pc, PCASM); CHKERRQ(ierr);

    /* -------------------------------------------------------------------
                    Define the problem decomposition
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      Basic method, should be sufficient for the needs of many users.
       Set the overlap, using the default PETSc decomposition via
           PCASMSetOverlap(pc,overlap);
       Could instead use the option -pc_asm_overlap <ovl>
       Set the total number of blocks via -pc_asm_blocks <blks>
       Note:  The ASM default is to use 1 block per processor.  To
       experiment on a single processor with various overlaps, you
       must specify use of multiple blocks!
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         More advanced method, setting user-defined subdomains
       Firstly, create index sets that define the subdomains.  The utility
       routine PCASMCreateSubdomains2D() is a simple example (that currently
       supports 1 processor only!).  More generally, the user should write
       a custom routine for a particular problem geometry.
       Then call either PCASMSetLocalSubdomains() or PCASMSetTotalSubdomains()
       to set the subdomains for the ASM preconditioner. */

    if (!user_subdomains) {
        /* basic version */
        ierr = PCASMSetOverlap(pc, overlap); CHKERRQ(ierr);
    } else{
        /* advanced version */
        if (m_size != 1){
            SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "PCASMCreateSubdomains2D() is "
                                                     "currently a uniprocessor routine only!");
        }
        ierr = PCASMCreateSubdomains2D(m, n, M, N, 1, overlap,
                                       &Nsub, &is, &is_local); CHKERRQ(ierr);
        ierr = PCASMSetLocalSubdomains(pc, Nsub, is, is_local); CHKERRQ(ierr);
        flg  = PETSC_FALSE;
        ierr = PetscOptionsGetBool(NULL, NULL, "-subdomain_view", &flg, NULL); CHKERRQ(ierr);

        if (flg) {
            ierr = PetscPrintf(PETSC_COMM_SELF, "Nmesh points: %D x %D; subdomain partition: %D x %D; overlap: %D; Nsub: %D\n",
                               m, n, M, N, overlap, Nsub); CHKERRQ(ierr);
            ierr = PetscPrintf(PETSC_COMM_SELF, "IS:\n"); CHKERRQ(ierr);

            for (i=0; i<Nsub; i++) {
                ierr = PetscPrintf(PETSC_COMM_SELF, "  IS[%D]\n", i); CHKERRQ(ierr);
                ierr = ISView(is[i], PETSC_VIEWER_STDOUT_SELF); CHKERRQ(ierr);
            }
            ierr = PetscPrintf(PETSC_COMM_SELF, "IS_local:\n"); CHKERRQ(ierr);

            for (i=0; i<Nsub; i++) {
                ierr = PetscPrintf(PETSC_COMM_SELF, "  IS_local[%D]\n", i); CHKERRQ(ierr);
                ierr = ISView(is_local[i], PETSC_VIEWER_STDOUT_SELF); CHKERRQ(ierr);
            }
        }
    }

    /* -------------------------------------------------------------------
                  Set the linear solvers for the subblocks
       -------------------------------------------------------------------
         Basic method, should be sufficient for the needs of most users.
            By default, the ASM preconditioner uses the same solver on each
            block of the problem.  To set the same solver options on all blocks,
            use the prefix -sub before the usual PC and KSP options, e.g.,
            -sub_pc_type <pc> -sub_ksp_type <ksp> -sub_ksp_rtol 1.e-4
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Advanced method, setting different solvers for various blocks.
        Note that each block's KSP context is completely independent of
        the others, and the full range of uniprocessor KSP options is
        available for each block.
         - Use PCASMGetSubKSP() to extract the array of KSP contexts for
            the local blocks.
        - See ex7.c for a simple example of setting different linear solvers
          for the individual blocks for the block Jacobi method (which is
          equivalent to the ASM method with zero overlap). */

    flg  = PETSC_FALSE;
    ierr = PetscOptionsGetBool(NULL, NULL, "-user_set_subdomain_solvers",
                               &flg, NULL); CHKERRQ(ierr);
    if (flg) {
        KSP       *subksp;        /* array of KSP contexts for local subblocks */
        PetscInt  nlocal,first;   /* number of local subblocks, first local subblock */
        PC        subpc;          /* PC context for subblock */
        PetscBool isasm;

        ierr = PetscPrintf(PETSC_COMM_WORLD, "User explicitly sets "
                                             "subdomain solvers.\n"); CHKERRQ(ierr);

        /* Set runtime options */
        ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);

        /* Flag an error if PCTYPE is changed from the runtime options */
        ierr = PetscObjectTypeCompare((PetscObject)pc, PCASM, &isasm); CHKERRQ(ierr);
        if (!isasm){
            SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP,"Cannot Change the PCTYPE "
                                                    "when manually changing the subdomain solver settings");
        }

        /* Call KSPSetUp() to set the block Jacobi data structures (including
         creation of an internal KSP context for each block).
         Note: KSPSetUp() MUST be called before PCASMGetSubKSP(). */
        ierr = KSPSetUp(ksp); CHKERRQ(ierr);

        /* Extract the array of KSP contexts for the local blocks */
        ierr = PCASMGetSubKSP(pc, &nlocal, &first, &subksp); CHKERRQ(ierr);

        /* Loop over the local blocks, setting various KSP options
         for each block. */
        for (i=0; i<nlocal; i++){
            ierr = KSPGetPC(subksp[i], &subpc); CHKERRQ(ierr);
            ierr = PCSetType(subpc, PCILU); CHKERRQ(ierr);
            ierr = KSPSetType(subksp[i], KSPGMRES); CHKERRQ(ierr);
            ierr = KSPSetTolerances(subksp[i], 1.e-7, PETSC_DEFAULT, PETSC_DEFAULT,
                                    PETSC_DEFAULT); CHKERRQ(ierr);
        }
    } else {
        /* Set runtime options */
        ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);
    }

    /* -------------------------------------------------------------------
                        Solve the linear system
       ------------------------------------------------------------------- */
    ierr = KSPSolve(ksp, b, x); CHKERRQ(ierr);

    /* -------------------------------------------------------------------
                        Compare result to the exact solution
       ------------------------------------------------------------------- */
    ierr = VecAXPY(x, -1.0, u); CHKERRQ(ierr);
    ierr = VecNorm(x, NORM_INFINITY, &e); CHKERRQ(ierr);

    ierr = KSPGetIterationNumber(ksp, &its); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Norm of error %g iterations %D\n",
                       (double)e, its); CHKERRQ(ierr);

    flg  = PETSC_FALSE;
    ierr = PetscOptionsGetBool(NULL, NULL, "-print_error", &flg, NULL); CHKERRQ(ierr);
    if (flg) {
        ierr = PetscPrintf(PETSC_COMM_WORLD, "Infinity norm of the error: %g\n",
                           (double) e); CHKERRQ(ierr);
    }

    /* Free work space.  All PETSc objects should be destroyed when they
       are no longer needed. */

    if (user_subdomains) {
        for (i=0; i<Nsub; i++) {
            ierr = ISDestroy(&is[i]); CHKERRQ(ierr);
            ierr = ISDestroy(&is_local[i]); CHKERRQ(ierr);
        }
        ierr = PetscFree(is); CHKERRQ(ierr);
        ierr = PetscFree(is_local); CHKERRQ(ierr);
    }
    ierr = KSPDestroy(&ksp); CHKERRQ(ierr);
    ierr = VecDestroy(&u); CHKERRQ(ierr);
    ierr = VecDestroy(&x); CHKERRQ(ierr);
    ierr = VecDestroy(&b); CHKERRQ(ierr);
    ierr = MatDestroy(&A); CHKERRQ(ierr);

    return ierr;
}

/* ------------------ KSP example ex_9 ------------------ */
PetscErrorCode testPETScKSP::testKSP_SolDiffLinSys()
{
    Vec            x1, b1;         /* solution and RHS vectors for systems #2 */
    Mat            A1;              /* matrices for systems #2 */
    KSP            ksp1;           /* KSP contexts for systems #1*/
    PetscInt       ntimes = 3;     /* number of times to solve the linear systems */
    PetscInt       low, high;
    PetscInt       ldim, iglobal;
    PetscInt       Istart1, Iend1;
    PetscInt       Ii, J, i, j;
    PetscBool      unsym = PETSC_TRUE;
    PetscScalar    v;

    m = 3;
    n = 2;
    ierr = PetscOptionsGetInt(NULL, NULL, "-m", &m, NULL); CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL, NULL, "-t", &ntimes, NULL); CHKERRQ(ierr);
    ierr = PetscOptionsGetBool(NULL, NULL, "-unsym", &unsym, NULL); CHKERRQ(ierr);
    n    = 2*m_size;

    /* - - - - - - - - - - - - Stage 0: - - - - - - - - - - - - - -
                          Preliminary Setup
       Create data structures for first linear system.*/
    ierr = MatCreate(PETSC_COMM_WORLD, &A); CHKERRQ(ierr);
    ierr = MatSetSizes(A,PETSC_DECIDE, PETSC_DECIDE, m*n, m*n); CHKERRQ(ierr);
    ierr = MatSetFromOptions(A); CHKERRQ(ierr);
    ierr = MatSetUp(A); CHKERRQ(ierr);
    ierr = MatGetOwnershipRange(A, &Istart, &Iend); CHKERRQ(ierr);

    RANK_COUT << "Istart = " << Istart << ", Iend = " << Iend << std::endl;

    ierr = VecCreate(PETSC_COMM_WORLD, &u); CHKERRQ(ierr);
    ierr = VecSetSizes(u, PETSC_DECIDE, m*n); CHKERRQ(ierr);
    ierr = VecSetFromOptions(u); CHKERRQ(ierr);
    ierr = VecDuplicate(u, &b); CHKERRQ(ierr);
    ierr = VecDuplicate(u, &x); CHKERRQ(ierr);

    /* Create first linear solver context.
       Set runtime options (e.g., -pc_type <type>).
       Note that the first linear system uses the default option
       names, while the second linear system uses a different
       options prefix. */
    ierr = KSPCreate(PETSC_COMM_WORLD, &ksp); CHKERRQ(ierr);
    ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);


    /* Create data structures for second linear system. */
    ierr = MatCreate(PETSC_COMM_WORLD, &A1); CHKERRQ(ierr);
    ierr = MatSetSizes(A1, PETSC_DECIDE, PETSC_DECIDE, m*n, m*n); CHKERRQ(ierr);
    ierr = MatSetFromOptions(A1); CHKERRQ(ierr);
    ierr = MatSetUp(A1); CHKERRQ(ierr);

    ierr = MatGetOwnershipRange(A1, &Istart1, &Iend1); CHKERRQ(ierr);
    ierr = VecDuplicate(u, &b1); CHKERRQ(ierr);
    ierr = VecDuplicate(u, &x1); CHKERRQ(ierr);

    /* Create second linear solver context */
    ierr = KSPCreate(PETSC_COMM_WORLD, &ksp1); CHKERRQ(ierr);

    /* Set different options prefix for second linear system.
       Set runtime options (e.g., -s2_pc_type <type>) */
    ierr = KSPAppendOptionsPrefix(ksp1, "s2_");CHKERRQ(ierr);
    ierr = KSPSetFromOptions(ksp1); CHKERRQ(ierr);

    /* Assemble exact solution vector in parallel.  Note that each
       processor needs to set only its local part of the vector. */
    ierr = VecGetLocalSize(u, &ldim); CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(u, &low, &high);CHKERRQ(ierr);

    for (i=0; i<ldim; i++) {
        iglobal = i + low;
        v       = (PetscScalar)(i + 100*m_rank);
        ierr    = VecSetValues(u, 1, &iglobal, &v, ADD_VALUES); CHKERRQ(ierr);
    }
    ierr = VecAssemblyBegin(u); CHKERRQ(ierr);
    ierr = VecAssemblyEnd(u); CHKERRQ(ierr);

    /* --------------------------------------------------------------
                          Linear solver loop:
        Solve 2 different linear systems several times in succession
       -------------------------------------------------------------- */

    for (PetscInt t=0; t<ntimes; t++) {

        /* - - - - - - - - - - - - Stage 1: - - - - - - - - - - - - - -
                   Assemble and solve first linear system
         Initialize all matrix entries to zero.  MatZeroEntries() retains
         the nonzero structure of the matrix for sparse formats. */
        if (t > 0){
            ierr = MatZeroEntries(A); CHKERRQ(ierr);
        }

        /* Set matrix entries in parallel.  Also, log the number of flops
            for computing matrix entries.
            - Each processor needs to insert only elements that it owns
            locally (but any non-local elements will be sent to the
                appropriate processor during matrix assembly).
            - Always specify global row and columns of matrix entries. */
        PetscScalar v1 = -1.0, v2 = 4.0;
        ierr = FormMatrixA(A, v1, v2, ADD_VALUES); CHKERRQ(ierr);

        /* Make matrix nonsymmetric */
        if (unsym) {
            for (Ii=Istart; Ii<Iend; Ii++) {
                v = -1.0*(t+0.5);
                i = Ii/n;
                if (i>0){
                    J = Ii - n;
                    ierr = MatSetValues(A, 1, &Ii, 1, &J, &v, ADD_VALUES); CHKERRQ(ierr);
                }
            }
        }

        ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
        ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

        /* Indicate same nonzero structure of successive linear system matrices */
        ierr = MatSetOption(A, MAT_NEW_NONZERO_LOCATIONS, PETSC_TRUE); CHKERRQ(ierr);

        /* Compute right-hand-side vector */
        ierr = MatMult(A, u, b); CHKERRQ(ierr);

        /* Set operators. Here the matrix that defines the linear system
           also serves as the preconditioning matrix. */
        ierr = KSPSetOperators(ksp, A, A); CHKERRQ(ierr);

        /* Use the previous solution of linear system #1 as the initial
            guess for the next solve of linear system #1.  The user MUST
            call KSPSetInitialGuessNonzero() in indicate use of an initial
            guess vector; otherwise, an initial guess of zero is used. */
        if (t>0) {
            ierr = KSPSetInitialGuessNonzero(ksp, PETSC_TRUE); CHKERRQ(ierr);
        }

        ierr = KSPSetUp(ksp); CHKERRQ(ierr);
        ierr = KSPSolve(ksp, b, x); CHKERRQ(ierr);
        //ierr = MatView(A, PETSC_VIEWER_STDERR_WORLD); CHKERRQ(ierr);

        ierr = PetscPrintf(PETSC_COMM_WORLD, "----------- % d -----------\n", t); CHKERRQ(ierr);
        //ierr = KSPView(ksp, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

        /* Check error of solution to first linear system */
        ierr = CheckError(x, u, ksp); CHKERRQ(ierr);

        /* - - - - - - - - - - - - Stage 2: - - - - - - - - - - - - - -
                   Assemble and solve second linear system
         Conclude profiling stage #1; begin profiling stage #2 */

        /* Initialize all matrix entries to zero */
        if (t > 0){
            ierr = MatZeroEntries(A1); CHKERRQ(ierr);
        }

        /* Assemble matrix in parallel. Also, log the number of flops
            for computing matrix entries.
             - To illustrate the features of parallel matrix assembly, we
               intentionally set the values differently from the way in
               which the matrix is distributed across the processors.  Each
               entry that is not owned locally will be sent to the appropriate
               processor during MatAssemblyBegin() and MatAssemblyEnd().
            - For best efficiency the user should strive to set as many
               entries locally as possible. */
        for (i=0; i<m; i++) {
            for (j=2*m_rank; j<2*m_rank+2; j++) {
                Ii = j + n*i;
                v = -1.0;
                if (i>0){
                    J = Ii - n;
                    ierr = MatSetValues(A1, 1, &Ii, 1, &J, &v, ADD_VALUES); CHKERRQ(ierr);
                }

                if (i<m-1){
                    J = Ii + n;
                    ierr = MatSetValues(A1, 1, &Ii, 1, &J, &v, ADD_VALUES); CHKERRQ(ierr);
                }

                if (j>0){
                    J = Ii - 1;
                    ierr = MatSetValues(A1, 1, &Ii, 1, &J, &v, ADD_VALUES); CHKERRQ(ierr);
                }

                if (j<n-1){
                    J = Ii + 1;
                    ierr = MatSetValues(A1, 1, &Ii, 1, &J, &v, ADD_VALUES); CHKERRQ(ierr);
                }

                v = 6.0 + t*0.5;
                ierr = MatSetValues(A1, 1, &Ii, 1, &Ii, &v, ADD_VALUES); CHKERRQ(ierr);
            }
        }

        if (unsym) {
            for (Ii=Istart1; Ii<Iend1; Ii++){
                /* Make matrix nonsymmetric */
                v = -1.0*(t+0.5);
                i = Ii/n;
                if (i>0){
                    J = Ii - n;
                    ierr = MatSetValues(A1, 1, &Ii, 1, &J, &v, ADD_VALUES); CHKERRQ(ierr);
                }
            }
        }
        ierr = MatAssemblyBegin(A1, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
        ierr = MatAssemblyEnd(A1, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

        /* Indicate same nonzero structure of successive linear system matrices */
        ierr = MatSetOption(A1, MAT_NEW_NONZERO_LOCATIONS, PETSC_FALSE); CHKERRQ(ierr);

        /* Compute right-hand-side vector */
        ierr = MatMult(A1, u, b1); CHKERRQ(ierr);

        /* Set operators. Here the matrix that defines the linear system
            also serves as the preconditioning matrix.  Indicate same nonzero
            structure of successive preconditioner matrices by setting flag
            SAME_NONZERO_PATTERN. */
        ierr = KSPSetOperators(ksp1, A1, A1); CHKERRQ(ierr);

        /* Solve the second linear system */
        ierr = KSPSetUp(ksp1); CHKERRQ(ierr);
        ierr = KSPSolve(ksp1, b1, x1); CHKERRQ(ierr);
        //ierr = MatView(A1, PETSC_VIEWER_STDERR_WORLD); CHKERRQ(ierr);

        /* Check error of solution to second linear system */
        ierr = CheckError(x1, u, ksp1); CHKERRQ(ierr);
    }

    /* Free work space.  All PETSc objects should be destroyed when they
       are no longer needed. */
    ierr = KSPDestroy(&ksp); CHKERRQ(ierr);
    ierr = KSPDestroy(&ksp1); CHKERRQ(ierr);
    ierr = VecDestroy(&x); CHKERRQ(ierr);
    ierr = VecDestroy(&x1); CHKERRQ(ierr);
    ierr = VecDestroy(&b); CHKERRQ(ierr);
    ierr = VecDestroy(&b1); CHKERRQ(ierr);
    ierr = MatDestroy(&A); CHKERRQ(ierr);
    ierr = MatDestroy(&A1); CHKERRQ(ierr);
    ierr = VecDestroy(&u); CHKERRQ(ierr);

    return ierr;
}

/* ------------------ KSP example ex_10 ------------------ */
PetscErrorCode testPETScKSP::testKSP_Preloading()
{
    Vec               b1;
    char              file[2][PETSC_MAX_PATH_LEN];
    PetscViewer       viewer;      /* viewer */
    PetscBool         flg, same;
    PetscBool         preload=PETSC_FALSE, trans=PETSC_FALSE;
    RHSType           rhstype = RHS_FILE;
    PetscInt          j, len, idx, n1, n2;
    const PetscScalar *val;

    /* Determine files from which we read the two linear systems
       (matrix and right-hand-side vector). */
    ierr = PetscOptionsGetBool(NULL, NULL, "-trans", &trans, &flg); CHKERRQ(ierr);
    ierr = PetscOptionsGetString(NULL, NULL,"-f", file[0], sizeof(file[0]), &flg); CHKERRQ(ierr);

    RANK_COUT << "trans = " << trans << ", flg = " << flg << std::endl;

    if (flg) {
        ierr    = PetscStrcpy(file[1], file[0]); CHKERRQ(ierr);
        preload = PETSC_FALSE;
    } else {
        ierr = PetscOptionsGetString(NULL, NULL, "-f0", file[0],
                sizeof(file[0]), &flg); CHKERRQ(ierr);

        if (!flg){
            SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "Must indicate binary"
                    " file with the -f0 or -f option");
        }

        ierr = PetscOptionsGetString(NULL, NULL, "-f1", file[1], sizeof(file[1]),
                &flg); CHKERRQ(ierr);
        if (!flg){
            /* don't bother with second system */
            preload = PETSC_FALSE;
        }
    }
    ierr = PetscOptionsGetEnum(NULL, NULL, "-rhs", RHSTypes, (PetscEnum*)&rhstype,
                               NULL); CHKERRQ(ierr);

    /* To use preloading, one usually has code like the following:
      PetscPreLoadBegin(preload,"first stage);
        lines of code
      PetscPreLoadStage("second stage");
        lines of code
      PetscPreLoadEnd();
      The two macro PetscPreLoadBegin() and PetscPreLoadEnd() implicitly form a
      loop with maximal two iterations, depending whether preloading is turned on or
      not. If it is, either through the preload arg of PetscPreLoadBegin or through
      -preload command line, the trip count is 2, otherwise it is 1. One can use the
      predefined variable PetscPreLoadIt within the loop body to get the current
      iteration number, which is 0 or 1. If preload is turned on, the runtime doesn't
      do profiling for the first iteration, but it will do profiling for the second
      iteration instead.
      One can solve a small system in the first iteration and a large system in
      the second iteration. This process preloads the instructions with the small
      system so that more accurate performance monitoring (via -log_view) can be done
      with the large one (that actually is the system of interest).
      But in this example, we turned off preloading and duplicated the code for
      the large system. In general, it is a bad practice and one should not duplicate
      code. We do that because we want to show profiling stages for both the small
      system and the large system. */

    PetscPreLoadBegin(preload, "Load System 0");

    /*=========================
        solve a small system
      =========================*/

    /* open binary file. Note that we use FILE_MODE_READ to indicate reading from this file */
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, file[0], FILE_MODE_READ,
                                 &viewer); CHKERRQ(ierr);

    /* load the matrix and vector; then destroy the viewer */
    ierr = MatCreate(PETSC_COMM_WORLD, &A); CHKERRQ(ierr);
    ierr = MatSetFromOptions(A); CHKERRQ(ierr);
    ierr = MatLoad(A, viewer); CHKERRQ(ierr);
    switch (rhstype) {
    case RHS_FILE:
        /* Vectors in the file might a different size than the matrix so we need a
         * Vec whose size hasn't been set yet.  It'll get fixed below.  Otherwise we
         * can create the correct size Vec. */
        ierr = VecCreate(PETSC_COMM_WORLD, &b); CHKERRQ(ierr);
        ierr = VecLoad(b, viewer); CHKERRQ(ierr);
        break;
    case RHS_ONE:
        ierr = MatCreateVecs(A, &b, NULL); CHKERRQ(ierr);
        ierr = VecSet(b, 1.0); CHKERRQ(ierr);
        break;
    case RHS_RANDOM:
        ierr = MatCreateVecs(A, &b, NULL); CHKERRQ(ierr);
        ierr = VecSetRandom(b, NULL); CHKERRQ(ierr);
        break;
    }
    ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

    /* if the loaded matrix is larger than the vector (due to being padded
       to match the block size of the system), then create a new padded vector */
    ierr = MatGetLocalSize(A, NULL, &n1); CHKERRQ(ierr);
    ierr = VecGetLocalSize(b, &n2); CHKERRQ(ierr);
    same = (n1 == n2)? PETSC_TRUE : PETSC_FALSE;
    ierr = MPIU_Allreduce(MPI_IN_PLACE, &same, 1, MPIU_BOOL, MPI_LAND,
                          PETSC_COMM_WORLD); CHKERRMPI(ierr);

    if (!same) {
        /* create a new vector b by padding the old one */
        ierr = VecCreate(PETSC_COMM_WORLD, &b1); CHKERRQ(ierr);
        ierr = VecSetSizes(b1, n1, PETSC_DECIDE); CHKERRQ(ierr);
        ierr = VecSetFromOptions(b1); CHKERRQ(ierr);

        ierr = VecGetOwnershipRange(b, &Istart, NULL); CHKERRQ(ierr);
        ierr = VecGetLocalSize(b, &len); CHKERRQ(ierr);
        ierr = VecGetArrayRead(b, &val); CHKERRQ(ierr);

        for (j=0; j<len; j++) {
            idx = Istart+j;
            ierr = VecSetValues(b1, 1, &idx, val+j, INSERT_VALUES); CHKERRQ(ierr);
        }

        ierr = VecRestoreArrayRead(b, &val); CHKERRQ(ierr);
        ierr = VecDestroy(&b); CHKERRQ(ierr);
        ierr = VecAssemblyBegin(b1); CHKERRQ(ierr);
        ierr = VecAssemblyEnd(b1); CHKERRQ(ierr);
        b    = b1;
    }
    ierr = VecDuplicate(b, &x);CHKERRQ(ierr);

    PetscPreLoadStage("KSPSetUp 0");
    ierr = KSPCreate(PETSC_COMM_WORLD, &ksp); CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp, A, A); CHKERRQ(ierr);
    ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);

    /* Here we explicitly call KSPSetUp() and KSPSetUpOnBlocks() to
      enable more precise profiling of setting up the preconditioner.
      These calls are optional, since both will be called within
      KSPSolve() if they haven't been called already. */
    ierr = KSPSetUp(ksp); CHKERRQ(ierr);
    ierr = KSPSetUpOnBlocks(ksp); CHKERRQ(ierr);

    PetscPreLoadStage("KSPSolve 0");
    if (trans){
        ierr = KSPSolveTranspose(ksp, b, x); CHKERRQ(ierr);
    }
    else {
        ierr = KSPSolve(ksp, b, x); CHKERRQ(ierr);
    }

    ierr = KSPGetTotalIterations(ksp, &its); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Number of iterations = %d\n", its); CHKERRQ(ierr);

    ierr = KSPGetResidualNorm(ksp, &norm); CHKERRQ(ierr);
    if (norm < 1.e-12) {
        ierr = PetscPrintf(PETSC_COMM_WORLD, "Residual norm < 1.e-12\n"); CHKERRQ(ierr);
    }
    else {
        ierr = PetscPrintf(PETSC_COMM_WORLD, "Residual norm %e\n", (double)norm); CHKERRQ(ierr);
    }

    ierr = KSPDestroy(&ksp); CHKERRQ(ierr);
    ierr = MatDestroy(&A); CHKERRQ(ierr);
    ierr = VecDestroy(&x); CHKERRQ(ierr);
    ierr = VecDestroy(&b); CHKERRQ(ierr);

    /*=========================
      solve a large system
      =========================*/
    /* the code is duplicated. Bad practice. See comments above */
    PetscPreLoadStage("Load System 1");
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, file[1], FILE_MODE_READ, &viewer); CHKERRQ(ierr);

    /* load the matrix and vector; then destroy the viewer */
    ierr = MatCreate(PETSC_COMM_WORLD, &A); CHKERRQ(ierr);
    ierr = MatSetFromOptions(A); CHKERRQ(ierr);
    ierr = MatLoad(A, viewer); CHKERRQ(ierr);

    switch (rhstype) {
    case RHS_FILE:
        /* Vectors in the file might a different size than the matrix so we need a
         * Vec whose size hasn't been set yet.  It'll get fixed below.  Otherwise we
         * can create the correct size Vec. */
        ierr = VecCreate(PETSC_COMM_WORLD, &b); CHKERRQ(ierr);
        ierr = VecLoad(b, viewer); CHKERRQ(ierr);
        break;
    case RHS_ONE:
        ierr = MatCreateVecs(A, &b, NULL); CHKERRQ(ierr);
        ierr = VecSet(b, 1.0); CHKERRQ(ierr);
        break;
    case RHS_RANDOM:
        ierr = MatCreateVecs(A, &b, NULL); CHKERRQ(ierr);
        ierr = VecSetRandom(b, NULL); CHKERRQ(ierr);
        break;
    }
    ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

    ierr = MatGetLocalSize(A, NULL, &n1); CHKERRQ(ierr);
    ierr = VecGetLocalSize(b, &n2); CHKERRQ(ierr);
    same = (n1 == n2)? PETSC_TRUE : PETSC_FALSE;
    ierr = MPIU_Allreduce(MPI_IN_PLACE, &same, 1, MPIU_BOOL, MPI_LAND,
                          PETSC_COMM_WORLD); CHKERRMPI(ierr);

    if (!same) {
        /* create a new vector b by padding the old one */
        ierr = VecCreate(PETSC_COMM_WORLD, &b1); CHKERRQ(ierr);
        ierr = VecSetSizes(b1, n1, PETSC_DECIDE); CHKERRQ(ierr);
        ierr = VecSetFromOptions(b1); CHKERRQ(ierr);

        ierr = VecGetOwnershipRange(b, &Istart, NULL); CHKERRQ(ierr);
        ierr = VecGetLocalSize(b, &len); CHKERRQ(ierr);
        ierr = VecGetArrayRead(b, &val); CHKERRQ(ierr);

        for (j=0; j<len; j++) {
            idx = Istart+j;
            ierr = VecSetValues(b1, 1, &idx, val+j, INSERT_VALUES); CHKERRQ(ierr);
        }
        ierr = VecRestoreArrayRead(b, &val); CHKERRQ(ierr);
        ierr = VecDestroy(&b); CHKERRQ(ierr);

        ierr = VecAssemblyBegin(b1); CHKERRQ(ierr);
        ierr = VecAssemblyEnd(b1); CHKERRQ(ierr);
        b    = b1;
    }
    ierr = VecDuplicate(b, &x); CHKERRQ(ierr);

    PetscPreLoadStage("KSPSetUp 1");
    ierr = KSPCreate(PETSC_COMM_WORLD, &ksp); CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp, A, A); CHKERRQ(ierr);
    ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);

    /* Here we explicitly call KSPSetUp() and KSPSetUpOnBlocks() to
       enable more precise profiling of setting up the preconditioner.
       These calls are optional, since both will be called within
       KSPSolve() if they haven't been called already. */
    ierr = KSPSetUp(ksp); CHKERRQ(ierr);
    ierr = KSPSetUpOnBlocks(ksp); CHKERRQ(ierr);

    PetscPreLoadStage("KSPSolve 1");
    if (trans){
        ierr = KSPSolveTranspose(ksp, b, x); CHKERRQ(ierr);
    }
    else{
        ierr = KSPSolve(ksp, b, x); CHKERRQ(ierr);
    }

    ierr = KSPGetTotalIterations(ksp, &its); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Number of iterations = %d\n", its); CHKERRQ(ierr);

    ierr = KSPGetResidualNorm(ksp, &norm); CHKERRQ(ierr);
    if (norm < 1.e-12) {
        ierr = PetscPrintf(PETSC_COMM_WORLD, "Residual norm < 1.e-12\n"); CHKERRQ(ierr);
    }
    else {
        ierr = PetscPrintf(PETSC_COMM_WORLD, "Residual norm %e\n", (double)norm); CHKERRQ(ierr);
    }

    ierr = KSPDestroy(&ksp); CHKERRQ(ierr);
    ierr = MatDestroy(&A); CHKERRQ(ierr);
    ierr = VecDestroy(&x); CHKERRQ(ierr);
    ierr = VecDestroy(&b); CHKERRQ(ierr);
    PetscPreLoadEnd();

    return ierr;
}

/* ------------------ KSP example ex_12 ------------------ */
PetscErrorCode testPETScKSP::testKSP_RegistNewPC()
{
    PetscScalar    one = 1.0;

    m = 8, n = 7;
    ierr = PetscOptionsGetInt(NULL, NULL, "-m", &m, NULL); CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL); CHKERRQ(ierr);

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
           Compute the matrix and right-hand-side vector that define
           the linear system, Ax = b.
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    /* Create parallel matrix, specifying only its global dimensions.
       When using MatCreate(), the matrix format can be specified at
       runtime. Also, the parallel partitioning of the matrix can be
       determined by PETSc at runtime. */
    ierr = MatCreate(PETSC_COMM_WORLD, &A); CHKERRQ(ierr);
    ierr = MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, m*n, m*n); CHKERRQ(ierr);
    ierr = MatSetFromOptions(A); CHKERRQ(ierr);
    ierr = MatSetUp(A); CHKERRQ(ierr);

    /* Currently, all PETSc parallel matrix formats are partitioned by
       contiguous chunks of rows across the processors.  Determine which
       rows of the matrix are locally owned. */
    ierr = MatGetOwnershipRange(A, &Istart, &Iend); CHKERRQ(ierr);

    /* Set matrix elements for the 2-D, five-point stencil in parallel.
        - Each processor needs to insert only elements that it owns
          locally (but any non-local elements will be sent to the
          appropriate processor during matrix assembly).
        - Always specify global rows and columns of matrix entries. */
    PetscScalar v1 = -1.0;
    PetscScalar v2 = 4.0;
    ierr = FormMatrixA(v1, v2, INSERT_VALUES); CHKERRQ(ierr);
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

    /* Create parallel vectors.
          - When using VecCreate(), VecSetSizes() and VecSetFromOptions(),
        we specify only the vector's global
            dimension; the parallel partitioning is determined at runtime.
          - When solving a linear system, the vectors and matrices MUST
            be partitioned accordingly.  PETSc automatically generates
            appropriately partitioned matrices and vectors when MatCreate()
            and VecCreate() are used with the same communicator.
          - Note: We form 1 vector from scratch and then duplicate as needed. */
    ierr = VecCreate(PETSC_COMM_WORLD, &u); CHKERRQ(ierr);
    ierr = VecSetSizes(u, PETSC_DECIDE, m*n); CHKERRQ(ierr);
    ierr = VecSetFromOptions(u); CHKERRQ(ierr);

    ierr = VecDuplicate(u, &b); CHKERRQ(ierr);
    ierr = VecDuplicate(b, &x); CHKERRQ(ierr);

    /* Set exact solution; then compute right-hand-side vector.
       Use an exact solution of a vector with all elements of 1.0; */
    ierr = VecSet(u, one); CHKERRQ(ierr);
    ierr = MatMult(A, u, b); CHKERRQ(ierr);

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                  Create the linear solver and set various options
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    /* Create linear solver context */
    ierr = KSPCreate(PETSC_COMM_WORLD, &ksp); CHKERRQ(ierr);

    /* Set operators. Here the matrix that defines the linear system
       also serves as the preconditioning matrix. */
    ierr = KSPSetOperators(ksp, A, A); CHKERRQ(ierr);

    /* First register a new PC type with the command PCRegister() */
    ierr = PCRegister("ourjacobi", PCCreate_Jacobi); CHKERRQ(ierr);

    /* Set the PC type to be the new method */
    ierr = KSPGetPC(ksp, &pc); CHKERRQ(ierr);
    ierr = PCSetType(pc, "ourjacobi"); CHKERRQ(ierr);

    /* Set runtime options, e.g.,
          -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
      These options will override those specified above as long as
      KSPSetFromOptions() is called _after_ any other customization
      routines. */
    ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                        Solve the linear system
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    ierr = KSPSolve(ksp, b, x); CHKERRQ(ierr);

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                        Check solution and clean up
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    /* Check the error */
    ierr = CheckError(x, u, ksp); CHKERRQ(ierr);

    /* Free work space.  All PETSc objects should be destroyed when they
       are no longer needed. */
    ierr = KSPDestroy(&ksp); CHKERRQ(ierr);
    ierr = VecDestroy(&u); CHKERRQ(ierr);
    ierr = VecDestroy(&x); CHKERRQ(ierr);
    ierr = VecDestroy(&b); CHKERRQ(ierr);
    ierr = MatDestroy(&A); CHKERRQ(ierr);

    return ierr;
}

PetscErrorCode testPETScKSP::CheckError(Vec xx, Vec uu, KSP kspp)
{
    /*
      Input Parameter
      xx   -- approx solution
      uu   -- exact solution
      kspp -- linear solver context
    */
    ierr = VecAXPY(xx, -1.0, uu); CHKERRQ(ierr);
    ierr = VecNorm(xx, NORM_2, &norm); CHKERRQ(ierr);
    ierr = KSPGetIterationNumber(kspp, &its); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Norm of error %g iterations %D\n",
                       (double)norm, its); CHKERRQ(ierr);

    return ierr;
}

PetscErrorCode testPETScKSP::FormMatrixA(PetscScalar v1, PetscScalar v2, InsertMode model)
{
    PetscScalar v;
    PetscInt    i, j;
    PetscInt    J;
    for (PetscInt Ii=Istart; Ii<Iend; Ii++) {
        v = v1;
        i = Ii/n;
        j = Ii - i*n;
        if (i>0){
            J = Ii - n;
            ierr = MatSetValues(A, 1, &Ii, 1, &J, &v, model); CHKERRQ(ierr);
        }

        if (i<m-1){
            J = Ii + n;
            ierr = MatSetValues(A, 1, &Ii, 1, &J, &v, model); CHKERRQ(ierr);
        }

        if (j>0){
            J = Ii - 1;
            ierr = MatSetValues(A, 1, &Ii, 1, &J, &v, model); CHKERRQ(ierr);
        }

        if (j<n-1){
            J = Ii + 1;
            ierr = MatSetValues(A, 1, &Ii, 1, &J, &v, model); CHKERRQ(ierr);
        }
        v = v2;
        ierr = MatSetValues(A, 1, &Ii, 1, &Ii, &v, model); CHKERRQ(ierr);
    }

    return ierr;
}

PetscErrorCode testPETScKSP::FormMatrixA(Mat &C, PetscScalar v1, PetscScalar v2,
                                         InsertMode mode)
{
    PetscScalar v;
    PetscInt    i, j;
    PetscInt    J;
    for (PetscInt Ii=Istart; Ii<Iend; Ii++) {
        v = v1;
        i = Ii/n;
        j = Ii - i*n;
        if (i>0){
            J = Ii - n;
            ierr = MatSetValues(C, 1, &Ii, 1, &J, &v, mode); CHKERRQ(ierr);
        }

        if (i<m-1){
            J = Ii + n;
            ierr = MatSetValues(C, 1, &Ii, 1, &J, &v, mode); CHKERRQ(ierr);
        }

        if (j>0){
            J = Ii - 1;
            ierr = MatSetValues(C, 1, &Ii, 1, &J, &v, mode); CHKERRQ(ierr);
        }

        if (j<n-1){
            J = Ii + 1;
            ierr = MatSetValues(C, 1, &Ii, 1, &J, &v, mode); CHKERRQ(ierr);
        }
        v = v2;
        ierr = MatSetValues(C, 1, &Ii, 1, &Ii, &v, mode); CHKERRQ(ierr);
    }

    return ierr;

}

PetscErrorCode testPETScKSP::FormBlockMatrixA(PetscScalar v1, PetscScalar v2, InsertMode mode)
{
    PetscInt    i, j;
    PetscInt    Ii, J;
    PetscInt    jj;
    PetscInt    II, JJ;
    PetscScalar v;

    for (Ii=Istart/bs; Ii<Iend/bs; Ii++) {
        v = v1;
        i = Ii/n;
        j = Ii - i*n;

        if (i>0) {
            J = Ii - n;
            for (jj=0; jj<bs; jj++) {
                II = Ii*bs + jj;
                JJ = J*bs + jj;
                ierr = MatSetValues(A, 1, &II, 1, &JJ, &v, mode); CHKERRQ(ierr);
            }
        }

        if (i<m-1) {
            J = Ii + n;
            for (jj=0; jj<bs; jj++) {
                II = Ii*bs + jj;
                JJ = J*bs + jj;
                ierr = MatSetValues(A, 1, &II, 1, &JJ, &v, mode); CHKERRQ(ierr);
            }
        }

        if (j>0) {
            J = Ii - 1;
            for (jj=0; jj<bs; jj++) {
                II = Ii*bs + jj;
                JJ = J*bs + jj;
                ierr = MatSetValues(A, 1, &II, 1, &JJ, &v, mode); CHKERRQ(ierr);
            }
        }

        if (j<n-1) {
            J = Ii + 1;
            for (jj=0; jj<bs; jj++) {
                II = Ii*bs + jj;
                JJ = J*bs + jj;
                ierr = MatSetValues(A, 1, &II, 1, &JJ, &v, mode); CHKERRQ(ierr);
            }
        }

        v = v2;
        for (jj=0; jj<bs; jj++) {
            II = Ii*bs + jj;
            ierr = MatSetValues(A, 1, &II, 1, &II, &v, mode); CHKERRQ(ierr);
        }
    }
    return ierr;
}
