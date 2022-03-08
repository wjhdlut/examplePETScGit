#include<iostream>
#include"testPETScMat.h"

testPETScMat::testPETScMat()
{}

testPETScMat::testPETScMat(int rank, int size)
{
    m_rank = rank;
    m_size = size;
}

testPETScMat::~testPETScMat()
{}

/* ------------------ Mat example ex_2 ------------------ */
PetscErrorCode testPETScMat::testMat_SeqDense()
{
    Mat            A, A11, A12, A21, A22;
    Vec            X, X1, X2, Y, Z, Z1, Z2;
    PetscScalar    *a, *b, *x, *y, *z, v, one=1;
    PetscReal      nrm;
    PetscInt       size=8, size1=6, size2=2, i, j;
    PetscRandom    rnd;

    ierr = PetscRandomCreate(PETSC_COMM_SELF, &rnd); CHKERRQ(ierr);

    /* Create matrix and three vectors: these are all normal */
    ierr = PetscMalloc1(size*size, &a); CHKERRQ(ierr);
    ierr = PetscMalloc1(size*size, &b); CHKERRQ(ierr);

    for (i=0; i<size; i++) {
        for (j=0; j<size; j++) {
            ierr = PetscRandomGetValue(rnd, &a[i+j*size]); CHKERRQ(ierr);
            b[i+j*size] = a[i+j*size];
        }
    }
    ierr = MatCreate(PETSC_COMM_SELF, &A); CHKERRQ(ierr);
    ierr = MatSetSizes(A, size, size, size, size);CHKERRQ(ierr);
    ierr = MatSetType(A, MATSEQDENSE); CHKERRQ(ierr);
    ierr = MatSeqDenseSetPreallocation(A, a); CHKERRQ(ierr);
    //ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    //ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatView(A, PETSC_VIEWER_STDOUT_SELF); CHKERRQ(ierr);

    ierr = PetscMalloc1(size, &x); CHKERRQ(ierr);
    for (i=0; i<size; i++) {
        x[i] = one;
    }
    ierr = VecCreateSeqWithArray(PETSC_COMM_SELF, 1, size, x, &X); CHKERRQ(ierr);
    //ierr = VecAssemblyBegin(X); CHKERRQ(ierr);
    //ierr = VecAssemblyEnd(X); CHKERRQ(ierr);
    ierr = VecView(X, PETSC_VIEWER_STDOUT_SELF); CHKERRQ(ierr);

    ierr = PetscMalloc1(size, &y); CHKERRQ(ierr);
    ierr = VecCreateSeqWithArray(PETSC_COMM_SELF, 1, size, y, &Y); CHKERRQ(ierr);
    //ierr = VecAssemblyBegin(Y); CHKERRQ(ierr);
    //ierr = VecAssemblyEnd(Y); CHKERRQ(ierr);

    ierr = PetscMalloc1(size, &z); CHKERRQ(ierr);
    ierr = VecCreateSeqWithArray(PETSC_COMM_SELF, 1, size, z, &Z); CHKERRQ(ierr);
    //ierr = VecAssemblyBegin(Z); CHKERRQ(ierr);
    //ierr = VecAssemblyEnd(Z); CHKERRQ(ierr);

    /* Now create submatrices and subvectors */
    ierr = MatCreate(PETSC_COMM_SELF, &A11); CHKERRQ(ierr);
    ierr = MatSetSizes(A11, size1, size1, size1, size1); CHKERRQ(ierr);
    ierr = MatSetType(A11, MATSEQDENSE); CHKERRQ(ierr);
    ierr = MatSeqDenseSetPreallocation(A11, b); CHKERRQ(ierr);
    ierr = MatDenseSetLDA(A11, size); CHKERRQ(ierr);
    //ierr = MatAssemblyBegin(A11, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    //ierr = MatAssemblyEnd(A11, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    std::cout << " ------------ MatViewA11 ------------- " << std::endl;
    ierr = MatView(A11, PETSC_VIEWER_STDOUT_SELF); CHKERRQ(ierr);

    ierr = MatCreate(PETSC_COMM_SELF, &A12); CHKERRQ(ierr);
    ierr = MatSetSizes(A12, size1, size2, size1, size2); CHKERRQ(ierr);
    ierr = MatSetType(A12, MATSEQDENSE); CHKERRQ(ierr);
    ierr = MatSeqDenseSetPreallocation(A12, b+size1*size); CHKERRQ(ierr);
    ierr = MatDenseSetLDA(A12, size);CHKERRQ(ierr);
    //ierr = MatAssemblyBegin(A12, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    //ierr = MatAssemblyEnd(A12, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

    ierr = MatCreate(PETSC_COMM_SELF, &A21); CHKERRQ(ierr);
    ierr = MatSetSizes(A21, size2, size1, size2, size1); CHKERRQ(ierr);
    ierr = MatSetType(A21, MATSEQDENSE); CHKERRQ(ierr);
    ierr = MatSeqDenseSetPreallocation(A21, b+size1); CHKERRQ(ierr);
    ierr = MatDenseSetLDA(A21, size); CHKERRQ(ierr);
    //ierr = MatAssemblyBegin(A21, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    //ierr = MatAssemblyEnd(A21, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

    ierr = MatCreate(PETSC_COMM_SELF,&A22); CHKERRQ(ierr);
    ierr = MatSetSizes(A22, size2, size2, size2, size2); CHKERRQ(ierr);
    ierr = MatSetType(A22, MATSEQDENSE);CHKERRQ(ierr);
    ierr = MatSeqDenseSetPreallocation(A22, b+size1*size+size1);CHKERRQ(ierr);
    ierr = MatDenseSetLDA(A22, size);CHKERRQ(ierr);
    //ierr = MatAssemblyBegin(A22,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    //ierr = MatAssemblyEnd(A22,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

    ierr = VecCreateSeqWithArray(PETSC_COMM_SELF, 1, size1, x, &X1); CHKERRQ(ierr);
    ierr = VecCreateSeqWithArray(PETSC_COMM_SELF, 1, size2, x+size1, &X2); CHKERRQ(ierr);
    ierr = VecCreateSeqWithArray(PETSC_COMM_SELF, 1, size1, z, &Z1); CHKERRQ(ierr);
    ierr = VecCreateSeqWithArray(PETSC_COMM_SELF, 1, size2, z+size1, &Z2); CHKERRQ(ierr);

    /* Now multiple matrix times input in two ways; compare the results */
    ierr = MatMult(A, X, Y); CHKERRQ(ierr);
    ierr = MatMult(A11, X1, Z1); CHKERRQ(ierr);
    ierr = MatMultAdd(A12, X2, Z1, Z1); CHKERRQ(ierr);
    ierr = MatMult(A22, X2, Z2); CHKERRQ(ierr);
    ierr = MatMultAdd(A21, X1, Z2, Z2); CHKERRQ(ierr);
    ierr = VecAXPY(Z, -1.0, Y); CHKERRQ(ierr);
    ierr = VecNorm(Z, NORM_2, &nrm); CHKERRQ(ierr);
    std::cout << "nrm = " << nrm << std::endl;
    if (nrm > 100.0*PETSC_MACHINE_EPSILON) {
        ierr = PetscPrintf(PETSC_COMM_WORLD, "---- Test1; error norm=%g ---\n",
                           (double)nrm); CHKERRQ(ierr);
    }

    /* Next test: change both matrices */
    ierr = PetscRandomGetValue(rnd, &v); CHKERRQ(ierr);
    i    = 1;
    j    = size-2;
    ierr = MatSetValues(A, 1, &i, 1, &j, &v, INSERT_VALUES); CHKERRQ(ierr);
    j   -= size1;
    ierr = MatSetValues(A12, 1, &i, 1, &j, &v, INSERT_VALUES); CHKERRQ(ierr);
    ierr = PetscRandomGetValue(rnd, &v); CHKERRQ(ierr);
    i    = j = size1+1;
    ierr = MatSetValues(A, 1, &i, 1, &j, &v, INSERT_VALUES); CHKERRQ(ierr);
    i     =j = 1;
    ierr = MatSetValues(A22, 1, &i, 1, &j, &v, INSERT_VALUES); CHKERRQ(ierr);
    ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyBegin(A12, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A12, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyBegin(A22, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A22, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

    ierr = MatMult(A, X, Y); CHKERRQ(ierr);
    ierr = MatMult(A11, X1, Z1); CHKERRQ(ierr);
    ierr = MatMultAdd(A12, X2, Z1, Z1); CHKERRQ(ierr);
    ierr = MatMult(A22, X2, Z2); CHKERRQ(ierr);
    ierr = MatMultAdd(A21, X1, Z2, Z2); CHKERRQ(ierr);
    ierr = VecAXPY(Z, -1.0, Y);CHKERRQ(ierr);
    ierr = VecNorm(Z, NORM_2, &nrm); CHKERRQ(ierr);
    if (nrm > 100.0*PETSC_MACHINE_EPSILON) {
        ierr = PetscPrintf(PETSC_COMM_WORLD, "Test2; error norm=%g\n",
                           (double)nrm);CHKERRQ(ierr);
    }

    /* Transpose product */
    ierr = MatMultTranspose(A, X, Y); CHKERRQ(ierr);
    ierr = MatMultTranspose(A11, X1, Z1); CHKERRQ(ierr);
    ierr = MatMultTransposeAdd(A21, X2, Z1, Z1); CHKERRQ(ierr);
    ierr = MatMultTranspose(A22, X2, Z2); CHKERRQ(ierr);
    ierr = MatMultTransposeAdd(A12, X1, Z2, Z2); CHKERRQ(ierr);
    ierr = VecAXPY(Z, -1.0, Y); CHKERRQ(ierr);
    ierr = VecNorm(Z, NORM_2, &nrm); CHKERRQ(ierr);
    if (nrm > 100.0*PETSC_MACHINE_EPSILON) {
        ierr = PetscPrintf(PETSC_COMM_WORLD, "Test3; error norm=%g\n",
                           (double)nrm); CHKERRQ(ierr);
    }
    ierr = PetscFree(a); CHKERRQ(ierr);
    ierr = PetscFree(b); CHKERRQ(ierr);
    ierr = PetscFree(x); CHKERRQ(ierr);
    ierr = PetscFree(y); CHKERRQ(ierr);
    ierr = PetscFree(z); CHKERRQ(ierr);
    ierr = MatDestroy(&A); CHKERRQ(ierr);
    ierr = MatDestroy(&A11); CHKERRQ(ierr);
    ierr = MatDestroy(&A12); CHKERRQ(ierr);
    ierr = MatDestroy(&A21); CHKERRQ(ierr);
    ierr = MatDestroy(&A22); CHKERRQ(ierr);

    ierr = VecDestroy(&X); CHKERRQ(ierr);
    ierr = VecDestroy(&Y); CHKERRQ(ierr);
    ierr = VecDestroy(&Z); CHKERRQ(ierr);

    ierr = VecDestroy(&X1); CHKERRQ(ierr);
    ierr = VecDestroy(&X2); CHKERRQ(ierr);
    ierr = VecDestroy(&Z1); CHKERRQ(ierr);
    ierr = VecDestroy(&Z2); CHKERRQ(ierr);
    ierr = PetscRandomDestroy(&rnd); CHKERRQ(ierr);

    return ierr;
}

/* ------------------ Mat example ex_3 ------------------ */
PetscErrorCode testPETScMat::testMat_1DLapla()
{
    MPI_Comm               comm = PETSC_COMM_WORLD;
    Mat                    A;
    Vec                    x, y;
    ISLocalToGlobalMapping map;
    PetscScalar            elemMat[4] = {1.0, -1.0, -1.0, 1.0};
    PetscReal              error;
    PetscInt               overlapSize = 2, globalIdx[2];

    /* Create local-to-global map */
    globalIdx[0] = m_rank;
    globalIdx[1] = m_rank+1;

    std::cout << "RANK[" << m_rank << "], globalIdx[0] = " << globalIdx[0]
              << ", globalIdx[1] = " << globalIdx[1] << std::endl;

    ierr = ISLocalToGlobalMappingCreate(comm, 1, overlapSize, globalIdx,
                                        PETSC_COPY_VALUES, &map); CHKERRQ(ierr);
    /* Create matrix */
    ierr = MatCreateIS(comm, 1, PETSC_DECIDE, PETSC_DECIDE,
                       m_size+1, m_size+1, map, map, &A); CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) A, "A"); CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingDestroy(&map); CHKERRQ(ierr);

    ierr = MatISSetPreallocation(A, overlapSize, NULL, overlapSize, NULL); CHKERRQ(ierr);
    ierr = MatSetValues(A, 2, globalIdx, 2, globalIdx, elemMat, ADD_VALUES); CHKERRQ(ierr);
    ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

    //ierr = MatView(A, PETSC_VIEWER_STDERR_WORLD); CHKERRQ(ierr);
    PetscInt row, n;
    ierr = MatGetOwnershipRange(A, &row, &n); CHKERRQ(ierr);
    std::cout << "RANK[" << m_rank << "] rowOfA = " << row
              << ", nOfA = " << n << std::endl;

    /* Check that the constant vector is in the nullspace */
    ierr = MatCreateVecs(A, &x, &y); CHKERRQ(ierr);
    ierr = VecSet(x, 1.0); CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) x, "x"); CHKERRQ(ierr);
    ierr = VecViewFromOptions(x, NULL, "-x_view"); CHKERRQ(ierr);

    PetscInt x_globalSize, x_localSize;
    ierr = VecGetSize(x, &x_globalSize); CHKERRQ(ierr);
    //std::cout << "RANK[" << m_rank << "], x_global_size = " << x_globalSize << std::endl;

    ierr = VecGetLocalSize(x, &x_localSize); CHKERRQ(ierr);
    std::cout << "RANK[" << m_rank << "], x_local_size = " << x_localSize << std::endl;

    ierr = MatMult(A, x, y); CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) y, "y"); CHKERRQ(ierr);
    ierr = VecViewFromOptions(y, NULL, "-y_view"); CHKERRQ(ierr);

    ierr = VecNorm(y, NORM_2, &error); CHKERRQ(ierr);

    /* Check that an interior unit vector gets mapped to something of 1-norm 4 */
    if (m_size > 1) {
        ierr = VecSet(x, 0.0); CHKERRQ(ierr);
        ierr = VecSetValue(x, 1, 1.0, INSERT_VALUES); CHKERRQ(ierr);
        ierr = VecAssemblyBegin(x); CHKERRQ(ierr);
        ierr = VecAssemblyEnd(x); CHKERRQ(ierr);
        ierr = MatMult(A, x, y); CHKERRQ(ierr);
        ierr = VecNorm(y, NORM_1, &error); CHKERRQ(ierr);
        ierr = VecView(y, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
    }

    /* Cleanup */
    ierr = MatDestroy(&A); CHKERRQ(ierr);
    ierr = VecDestroy(&x); CHKERRQ(ierr);
    ierr = VecDestroy(&y); CHKERRQ(ierr);

    return ierr;
}

/* ------------------ Mat example ex_4 ------------------ */
PetscErrorCode testPETScMat::testMat_ResetPreallocation()
{
    Mat             A;
    PetscInt        i, rstart, rend, M, N;

    MPI_Comm        comm = MPI_COMM_WORLD;
    PetscInt        n=5, m=5;

    PetscInt        *dnnz = new PetscInt[m];
    PetscInt        *onnz = new PetscInt[m];

    for (i=0; i<m; i++)
    {
        dnnz[i] = 1;
        onnz[i] = 1;
    }
    ierr = MatCreateAIJ(comm , m, n, PETSC_DECIDE, PETSC_DETERMINE,
                        PETSC_DECIDE, dnnz, PETSC_DECIDE, onnz, &A); CHKERRQ(ierr);
    ierr = MatSetFromOptions(A); CHKERRQ(ierr);
    ierr = MatSetUp(A); CHKERRQ(ierr);

    /* This assembly shrinks memory because we do not insert enough number of values */
    ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

    //ierr = MatView(A, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

    /* MatResetPreallocation restores the memory required by users */
    ierr = MatResetPreallocation(A); CHKERRQ(ierr);
    ierr = MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE); CHKERRQ(ierr);
    ierr = MatGetOwnershipRange(A, &rstart, &rend); CHKERRQ(ierr);
    ierr = MatGetSize(A, &M, &N); CHKERRQ(ierr);

    std::cout << "RANK[" << m_rank << "] rstart = " << rstart << ", rend = " << rend << std::endl;
    std::cout << "RANK[" << m_rank << "] M = " << M << ", N = " << N << std::endl;

    for (i=rstart; i<rend; i++)
    {
        ierr = MatSetValue(A, i, i, 2.0, INSERT_VALUES); CHKERRQ(ierr);
        if (rend<N)
        {
            ierr = MatSetValue(A, i, rend, 1.0, INSERT_VALUES); CHKERRQ(ierr);
        }
        else
        {
            ierr = MatSetValue(A, i, rstart-1, 3.0, INSERT_VALUES); CHKERRQ(ierr);
        }
    }
    ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatView(A, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
    ierr = MatDestroy(&A); CHKERRQ(ierr);

    delete[] dnnz;
    delete[] onnz;

    return ierr;
}

/* ------------------ Mat example ex_7 ------------------ */
PetscErrorCode testPETScMat::testMat_PetscInfo()
{
    Mat             A, Aself;
    Vec             b, bself;
#if defined(PETSC_USE_INFO)
    PetscInt        testarg = 1234;
#endif
    int             numClasses;
    PetscClassId    testMatClassid, testVecClassid, testSysClassid;
    PetscBool       isEnabled = PETSC_FALSE, invert = PETSC_FALSE;
    char            *testClassesStr, *filename;
    const char      *testMatClassname, *testVecClassname;
    char            **testClassesStrArr;
    FILE            *infoFile;

    /* Examples on how to call PetscInfo() using different objects with or
       without arguments, and different communicators.
       - Until PetscInfoDestroy() is called all PetscInfo() behaviour is goverened
         by command line options, which are processed during PetscInitialize(). */

    ierr = MatCreate(PETSC_COMM_WORLD, &A); CHKERRQ(ierr);
    ierr = VecCreate(PETSC_COMM_WORLD, &b); CHKERRQ(ierr);

    ierr = PetscInfo(A, "Mat info on PETSC_COMM_WORLD"
                        "with no arguments\n"); CHKERRQ(ierr);

    ierr = PetscInfo1(A, "Mat info on PETSC_COMM_WORLD with 1"
                         "argument equal to 1234: %D\n", testarg); CHKERRQ(ierr);

    ierr = PetscInfo(b, "Vec info on PETSC_COMM_WORLD with"
                        " no arguments\n"); CHKERRQ(ierr);

    ierr = PetscInfo1(b, "Vec info on PETSC_COMM_WORLD with 1"
                         "argument equal to 1234: %D\n", testarg); CHKERRQ(ierr);

    ierr = PetscInfo(NULL, "Sys info on PETSC_COMM_WORLD with"
                           " no arguments\n"); CHKERRQ(ierr);

    ierr = PetscInfo1(NULL, "Sys info on PETSC_COMM_WORLD with 1"
                            " argument equal to 1234: %D\n", testarg); CHKERRQ(ierr);

    ierr = MatCreate(PETSC_COMM_SELF, &Aself); CHKERRQ(ierr);
    ierr = VecCreate(PETSC_COMM_SELF, &bself); CHKERRQ(ierr);

    ierr = PetscInfo(Aself, "Mat info on PETSC_COMM_SELF with"
                            " no arguments\n"); CHKERRQ(ierr);

    ierr = PetscInfo1(Aself, "Mat info on PETSC_COMM_SELF with 1"
                             " argument equal to 1234: %D\n", testarg); CHKERRQ(ierr);

    ierr = PetscInfo(bself, "Vec info on PETSC_COMM_SELF with"
                            " no arguments\n"); CHKERRQ(ierr);

    ierr = PetscInfo1(bself, "Vec info on PETSC_COMM_SELF with"
                             " 1 argument equal to 1234: %D\n", testarg); CHKERRQ(ierr);

    ierr = PetscInfo(NULL, "Sys info on PETSC_COMM_SELF with"
                           " no arguments\n"); CHKERRQ(ierr);

    ierr = PetscInfo1(NULL, "Sys info on PETSC_COMM_SELF with"
                            " 1 argument equal to 1234: %D\n", testarg); CHKERRQ(ierr);

    ierr = MatDestroy(&Aself); CHKERRQ(ierr);
    ierr = VecDestroy(&bself); CHKERRQ(ierr);

    /* First retrieve some basic information regarding the classes
       for which we want to filter */
    ierr = PetscObjectGetClassId((PetscObject) A, &testMatClassid); CHKERRQ(ierr);
    ierr = PetscObjectGetClassId((PetscObject) b, &testVecClassid); CHKERRQ(ierr);

    /* Sys class has PetscClassId = PETSC_SMALLEST_CLASSID */
    testSysClassid = PETSC_SMALLEST_CLASSID;
    ierr = PetscObjectGetClassName((PetscObject) A, &testMatClassname); CHKERRQ(ierr);
    ierr = PetscObjectGetClassName((PetscObject) b, &testVecClassname); CHKERRQ(ierr);

    /* Examples on how to use individual PetscInfo() commands. */
    ierr = PetscInfoEnabled(testMatClassid, &isEnabled); CHKERRQ(ierr);
    if (isEnabled) { ierr = PetscInfo(A, "Mat info is enabled\n"); CHKERRQ(ierr);}

    ierr = PetscInfoEnabled(testVecClassid, &isEnabled); CHKERRQ(ierr);
    if (isEnabled) { ierr = PetscInfo(b, "Vec info is enabled\n"); CHKERRQ(ierr);}

    ierr = PetscInfoEnabled(testSysClassid, &isEnabled); CHKERRQ(ierr);
    if (isEnabled) { ierr = PetscInfo(NULL, "Sys info is enabled\n"); CHKERRQ(ierr);}

    /* Retrieve filename to append later entries to */
    ierr = PetscInfoGetFile(&filename, &infoFile); CHKERRQ(ierr);

    /* Destroy existing PetscInfo() configuration and reset all internal flags
       to default values. This allows the user to change filters
       midway through a program. */
    ierr = PetscInfoDestroy(); CHKERRQ(ierr);

    /* Test if existing filters are reset.
        - Note these should NEVER print. */
    ierr = PetscInfoEnabled(testMatClassid, &isEnabled); CHKERRQ(ierr);
    if (isEnabled)
    {
        ierr = PetscInfo(A, "Mat info is enabled after PetscInfoDestroy\n"); CHKERRQ(ierr);
    }

    ierr = PetscInfoEnabled(testVecClassid, &isEnabled); CHKERRQ(ierr);
    if (isEnabled)
    {
        ierr = PetscInfo(b, "Vec info is enabled after PetscInfoDestroy\n"); CHKERRQ(ierr);
    }

    ierr = PetscInfoEnabled(testSysClassid, &isEnabled); CHKERRQ(ierr);
    if (isEnabled)
    {
        ierr = PetscInfo(NULL, "Sys info is enabled after PetscInfoDestroy\n"); CHKERRQ(ierr);
    }

    /* Reactivate PetscInfo() printing in one of two ways.
        - First we must reactivate PetscInfo() printing as a whole.
        - Keep in mind that by default ALL classes are allowed to print
          if PetscInfo() is enabled, so we deactivate relevant classes
          first to demonstrate activation functionality. */
    ierr = PetscInfoAllow(PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscInfoSetFile(filename, "a");CHKERRQ(ierr);
    ierr = PetscInfoDeactivateClass(testMatClassid);CHKERRQ(ierr);
    ierr = PetscInfoDeactivateClass(testVecClassid);CHKERRQ(ierr);
    ierr = PetscInfoDeactivateClass(testSysClassid);CHKERRQ(ierr);

    /* Activate PetscInfo() on a per-class basis */
    ierr = PetscInfoActivateClass(testMatClassid);CHKERRQ(ierr);
    ierr = PetscInfo(A, "Mat info is enabled again through PetscInfoActivateClass\n");CHKERRQ(ierr);
    ierr = PetscInfoDeactivateClass(testMatClassid);CHKERRQ(ierr);
    ierr = PetscInfoActivateClass(testVecClassid);CHKERRQ(ierr);
    ierr = PetscInfo(b, "Vec info is enabled again through PetscInfoActivateClass\n");CHKERRQ(ierr);
    ierr = PetscInfoDeactivateClass(testVecClassid);CHKERRQ(ierr);
    ierr = PetscInfoActivateClass(testSysClassid);CHKERRQ(ierr);
    ierr = PetscInfo(NULL, "Sys info is enabled again through PetscInfoActivateClass\n");CHKERRQ(ierr);
    ierr = PetscInfoDeactivateClass(testVecClassid);CHKERRQ(ierr);

    /* Activate PetscInfo() by specifying specific classnames to activate */
    ierr = PetscStrallocpy("mat,vec,sys", &testClassesStr);CHKERRQ(ierr);
    ierr = PetscStrToArray((const char *)testClassesStr, ',', &numClasses, &testClassesStrArr);CHKERRQ(ierr);
    ierr = PetscInfoSetClasses(invert, (PetscInt) numClasses, (const char *const *) testClassesStrArr);CHKERRQ(ierr);
    ierr = PetscInfoProcessClass(testMatClassname, 1, &testMatClassid);CHKERRQ(ierr);
    ierr = PetscInfoProcessClass(testVecClassname, 1, &testVecClassid);CHKERRQ(ierr);
    ierr = PetscInfoProcessClass("sys", 1, &testSysClassid);CHKERRQ(ierr);

    ierr = PetscInfo(A, "Mat info is enabled again through PetscInfoSetClasses\n");CHKERRQ(ierr);
    ierr = PetscInfo(b, "Vec info is enabled again through PetscInfoSetClasses\n");CHKERRQ(ierr);
    ierr = PetscInfo(NULL, "Sys info is enabled again through PetscInfoSetClasses\n");CHKERRQ(ierr);

    ierr = PetscStrToArrayDestroy(numClasses, testClassesStrArr);CHKERRQ(ierr);
    ierr = PetscFree(testClassesStr);CHKERRQ(ierr);

    /* Activate PetscInfo() with an inverted filter selection.
        - Inverting our selection of filters enables PetscInfo()
          for all classes EXCEPT those specified.
        - Note we must reset PetscInfo() internal flags with PetscInfoDestroy()
          as invoking PetscInfoProcessClass() locks filters in place. */
    ierr = PetscInfoDestroy();CHKERRQ(ierr);
    ierr = PetscInfoAllow(PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscInfoSetFile(filename, "a");CHKERRQ(ierr);
    ierr = PetscStrallocpy("vec,sys", &testClassesStr);CHKERRQ(ierr);
    ierr = PetscStrToArray((const char *)testClassesStr, ',', &numClasses, &testClassesStrArr);CHKERRQ(ierr);
    invert = PETSC_TRUE;
    ierr = PetscInfoSetClasses(invert, (PetscInt) numClasses, (const char *const *) testClassesStrArr);CHKERRQ(ierr);
    ierr = PetscInfoProcessClass(testMatClassname, 1, &testMatClassid);CHKERRQ(ierr);
    ierr = PetscInfoProcessClass(testVecClassname, 1, &testVecClassid);CHKERRQ(ierr);
    ierr = PetscInfoProcessClass("sys", 1, &testSysClassid);CHKERRQ(ierr);

    /* Here only the Mat() call will successfully print. */
    ierr = PetscInfo(A, "Mat info is enabled again through inverted PetscInfoSetClasses\n");CHKERRQ(ierr);
    ierr = PetscInfo(b, "Vec info is enabled again through PetscInfoSetClasses\n");CHKERRQ(ierr);
    ierr = PetscInfo(NULL, "Sys info is enabled again through PetscInfoSetClasses\n");CHKERRQ(ierr);

    ierr = PetscStrToArrayDestroy(numClasses, testClassesStrArr);CHKERRQ(ierr);
    ierr = PetscFree(testClassesStr);CHKERRQ(ierr);
    ierr = PetscFree(filename);CHKERRQ(ierr);
    ierr = MatDestroy(&A);CHKERRQ(ierr);
    ierr = VecDestroy(&b);CHKERRQ(ierr);

    return ierr;
}
