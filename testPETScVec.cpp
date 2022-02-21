#include <iostream>
#include "testPETScVec.h"

testPETScVec::testPETScVec(){}

testPETScVec::testPETScVec(int rank, int size)
{
    m_rank = rank;
    m_size = size;
}

testPETScVec::~testPETScVec()
{

}

/* ------------------ Vec example ex_1 ------------------ */
PetscErrorCode testPETScVec::testVec_CreateBasicOperation()
{
    Vec            x, y, w;
    Vec            *z;
    PetscReal      norm, v, v1, v2, maxval;
    PetscInt       n = 20;
    PetscInt       maxind;
    PetscScalar    one = 1.0;
    PetscScalar    two = 2.0;
    PetscScalar    three = 3.0;
    PetscScalar    dots[3];
    PetscScalar    dot;

    ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL); CHKERRQ(ierr);
    std::cout << "RANK[" << m_rank << "] n = " << n << std::endl;

    ierr = VecCreate(PETSC_COMM_WORLD, &x); CHKERRQ(ierr);
    ierr = VecSetSizes(x, PETSC_DECIDE, n); CHKERRQ(ierr);
    ierr = VecSetFromOptions(x); CHKERRQ(ierr);

    ierr = VecDuplicate(x, &y); CHKERRQ(ierr);
    ierr = VecDuplicate(x, &w); CHKERRQ(ierr);

    ierr = VecDuplicateVecs(x,3,&z); CHKERRQ(ierr);

    ierr = VecSet(x, one); CHKERRQ(ierr);
    //ierr = VecView(x, PETSC_VIEWER_STDOUT_SELF);

    ierr = VecSet(y, two); CHKERRQ(ierr);
    ierr = VecSet(z[0], one); CHKERRQ(ierr);
    ierr = VecSet(z[1], two); CHKERRQ(ierr);
    ierr = VecSet(z[2], three); CHKERRQ(ierr);

    ierr = VecDot(x, y, &dot); CHKERRQ(ierr);
    std::cout << "RANK[" << m_rank << "] dot = " << dot << std::endl;
    ierr = VecMDot(x, 3, z, dots); CHKERRQ(ierr);

    ierr = PetscPrintf(PETSC_COMM_WORLD, "Vector length %" PetscInt_FMT "\n", n); CHKERRQ(ierr);
    ierr = VecMax(x, &maxind, &maxval); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "VecMax %g, VecInd %" PetscInt_FMT "\n", (double)maxval, maxind); CHKERRQ(ierr);

    ierr = VecMin(x, &maxind, &maxval); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "VecMin %g, VecInd %" PetscInt_FMT "\n", (double)maxval, maxind); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "All other values should be near zero\n"); CHKERRQ(ierr);

    ierr = VecScale(x, two); CHKERRQ(ierr);
    ierr = VecNorm(x, NORM_2, &norm); CHKERRQ(ierr);
    std::cout << "RANK[" << m_rank << "] normOfVecx = " << norm << std::endl;

    v    = norm - 2.0 * PetscSqrtReal((PetscReal)n);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "v = %g\n", (double)v); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "PETSC_SMALL = %g\n", (double)PETSC_SMALL); CHKERRQ(ierr);

    if (v > -PETSC_SMALL && v < PETSC_SMALL) v = 0.0;
    ierr = PetscPrintf(PETSC_COMM_WORLD, "VecScale %g\n", (double)v); CHKERRQ(ierr);

    ierr = VecCopy(x, w); CHKERRQ(ierr);
    ierr = VecNorm(w, NORM_2, &norm); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "normOfVecw = %g\n", norm); CHKERRQ(ierr);
    v    = norm - 2.0 * PetscSqrtReal((PetscReal)n); if (v > -PETSC_SMALL && v < PETSC_SMALL) v = 0.0;
    ierr = PetscPrintf(PETSC_COMM_WORLD,"VecCopy  %g\n",(double)v);CHKERRQ(ierr);

    ierr = VecAXPY(y, three, x); CHKERRQ(ierr);
    ierr = VecNorm(y, NORM_2, &norm); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "normOfVecy = %g\n", norm); CHKERRQ(ierr);
    v    = norm - 8.0 * PetscSqrtReal((PetscReal)n); if (v > -PETSC_SMALL && v < PETSC_SMALL) v = 0.0;
    ierr = PetscPrintf(PETSC_COMM_WORLD, "VecAXPY %g\n", (double)v); CHKERRQ(ierr);

    ierr = VecAYPX(y, two, x); CHKERRQ(ierr);
    ierr = VecNorm(y, NORM_2, &norm); CHKERRQ(ierr);
    v    = norm - 18.0 * PetscSqrtReal((PetscReal)n); if (v > -PETSC_SMALL && v < PETSC_SMALL) v = 0.0;
    ierr = PetscPrintf(PETSC_COMM_WORLD, "VecAYPX %g\n", (double)v); CHKERRQ(ierr);

    ierr = VecSwap(x, y); CHKERRQ(ierr);
    ierr = VecNorm(y, NORM_2, &norm); CHKERRQ(ierr);
    v    = norm - 2.0 * PetscSqrtReal((PetscReal)n); if (v > -PETSC_SMALL && v < PETSC_SMALL) v = 0.0;
    ierr = PetscPrintf(PETSC_COMM_WORLD, "VecSwap  %g\n", (double)v); CHKERRQ(ierr);

    ierr = VecNorm(x,NORM_2,&norm);CHKERRQ(ierr);
    v    = norm - 18.0 * PetscSqrtReal((PetscReal)n); if (v > -PETSC_SMALL && v < PETSC_SMALL) v = 0.0;
    ierr = PetscPrintf(PETSC_COMM_WORLD, "VecSwap  %g\n", (double)v); CHKERRQ(ierr);

    ierr = VecWAXPY(w, two, x, y); CHKERRQ(ierr);
    ierr = VecNorm(w, NORM_2, &norm); CHKERRQ(ierr);
    v    = norm - 38.0 * PetscSqrtReal((PetscReal)n); if (v > -PETSC_SMALL && v < PETSC_SMALL) v = 0.0;
    ierr = PetscPrintf(PETSC_COMM_WORLD, "VecWAXPY %g\n", (double)v); CHKERRQ(ierr);

    ierr = VecPointwiseMult(w, y, x); CHKERRQ(ierr);
    ierr = VecNorm(w, NORM_2, &norm); CHKERRQ(ierr);
    v    = norm - 36.0 * PetscSqrtReal((PetscReal)n); if (v > -PETSC_SMALL && v < PETSC_SMALL) v = 0.0;
    ierr = PetscPrintf(PETSC_COMM_WORLD, "VecPointwiseMult %g\n", (double)v); CHKERRQ(ierr);

    ierr = VecPointwiseDivide(w, x, y);CHKERRQ(ierr);
    ierr = VecNorm(w, NORM_2, &norm);CHKERRQ(ierr);
    v    = norm - 9.0 * PetscSqrtReal((PetscReal)n); if (v > -PETSC_SMALL && v < PETSC_SMALL) v = 0.0;
    ierr = PetscPrintf(PETSC_COMM_WORLD, "VecPointwiseDivide %g\n", (double)v); CHKERRQ(ierr);

    dots[0] = one;
    dots[1] = three;
    dots[2] = two;

    ierr = VecSet(x, one); CHKERRQ(ierr);
    ierr = VecMAXPY(x, 3, dots, z); CHKERRQ(ierr);
    ierr = VecNorm(z[0], NORM_2, &norm); CHKERRQ(ierr);
    v    = norm - PetscSqrtReal((PetscReal)n); if (v > -PETSC_SMALL && v < PETSC_SMALL) v = 0.0;
    ierr = VecNorm(z[1], NORM_2, &norm); CHKERRQ(ierr);
    v1   = norm - 2.0 * PetscSqrtReal((PetscReal)n); if (v1 > -PETSC_SMALL && v1 < PETSC_SMALL) v1 = 0.0;
    ierr = VecNorm(z[2], NORM_2, &norm); CHKERRQ(ierr);
    v2   = norm - 3.0 * PetscSqrtReal((PetscReal)n); if (v2 > -PETSC_SMALL && v2 < PETSC_SMALL) v2 = 0.0;
    ierr = PetscPrintf(PETSC_COMM_WORLD, "VecMAXPY %g %g %g \n", (double)v, (double)v1, (double)v2); CHKERRQ(ierr);

    ierr = VecDestroy(&x); CHKERRQ(ierr);
    ierr = VecDestroy(&y); CHKERRQ(ierr);
    ierr = VecDestroy(&w); CHKERRQ(ierr);
    ierr = VecDestroyVecs(3, &z); CHKERRQ(ierr);
    return ierr;
}

/* ------------------ Vec example ex_2 ------------------ */
PetscErrorCode testPETScVec::testVec_VecSetValues()
{
    PetscInt       i,N;
    PetscScalar    one = 1.0;
    Vec            x;
    PetscInt       localN;

    ierr = VecCreate(PETSC_COMM_WORLD, &x); CHKERRQ(ierr);
    ierr = VecSetSizes(x, m_rank+1, PETSC_DECIDE); CHKERRQ(ierr);
    ierr = VecSetFromOptions(x); CHKERRQ(ierr);

    ierr = VecGetSize(x, &N); CHKERRQ(ierr);
    ierr = VecSet(x, one); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Global size of vector x is %d\n", N); CHKERRQ(ierr);
    //ierr = VecView(x, PETSC_VIEWER_STDOUT_WORLD);

    ierr   = VecGetLocalSize(x, &localN);
    std::cout << "RANK[" << m_rank << "] localN = " << localN << std::endl;

    for (i=0; i<N-m_rank; i++)
    {
      ierr = VecSetValues(x, 1, &i, &one, ADD_VALUES); CHKERRQ(ierr);
    }

    ierr = VecAssemblyBegin(x); CHKERRQ(ierr);
    ierr = VecAssemblyEnd(x); CHKERRQ(ierr);

    ierr = VecView(x, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
    ierr = VecDestroy(&x); CHKERRQ(ierr);

    return ierr;
}

/* ------------------ Vec example ex_3 ------------------ */
PetscErrorCode testPETScVec::testVec_ex3()
{
    PetscInt       i, nlocal;
    PetscInt       n = 6;
    PetscScalar    *array;
    PetscScalar    v;
    Vec            x;
    PetscViewer    viewer;

    ierr = PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL); CHKERRQ(ierr);

    ierr = VecCreate(PETSC_COMM_WORLD, &x); CHKERRQ(ierr);
    ierr = VecSetSizes(x, PETSC_DECIDE, n); CHKERRQ(ierr);
    ierr = VecSetFromOptions(x); CHKERRQ(ierr);

    ierr = VecGetOwnershipRange(x, &rstart, &rend);CHKERRQ(ierr);
    std::cout << "RANK[" << m_rank << "] rstart = " << rstart
              << ", rend = " << rend << std::endl;

    Vec tempY;
    ierr = VecDuplicate(x, &tempY); CHKERRQ(ierr);
    for(i=0; i<n; i++)
    {
        v    = (PetscReal)(m_rank*i);
        //ierr = VecSetValues(tempY, 1, &i, &v, ADD_VALUES); CHKERRQ(ierr);
        ierr = VecSetValue(tempY, i, v, ADD_VALUES); CHKERRQ(ierr);
    }
    ierr = VecAssemblyBegin(tempY); CHKERRQ(ierr);
    ierr = VecAssemblyEnd(tempY); CHKERRQ(ierr);
    ierr = VecView(tempY, PETSC_VIEWER_STDOUT_WORLD);
    ierr = VecDestroy(&tempY);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "RANK[%d]--------------\n", m_rank);

    PetscScalar    *value = new PetscScalar[n];
    PetscInt       *index = new PetscInt[n];
    for (i=0; i<n; i++) {
      value[i]     = (PetscReal)(m_rank*i);
      index[i] = i;
    }
    ierr = VecSetValues(x, n, index, value, ADD_VALUES); CHKERRQ(ierr);
    ierr = VecAssemblyBegin(x); CHKERRQ(ierr);
    ierr = VecAssemblyEnd(x); CHKERRQ(ierr);
    ierr = VecView(x, PETSC_VIEWER_STDOUT_WORLD);
    delete[] value;
    delete[] index;

    ierr = PetscViewerDrawOpen(PETSC_COMM_WORLD, NULL, NULL, 0, 0, 300, 300, &viewer); CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)viewer, "Line graph Plot"); CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_DRAW_LG); CHKERRQ(ierr);
    ierr = VecView(x, viewer); CHKERRQ(ierr);

    ierr = VecGetLocalSize(x, &nlocal); CHKERRQ(ierr);
    ierr = VecGetArray(x, &array); CHKERRQ(ierr);
    for (i=0; i<nlocal; i++)
    {
        array[i] = m_rank + 1;
    }
    ierr = VecRestoreArray(x, &array); CHKERRQ(ierr);
    ierr = VecView(x, PETSC_VIEWER_STDOUT_WORLD);

    ierr = VecView(x,viewer);CHKERRQ(ierr);

    ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    ierr = VecDestroy(&x);CHKERRQ(ierr);

    return ierr;
}

/* ------------------ Vec example ex_5 ------------------ */
PetscErrorCode testPETScVec::testVec_Print()
{
    PetscInt       i, m = 10;
    PetscInt       ldim,iglobal;
    PetscScalar    v;
    Vec            u;
    PetscViewer    viewer;

    ierr = PetscOptionsGetInt(NULL, NULL, "-m", &m, NULL);CHKERRQ(ierr);

    /* PART 1:  Generate vector, then write it in binary format */

    /* Generate vector */
    ierr = VecCreate(PETSC_COMM_WORLD, &u); CHKERRQ(ierr);
    ierr = VecSetSizes(u, PETSC_DECIDE, m); CHKERRQ(ierr);
    ierr = VecSetFromOptions(u); CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(u, &rstart, &rend); CHKERRQ(ierr);
    ierr = VecGetLocalSize(u, &ldim); CHKERRQ(ierr);

    for (i=0; i<ldim; i++) {
      iglobal = i + rstart;
      v       = (PetscScalar)(i + 100*m_rank);
      ierr    = VecSetValues(u, 1, &iglobal, &v, INSERT_VALUES); CHKERRQ(ierr);
    }
    ierr = VecAssemblyBegin(u); CHKERRQ(ierr);
    ierr = VecAssemblyEnd(u); CHKERRQ(ierr);
    ierr = VecView(u, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

    ierr = PetscPrintf(PETSC_COMM_WORLD, "writing vector in binary to vector.dat ...\n"); CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, "vector.dat", FILE_MODE_WRITE, &viewer); CHKERRQ(ierr);
    ierr = VecView(u, viewer); CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
    ierr = VecDestroy(&u); CHKERRQ(ierr);
    ierr = PetscOptionsSetValue(NULL, "-viewer_binary_mpiio", ""); CHKERRQ(ierr);

    /* PART 2:  Read in vector in binary format */

    /* Read new vector in binary format */
    ierr = PetscPrintf(PETSC_COMM_WORLD, "reading vector in binary from vector.dat ...\n"); CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, "vector.dat", FILE_MODE_READ, &viewer); CHKERRQ(ierr);
    ierr = VecCreate(PETSC_COMM_WORLD, &u); CHKERRQ(ierr);
    ierr = VecLoad(u, viewer); CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

    ierr = VecView(u, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

    /* Free data structures */
    ierr = VecDestroy(&u); CHKERRQ(ierr);
    return ierr;
}

/* ------------------ Vec example ex_8 ------------------ */
PetscErrorCode testPETScVec::testVec_LocalToGlobalIndex()
{
    PetscInt       i;
    PetscInt       ng;
    PetscInt       *gindices;
    PetscInt       M;
    PetscScalar    one = 1.0;
    Vec            x;

    ierr = VecCreate(PETSC_COMM_WORLD, &x); CHKERRQ(ierr);
    ierr = VecSetSizes(x, m_rank+1, PETSC_DECIDE); CHKERRQ(ierr);
    ierr = VecSetFromOptions(x); CHKERRQ(ierr);
    ierr = VecSet(x, one); CHKERRQ(ierr);

    ierr = VecGetSize(x, &M); CHKERRQ(ierr);
    std::cout << "RANK[" << m_rank << "] M = " << M << std::endl;

    ierr = VecGetOwnershipRange(x, &rstart, &rend);CHKERRQ(ierr);
    std::cout << "RANK[" << m_rank << "] rstart = " << rstart << ", rend = " << rend << std::endl;

//    ng   = rend - rstart;
//    ierr = PetscMalloc1(ng, &gindices); CHKERRQ(ierr);

//    for (i=rstart; i<rend; i++)
//    {
//        gindices[i-rstart] = i;
//    }

    ng   = rend - rstart + 2;
    ierr = PetscMalloc1(ng, &gindices); CHKERRQ(ierr);

    // Golbal index of Vec x
    gindices[0] = rstart - 1;
    for (i=0; i<ng-1; i++)
    {
        gindices[i+1] = gindices[i] + 1;
    }

    /* map the first and last point as periodic */
    if (gindices[0]    == -1)
    {
        gindices[0]    = M - 1;
    }

    if (gindices[ng-1] == M)
    {
        gindices[ng-1] = 0;
    }

    ISLocalToGlobalMapping ltog;
    ierr = ISLocalToGlobalMappingCreate(PETSC_COMM_SELF, 1, ng, gindices,
                                        PETSC_COPY_VALUES, &ltog); CHKERRQ(ierr);
    ierr = VecSetLocalToGlobalMapping(x, ltog); CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingDestroy(&ltog); CHKERRQ(ierr);

    ierr = PetscFree(gindices); CHKERRQ(ierr);

    for (i=0; i<ng; i++)
    {
      ierr = VecSetValuesLocal(x, 1, &i, &one, ADD_VALUES); CHKERRQ(ierr);
    }
    ierr = VecAssemblyBegin(x); CHKERRQ(ierr);
    ierr = VecAssemblyEnd(x); CHKERRQ(ierr);

    ierr = VecView(x, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
    ierr = VecDestroy(&x); CHKERRQ(ierr);

    return ierr;
}

/* ------------------ Vec example ex_9 ------------------ */
PetscErrorCode testPETScVec::testVec_CreateGhost()
{
    PetscInt       nlocal = 6;
    PetscInt       nghost = 2;
    PetscInt       ifrom[2];
    PetscInt       i;
    PetscBool      flg, flg2, flg3;
    PetscScalar    value, *array, *tarray = 0;
    Vec            lx, gx, gxs;

    if (m_size != 2)
    {
        SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE,
                "Must run example with two processors\n");
    }

    if (m_rank == 0)
    {
        ifrom[0] = 6;
        ifrom[1] = 11;
    }
    else
    {
        ifrom[0] = 0;
        ifrom[1] = 5;
    }

    ierr = PetscOptionsHasName(NULL, NULL, "-allocate", &flg); CHKERRQ(ierr);
    ierr = PetscOptionsHasName(NULL, NULL, "-vecmpisetghost", &flg2); CHKERRQ(ierr);
    ierr = PetscOptionsHasName(NULL, NULL, "-minvalues", &flg3); CHKERRQ(ierr);

    std::cout << "RANK[" << m_rank << "] flg = " << flg << std::endl;
    std::cout << "RANK[" << m_rank << "] flg2 = " << flg2 << std::endl;
    std::cout << "RANK[" << m_rank << "] flg3 = " << flg3 << std::endl;

    /* Create the vector */
    if (flg) {
        ierr = PetscMalloc1(nlocal+nghost, &tarray); CHKERRQ(ierr);
        ierr = VecCreateGhostWithArray(PETSC_COMM_WORLD, nlocal, PETSC_DECIDE,
                                       nghost, ifrom, tarray, &gxs); CHKERRQ(ierr);
    }
    else if (flg2) {
        ierr = VecCreate(PETSC_COMM_WORLD, &gxs); CHKERRQ(ierr);
        ierr = VecSetType(gxs, VECMPI); CHKERRQ(ierr);
        ierr = VecSetSizes(gxs, nlocal, PETSC_DECIDE); CHKERRQ(ierr);
        ierr = VecMPISetGhost(gxs, nghost, ifrom); CHKERRQ(ierr);
    }
    else {
        ierr = VecCreateGhost(PETSC_COMM_WORLD, nlocal, PETSC_DECIDE,
                              nghost, ifrom, &gxs); CHKERRQ(ierr);
    }

    ierr = VecDuplicate(gxs, &gx); CHKERRQ(ierr);
    ierr = VecDestroy(&gxs); CHKERRQ(ierr);

    /* Output the global and local size of global and local vector */
    ierr = VecGhostGetLocalForm(gx, &lx); CHKERRQ(ierr);

    PetscInt   nlx, ngx;
    ierr = VecGetSize(lx, &nlx); CHKERRQ(ierr);
    ierr = VecGetSize(gx, &ngx); CHKERRQ(ierr);
    std::cout << "RANK[" << m_rank << "] nlx = " << nlx << ", ngx = " << ngx << std::endl;

    PetscInt   nlocallx, nlocalgx;
    ierr = VecGetLocalSize(lx, &nlocallx); CHKERRQ(ierr);
    ierr = VecGetLocalSize(gx, &nlocalgx); CHKERRQ(ierr);
    std::cout << "RANK[" << m_rank << "] nlocallx = " << nlocallx
              << ", nlocalgx = " << nlocalgx << std::endl;


    /* Set the values from 0 to 12 into the "global" vector */
    ierr = VecGetOwnershipRange(gx, &rstart, &rend); CHKERRQ(ierr);

    for (i=rstart; i<rend; i++)
    {
        value = (PetscScalar) i + (m_rank+1)*10;
        ierr  = VecSetValues(gx, 1, &i, &value, INSERT_VALUES); CHKERRQ(ierr);
    }
    ierr = VecAssemblyBegin(gx); CHKERRQ(ierr);
    ierr = VecAssemblyEnd(gx); CHKERRQ(ierr);
    std::cout << "RANK[" << m_rank << "] rstart = " << rstart
              << ", rend = " << rend << std::endl;

    ierr = VecView(gx, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

    PetscSynchronizedPrintf(PETSC_COMM_WORLD, "Before VecGhostUpdateBegin() rank = %d\n", m_rank);
    ierr = VecGetArray(lx, &array); CHKERRQ(ierr);
    for (i=0; i<nlocal+nghost; i++)
    {
      ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "%" PetscInt_FMT " %g\n",
                                     i, (double)PetscRealPart(array[i])); CHKERRQ(ierr);
    }
    ierr = VecRestoreArray(lx, &array); CHKERRQ(ierr);

    ierr = VecGhostUpdateBegin(gx, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
    ierr = VecGhostUpdateEnd(gx, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);

    /* Print out each vector, including the ghost padding region. */
    PetscSynchronizedPrintf(PETSC_COMM_WORLD, "After VecGhostUpdateBegin() rank = %d\n", m_rank);
    ierr = VecGetArray(lx, &array); CHKERRQ(ierr);
    for (i=0; i<nlocal+nghost; i++)
    {
      ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "%" PetscInt_FMT " %g\n",
                                     i, (double)PetscRealPart(array[i])); CHKERRQ(ierr);
      array[i] = array[i] + 10;
    }
    ierr = VecRestoreArray(lx, &array); CHKERRQ(ierr);

    ierr = VecView(gx, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
    ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT); CHKERRQ(ierr);
    ierr = VecGhostRestoreLocalForm(gx, &lx); CHKERRQ(ierr);

    /* Another test that sets ghost values and then accumulates
     * onto the owning processors using MIN_VALUES */
    if (flg3)
    {
        if (m_rank == 0){
            ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,
                   "\nTesting VecGhostUpdate with MIN_VALUES rank = %d\n", m_rank); CHKERRQ(ierr);
        }

        ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "flag3 = 1 rank = %d \n", m_rank); CHKERRQ(ierr);
        ierr = VecGhostGetLocalForm(gx, &lx); CHKERRQ(ierr);
        ierr = VecGetArray(lx, &array); CHKERRQ(ierr);

        for (i=0; i<nghost; i++)
        {
            array[nlocal+i] = m_rank ? (PetscScalar)4 : (PetscScalar)8;
        }
        ierr = VecRestoreArray(lx, &array);CHKERRQ(ierr);
        ierr = VecGhostRestoreLocalForm(gx, &lx);CHKERRQ(ierr);

        ierr = VecGhostUpdateBegin(gx, MIN_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);
        ierr = VecGhostUpdateEnd(gx, MIN_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);

        ierr = VecGhostGetLocalForm(gx, &lx); CHKERRQ(ierr);
        ierr = VecGetArray(lx, &array); CHKERRQ(ierr);

        for (i=0; i<nlocal+nghost; i++) {
            ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "%" PetscInt_FMT " %g\n",
                                           i, (double)PetscRealPart(array[i])); CHKERRQ(ierr);
        }
        ierr = VecRestoreArray(lx, &array); CHKERRQ(ierr);
        ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT); CHKERRQ(ierr);
        ierr = VecGhostRestoreLocalForm(gx, &lx); CHKERRQ(ierr);
    }

    ierr = VecDestroy(&gx); CHKERRQ(ierr);

    if (flg){
        ierr = PetscFree(tarray);CHKERRQ(ierr);
    }
    return ierr;
}

/* ------------------ Vec example ex_11 ------------------ */
PetscErrorCode testPETScVec::testVec_StrideNorm()
{
    Vec            x;
    PetscReal      norm;
    PetscInt       n = 18;
    PetscScalar    one = 1.0;

    ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);

    ierr = VecCreate(PETSC_COMM_WORLD, &x); CHKERRQ(ierr);
    ierr = VecSetSizes(x, PETSC_DECIDE, n); CHKERRQ(ierr);
    ierr = VecSetBlockSize(x, 2); CHKERRQ(ierr);
    ierr = VecSetFromOptions(x); CHKERRQ(ierr);

    ierr = VecSet(x, one); CHKERRQ(ierr);

    ierr = VecGetSize(x, &globalSize); CHKERRQ(ierr);
    ierr = VecGetLocalSize(x, &localSize); CHKERRQ(ierr);

    std::cout << "RANK[" << m_rank << "] localSize = " << localSize
              << ", globalSize = " << globalSize << std::endl;

    //ierr = VecView(x, PETSC_VIEWER_STDERR_WORLD); CHKERRQ(ierr);

    ierr = VecNorm(x, NORM_2, &norm); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "L_2 Norm of entire vector: %g\n", (double)norm); CHKERRQ(ierr);

    ierr = VecNorm(x, NORM_1, &norm);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "L_1 Norm of entire vector: %g\n", (double)norm); CHKERRQ(ierr);

    ierr = VecNorm(x,NORM_INFINITY, &norm); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "L_inf Norm of entire vector: %g\n", (double)norm); CHKERRQ(ierr);

    ierr = VecStrideNorm(x, 0, NORM_2, &norm); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "L_2 Norm of sub-vector 0: %g\n", (double)norm); CHKERRQ(ierr);

    ierr = VecStrideNorm(x, 0, NORM_1, &norm); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "L_1 Norm of sub-vector 0: %g\n", (double)norm); CHKERRQ(ierr);

    ierr = VecStrideNorm(x, 0, NORM_INFINITY, &norm); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "L_inf Norm of sub-vector 0: %g\n", (double)norm); CHKERRQ(ierr);

    ierr = VecStrideNorm(x, 1, NORM_2, &norm); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "L_2 Norm of sub-vector 1: %g\n", (double)norm); CHKERRQ(ierr);

    ierr = VecStrideNorm(x, 1, NORM_1, &norm); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "L_1 Norm of sub-vector 1: %g\n", (double)norm); CHKERRQ(ierr);

    ierr = VecStrideNorm(x, 1, NORM_INFINITY, &norm); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "L_inf Norm of sub-vector 1: %g\n", (double)norm); CHKERRQ(ierr);

    ierr = VecDestroy(&x); CHKERRQ(ierr);
    return ierr;
}

/* ------------------ Vec example ex_12 ------------------ */
PetscErrorCode testPETScVec::testVec_Stride()
{
      Vec            v, s;
      PetscInt       n   = 20;
      PetscScalar    one = 1.0;

      ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);

      /* Form a multi-component vector and initialization */
      ierr = VecCreate(PETSC_COMM_WORLD, &v); CHKERRQ(ierr);
      ierr = VecSetSizes(v, PETSC_DECIDE, n); CHKERRQ(ierr);
      ierr = VecSetBlockSize(v, 2); CHKERRQ(ierr);
      ierr = VecSetFromOptions(v); CHKERRQ(ierr);
      ierr = VecSet(v, one);CHKERRQ(ierr);

      /* Form a single component vector */
      ierr = VecCreate(PETSC_COMM_WORLD, &s); CHKERRQ(ierr);
      ierr = VecSetSizes(s, PETSC_DECIDE, n/2); CHKERRQ(ierr);
      ierr = VecSetFromOptions(s); CHKERRQ(ierr);

      /* send the 0-th element of two multi-component vector to single
       * component vector */
      ierr = VecStrideGather(v, 0, s, INSERT_VALUES); CHKERRQ(ierr);

      ierr = VecView(s, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

      /* send the single component vector to the first element of
       * the two component vector */
      ierr = VecStrideScatter(s, 1, v, ADD_VALUES); CHKERRQ(ierr);

      ierr = VecView(v, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

      ierr = VecDestroy(&v); CHKERRQ(ierr);
      ierr = VecDestroy(&s); CHKERRQ(ierr);
      return ierr;
}

/* ------------------ Vec example ex_16 ------------------ */
PetscErrorCode testPETScVec::testVec_StrideAll()
{
    Vec            v, s, r, vecs[2];
    PetscInt       i, n = 20;
    PetscScalar    value;

    ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);

    /* Create multi-component vector with 2 components */
    ierr = VecCreate(PETSC_COMM_WORLD, &v); CHKERRQ(ierr);
    ierr = VecSetSizes(v, PETSC_DECIDE, n); CHKERRQ(ierr);
    ierr = VecSetBlockSize(v, 4); CHKERRQ(ierr);
    ierr = VecSetFromOptions(v); CHKERRQ(ierr);

    ierr = VecGetSize(v, &globalSize); CHKERRQ(ierr);
    ierr = VecGetLocalSize(v, &localSize); CHKERRQ(ierr);
    std::cout << "RANK[" << m_rank << "] the size of Vec v\nlocalSize = " << localSize
              << ", globalSize = " << globalSize << std::endl;

    /* Create a double-component vector */
    ierr = VecCreate(PETSC_COMM_WORLD, &s); CHKERRQ(ierr);
    ierr = VecSetSizes(s, PETSC_DECIDE, n/2); CHKERRQ(ierr);
    ierr = VecSetBlockSize(s, 2); CHKERRQ(ierr);
    ierr = VecSetFromOptions(s); CHKERRQ(ierr);
    ierr = VecDuplicate(s, &r); CHKERRQ(ierr);

    ierr = VecGetSize(s, &globalSize); CHKERRQ(ierr);
    ierr = VecGetLocalSize(s, &localSize); CHKERRQ(ierr);
    std::cout << "RANK[" << m_rank << "] the size of Vec s\nlocalSize = " << localSize
              << ", globalSize = " << globalSize << std::endl;

    vecs[0] = s;
    vecs[1] = r;

    /* Set the vector values */
    ierr = VecGetOwnershipRange(v, &rstart, &rend); CHKERRQ(ierr);
    std::cout << "RANK[" << m_rank << "], rstart = " << rstart
              << ", rend = " << rend << std::endl;

    for (i=rstart; i<rend; i++)
    {
      value = i;
      ierr  = VecSetValues(v, 1, &i, &value, INSERT_VALUES); CHKERRQ(ierr);
    }

    ierr = PetscPrintf(PETSC_COMM_WORLD, "Print of Vec v\n");
    ierr = VecView(v, PETSC_VIEWER_STDERR_WORLD); CHKERRQ(ierr);

    /* Get the components from the multi-component vector to the other vectors */
    ierr = VecStrideGatherAll(v, vecs, INSERT_VALUES); CHKERRQ(ierr);

    ierr = VecView(s, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
    ierr = VecView(r, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

    ierr = VecStrideScatterAll(vecs, v, ADD_VALUES); CHKERRQ(ierr);

    ierr = VecView(v, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

    Vec singleVec, *singleVecs;
    ierr = VecCreate(PETSC_COMM_WORLD, &singleVec); CHKERRQ(ierr);
    ierr = VecSetSizes(singleVec, PETSC_DECIDE, n/4); CHKERRQ(ierr);
    ierr = VecSetFromOptions(singleVec); CHKERRQ(ierr);
    ierr = VecDuplicateVecs(singleVec, 4, &singleVecs); CHKERRQ(ierr);

    ierr = VecStrideGatherAll(v, singleVecs, INSERT_VALUES); CHKERRQ(ierr);
    ierr = VecView(singleVecs[1], PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

    ierr = VecDestroy(&singleVec); CHKERRQ(ierr);
    ierr = VecDestroyVecs(4, &singleVecs); CHKERRQ(ierr);
    ierr = VecDestroy(&v); CHKERRQ(ierr);
    ierr = VecDestroy(&s); CHKERRQ(ierr);
    ierr = VecDestroy(&r); CHKERRQ(ierr);
    return ierr;
}

/* ------------------ Vec example ex_18 ------------------ */
PetscErrorCode testPETScVec::testVec_CompIntegral()
{
    PetscInt       i, k, N;
    PetscInt       numPoints = 100000000;
    PetscScalar    dummy;
    PetscScalar    result = 0;
    PetscScalar    h=1.0/numPoints;
    PetscScalar    *xarray;
    Vec            x, xend;

    /* Create a parallel vector. */
    ierr   = VecCreate(PETSC_COMM_WORLD, &x); CHKERRQ(ierr);
    ierr   = VecSetSizes(x, PETSC_DECIDE, numPoints); CHKERRQ(ierr);
    ierr   = VecSetFromOptions(x); CHKERRQ(ierr);
    ierr   = VecGetSize(x, &N); CHKERRQ(ierr);
    ierr   = VecSet(x, result); CHKERRQ(ierr);
    ierr   = VecDuplicate(x, &xend);CHKERRQ(ierr);
    std::cout << "RANK[" << m_rank << "], size of the vector x is " << N << std::endl;

    result = 0.5;
    if (m_rank == 0) {
      i    = 0;
      ierr = VecSetValues(xend, 1, &i, &result, INSERT_VALUES); CHKERRQ(ierr);
    }
    if (m_rank == m_size-1) {
      i    = N-1;
      ierr = VecSetValues(xend, 1, &i, &result, INSERT_VALUES); CHKERRQ(ierr);
    }
    ierr = VecAssemblyBegin(xend);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(xend);CHKERRQ(ierr);

    /* Set the x vector elements. */
    ierr = VecGetOwnershipRange(x, &rstart, &rend); CHKERRQ(ierr);
    std::cout << "RANK[" << m_rank << "], rstart = " << rstart
              << ", rend = " << rend << std::endl;
    ierr = VecGetArray(x, &xarray); CHKERRQ(ierr);
    k    = 0;
    for (i=rstart; i<rend; i++, k++) {
      xarray[k] = (PetscScalar)i*h;
      xarray[k] = func(xarray[k]);
    }
    ierr = VecRestoreArray(x, &xarray); CHKERRQ(ierr);

    /* Evaluates the integral. */
    ierr   = VecSum(x, &result); CHKERRQ(ierr);
    result = result*h;
    ierr   = VecDot(x, xend, &dummy); CHKERRQ(ierr);
    result = result - h*dummy;

    /* Return the value of the integral. */
    ierr = PetscPrintf(PETSC_COMM_WORLD, "ln(2) is %g\n",
                       (double)PetscRealPart(result)); CHKERRQ(ierr);
    ierr = VecDestroy(&x); CHKERRQ(ierr);
    ierr = VecDestroy(&xend); CHKERRQ(ierr);

    return ierr;
}

PetscScalar testPETScVec::func(PetscScalar a)
{
    return (PetscScalar)2.*a/((PetscScalar)1. + a*a);
}
