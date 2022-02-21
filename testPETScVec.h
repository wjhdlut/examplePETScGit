#ifndef TESTPETSCVEC_H
#define TESTPETSCVEC_H

#include <petscksp.h>
#include "testPETSc.h"

class testPETScVec : public testPETSc
{
public:
    testPETScVec();
    testPETScVec(int rank, int size);
    ~testPETScVec();

public:
    /* ------ Vec example ex_1 ------ */
    PetscErrorCode testVec_CreateBasicOperation();

    /* ------ Vec example ex_2 ------ */
    PetscErrorCode testVec_VecSetValues();

    /* ------ Vec example ex_3 ------ */
    PetscErrorCode testVec_ex3();

    /* ------ Vec example ex_5 ------ */
    PetscErrorCode testVec_Print();

    /* ------ Vec example ex_8 ------ */
    PetscErrorCode testVec_LocalToGlobalIndex();

    /* ------ Vec example ex_9 ------ */
    PetscErrorCode testVec_CreateGhost();

    /* ------ Vec example ex_11 ------ */
    PetscErrorCode testVec_StrideNorm();

    /* ------ Vec example ex_12 ------ */
    PetscErrorCode testVec_Stride();

    /* ------ Vec example ex_16 ------ */
    PetscErrorCode testVec_StrideAll();

    /* ------ Vec example ex_18 ------ */
    PetscErrorCode testVec_CompIntegral();

private:
    PetscScalar func(PetscScalar a);

private:
    PetscInt        localSize;
    PetscInt        globalSize;
    PetscInt        rstart;
    PetscInt        rend;
};



#endif // TESTPETSCVEC_H
