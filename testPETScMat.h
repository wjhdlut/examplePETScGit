#ifndef TESTPETSCMAT_H
#define TESTPETSCMAT_H

#include"testPETSc.h"
#include"petscmat.h"

class testPETScMat : public testPETSc
{
public:
    testPETScMat();
    testPETScMat(int rank, int size);
    ~testPETScMat();

public:
    /* ------ Mat example ex_2 ------ */
    PetscErrorCode testMat_SeqDense();

    /* ------ Mat example ex_3 ------ */
    PetscErrorCode testMat_1DLapla();

    /* ------ Mat example ex_4 ------ */
    PetscErrorCode testMat_ResetPreallocation();

    /* ------ Mat example ex_7 ------ */
    PetscErrorCode testMat_PetscInfo();

public:

};
#endif // TESTPETSCMAT_H
