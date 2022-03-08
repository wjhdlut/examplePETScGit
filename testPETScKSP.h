#ifndef TESTPETSCKSP_H
#define TESTPETSCKSP_H

#include<petscksp.h>
#include"testPETSc.h"

typedef enum {
  RHS_FILE,
  RHS_ONE,
  RHS_RANDOM
} RHSType;

const char *const RHSTypes[] = {"FILE", "ONE", "RANDOM", "RHSType", "RHS_", NULL};

PETSC_EXTERN PetscErrorCode PCCreate_Jacobi(PC);

class testPETScKSP : public testPETSc
{
public:
    testPETScKSP();
    testPETScKSP(int rank, int size);
    ~testPETScKSP();

public:
    /* ------ KSP example ex_1 ------ */
    PetscErrorCode testKSP_SolTridiagonalLinearSysSeq();

    /* ------ KSP example ex_2 ------ */
    PetscErrorCode testKSP_SolTridiagonalLinearSysPar();

    /* ------ KSP example ex_3 ------ */
    PetscErrorCode testKSP_Laplacian();

    /* ------ KSP example ex_4 ------ */
    PetscErrorCode testKSP_PCHMG();

    /* ------ KSP example ex_5 ------ */
    PetscErrorCode testKSP_SolTwoLinearSys();

    /* ------ KSP example ex_6 ------ */
    PetscErrorCode testKSP_SolTriLinearKSP();

    /* ------ KSP example ex_7 ------ */
    PetscErrorCode testKSP_BlockJacPC();

    /* ------ KSP example ex_8 ------ */
    PetscErrorCode testKSP_PCASM();

    /* ------ KSP example ex_9 ------ */
    PetscErrorCode testKSP_SolDiffLinSys();

    /* ------ KSP example ex_10 ------ */
    /* reference the profiling character in PETSc Manual*/
    PetscErrorCode testKSP_Preloading();

    /* ------ KSP example ex_12 ------ */
    PetscErrorCode testKSP_RegistNewPC();

    /* ------ KSP example ex_13 ------ */
    PetscErrorCode testKSP_PoissonPro();

    /* ------ KSP example ex_16 ------ */
    PetscErrorCode testKSP_SolDiffRHSKSP();

    /* ------ KSP example ex_18 ------ */
    PetscErrorCode testKSP_SolPErmutedLinearSysKSP();

    /* ------ KSP example ex_46 ------ */
    PetscErrorCode testKSP_SolLinearSysDM();


public:
    Vec            x, b, u;      /* approx solution, RHS, exact solution */
    Mat            A;            /* linear system matrix */
    KSP            ksp;          /* linear solver context */
    PetscInt       Istart, Iend;
    PetscInt       its;
    PC             pc;           /* preconditioner context */
    PetscReal      norm;         /* norm of solution error */
    PetscInt       m, n;         /* mesh dimensions in x- and y- directions */
    PetscInt       bs;

private:
    PetscErrorCode FormElementStiffness(PetscReal H,PetscScalar *Ke);
    PetscErrorCode FormElementRhs(PetscScalar x, PetscScalar y, PetscReal H, PetscScalar *r);
    PetscErrorCode FormMatrixA(PetscScalar v1 = -1.0, PetscScalar v2 = 4.0, InsertMode mode = ADD_VALUES);
    PetscErrorCode FormMatrixA(Mat &C, PetscScalar v1, PetscScalar v2, InsertMode mode);
    PetscErrorCode FormBlockMatrixA(PetscScalar v1, PetscScalar v2, InsertMode mode);
    PetscErrorCode CheckError(const Vec x, const Vec u, const KSP ksp);
};

#endif // TESTPETSCKSP_H
