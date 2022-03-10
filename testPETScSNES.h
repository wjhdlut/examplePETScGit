#ifndef TESTSNES_H
#define TESTSNES_H

#include<petscsnes.h>
#include "testPETSc.h"

struct MonitorCtx{
    PetscViewer viewer;
};

class testPETScSNES : public testPETSc
{
public:
    testPETScSNES();
    testPETScSNES(int rank, int size);
    ~testPETScSNES();

public:
    SNES           snes;         /* nonlinear solver context */
    KSP            ksp;          /* linear solver context */
    PC             pc;           /* preconditioner context */
    Vec            x, r;         /* solution, residual vectors */
    Mat            J;            /* Jacobian matrix */
    PetscInt       glosize;      /* the global size of vector */

public:
    /* ------ SNES example ex_1 ------ */
    PetscErrorCode testSNES_Sol2VarSysSeq();

    /* ------ SNES example ex_2 ------ */
    PetscErrorCode testSNES_SolNewtonMethSeq();

private:
    static PetscErrorCode FormJacobianEx1_1(SNES snes,Vec x,Mat jac,Mat B,void *dummy);
    static PetscErrorCode FormFunctionEx1_1(SNES snes,Vec x,Vec f,void *ctx);
    static PetscErrorCode FormJacobianEx1_2(SNES snes,Vec x,Mat jac,Mat B,void *dummy);
    static PetscErrorCode FormFunctionEx1_2(SNES snes,Vec x,Vec f,void *dummy);

    static PetscErrorCode FormJacobianEx2(SNES snes,Vec x,Mat jac,Mat B,void *dummy);
    static PetscErrorCode FormFunctionEx2(SNES snes,Vec x,Vec f,void *ctx);
    static PetscErrorCode Monitor(SNES snes,PetscInt its,PetscReal fnorm,void *ctx);
};

#endif // TESTSNES_H
