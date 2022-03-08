#ifndef TESTSNES_H
#define TESTSNES_H

#include "testPETSc.h"

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

public:
    /* ------ SNES example ex_1 ------ */
    PetscErrorCode testSNES_Sol2VarSysSeq();

private:
    static PetscErrorCode FormJacobianEx1_1(SNES snes,Vec x,Mat jac,Mat B,void *dummy);
    static PetscErrorCode FormFunctionEx1_1(SNES snes,Vec x,Vec f,void *ctx);
    static PetscErrorCode FormJacobianEx1_2(SNES snes,Vec x,Mat jac,Mat B,void *dummy);
    static PetscErrorCode FormFunctionEx1_2(SNES snes,Vec x,Vec f,void *dummy);

};

#endif // TESTSNES_H
