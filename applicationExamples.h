#ifndef APPLICATIONEXAMPLES_H
#define APPLICATIONEXAMPLES_H

#include"testPETSc.h"

struct UserCtx
{
    Mat         A;            /* stiffness matrix */
    Vec         x, b;         /* approximate solution vector, right-hand-side vector */
    KSP         ksp;          /* linear solver context*/
    PetscScalar hx2, hy2;     /* 1/(m+1)*(m+1) and 1/(n+1)*(n+1) */
    PC          pc;
};

class applicationExamples : public testPETSc
{
public:
    applicationExamples();
    applicationExamples(int rank, int size);
    ~applicationExamples();

public:
    /* ------ KSP example ex_13 ------ */
    PetscErrorCode SolPoissonProblemKSP();


public:
    PetscInt    m_m, m_n;       /* grid dimensions*/
    PetscInt    m_N;            /* total number of grid */
    UserCtx     m_userCtx;

private:
    PetscErrorCode InitializeLinearSolver(UserCtx *userCtx);
    PetscErrorCode FinalizeLinearSolver(UserCtx *userCtx);
};

#endif // APPLICATIONEXAMPLES_H
