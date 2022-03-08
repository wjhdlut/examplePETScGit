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
struct AppCtx
{
    PetscInt    k;
    PetscScalar e;
};

enum BCType{DIRICHLET, NEUMANN};

struct UserContext
{
    PetscReal rho;
    PetscReal nu;
    BCType    bcType;
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

    /* ------ KSP example ex_25 ------ */
    PetscErrorCode SolPartialDiffEqu();
    PetscErrorCode SolPartialDiffEqu(int temp);

    /* ------ KSP example ex_28 ------ */
    PetscErrorCode Sol1DWaveEqu();

    /* ------ KSP example ex_29 ------ */
    PetscErrorCode SolInhomoLapl2D();

    /* ------ KSP example ex_30 ------ */
    PetscErrorCode SolInhomoLapl2DEx32();



public:
    PetscInt    m_m, m_n;       /* grid dimensions*/
    PetscInt    m_N;            /* total number of grid */
    UserCtx     m_userCtx;

private:
    PetscErrorCode InitializeLinearSolver(UserCtx *userCtx);
    PetscErrorCode FinalizeLinearSolver(UserCtx *userCtx);

    static PetscErrorCode CompStiffMatrix(KSP ksp, Mat J, Mat jac, void *ctx);
    static PetscErrorCode CompRHS(KSP ksp, Vec b, void *ctx);

    /* member function for example 28 */
    static PetscErrorCode CompRHSEx28(KSP ksp,Vec b,void *ctx);
    static PetscErrorCode CompStiffMatrixEx28(KSP ksp, Mat J, Mat jac, void *ctx);
    PetscErrorCode CompInitialSolution(DM da, Vec x);

    /* member function for example 29 */
    static PetscErrorCode CompRHSEx29(KSP ksp, Vec b, void *ctx);
    static PetscErrorCode CompStiffMatrixEx29(KSP ksp, Mat J, Mat jac, void *ctx);

    /* member function for example 32 */
    static PetscErrorCode CompRHSEx32(KSP ksp,Vec b,void *ctx);
    static PetscErrorCode CompStiffMatrixEx32(KSP ksp, Mat J,Mat jac, void *ctx);

public:
    static applicationExamples *m_aEP;
};

#endif // APPLICATIONEXAMPLES_H
