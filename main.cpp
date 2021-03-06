#include<mpi.h>
#include"testPETScVec.h"
#include"testPETScMat.h"
#include"testPETScKSP.h"
#include"applicationExamples.h"
#include"testPETScSNES.h"

static char help[] = "Solves a linear system in parallel with KSP.\n";

PetscErrorCode main(int argc,char **args)
{
    PetscErrorCode ierr;
    PetscInt       rank, size;
    ierr = PetscInitialize(&argc, &args, (char*)0, help);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* test for the testPETScVec */
    //testPETScVec tp(rank, size);
    //tp.testVec_CreateBasicOperation();
    //tp.testVec_VecSetValues();
    //tp.testVec_ex3();
    //tp.testVec_Print();
    //tp.testVec_LocalToGlobalIndex();
    //tp.testVec_CreateGhost();
    //tp.testVec_StrideNorm();
    //tp.testVec_Stride();
    //tp.testVec_StrideAll();
    //tp.testVec_CompIntegral();

    //tp.simpleKSP();

    /* test for the testPETScMat */
    //testPETScMat tpM(rank, size);
    //tpM.testMat_SeqDense();
    //tpM.testMat_1DLapla();
    //tpM.testMat_ResetPreallocation();
    //tpM.testMat_PetscInfo();

    /* test for the testPETScKSP */
    //testPETScKSP tpKSP(rank, size);
    //tpKSP.testKSP_SolTridiagonalLinearSysSeq();
    //tpKSP.testKSP_SolTridiagonalLinearSysPar();
    //tpKSP.testKSP_Laplacian();
    //tpKSP.testKSP_PCHMG();
    //tpKSP.testKSP_SolTwoLinearSys();
    //tpKSP.testKSP_SolTriLinearKSP();
    //tpKSP.testKSP_BlockJacPC();
    //tpKSP.testKSP_PCASM();
    //tpKSP.testKSP_SolDiffLinSys();
    //tpKSP.testKSP_Preloading();
    //tpKSP.testKSP_RegistNewPC();
    //tpKSP.testKSP_SolDiffRHSKSP();
    //tpKSP.testKSP_SolPErmutedLinearSysKSP();

    /* test for the application examples */
    //applicationExamples aE(rank, size);
    //aE.SolPoissonProblemKSP();
    //aE.SolPartialDiffEqu();
    //aE.SolPartialDiffEqu(0);

    testPETScSNES tpSNES(rank, size);
    //tpSNES.testSNES_Sol2VarSysSeq();
    tpSNES.testSNES_SolNewtonMethSeq();

    ierr = PetscFinalize();

    return ierr;
}
