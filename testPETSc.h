#ifndef TESTPETSC_H
#define TESTPETSC_H

#include<iostream>
#include"petsc.h"

#define RANK_COUT std::cout << "RANK[" << m_rank << "], "

class testPETSc
{
public:
    PetscErrorCode  ierr;
    int             m_rank;
    int             m_size;
};

#endif // TESTPETSC_H
