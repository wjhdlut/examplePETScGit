#ifndef COMMONFUNCTIONS_H
#define COMMONFUNCTIONS_H

#include"testPETSc.h"

PetscErrorCode CheckError(Vec xx, Vec uu, KSP kspp)
{
    /*
      Input Parameter
      xx   -- approx solution
      uu   -- exact solution
      kspp -- linear solver context
    */
    PetscInt       its;
    PetscScalar    norm;
    PetscErrorCode ierr;

    ierr = VecAXPY(xx, -1.0, uu); CHKERRQ(ierr);
    ierr = VecNorm(xx, NORM_2, &norm); CHKERRQ(ierr);
    ierr = KSPGetIterationNumber(kspp, &its); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Norm of error %g iterations %D\n",
                       (double)norm, its); CHKERRQ(ierr);

    return ierr;
}
#endif // COMMONFUNCTIONS_H
