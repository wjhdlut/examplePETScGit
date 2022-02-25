cmake_minimum_required(VERSION 3.5)

project(examplePETScGit LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#file(GLOB_RECURSE SOURCE_FILEs ${PROJECT_DIR}/*.cpp)

# --------------------------- MPI Pacakage ------------------------------
set(MPI_INC_DIR "/home/wangjianhua/Downloads/PETSc/petsc-main/test/include")
set(MPI_LINK_DIR "/home/wangjianhua/Downloads/PETSc/petsc-main/test/lib")
include_directories(${MPI_INC_DIR})
link_directories(${MPI_LINK_DIR})

# --------------------------- PETSc Pacakage ------------------------------
set(PTESC_INC_DIR "/home/wangjianhua/Downloads/PETSc/petsc-main/include/")
set(PETSC_LINK_DIR "/home/wangjianhua/Downloads/PETSc/petsc-main/lib/petsc")
include_directories(${PTESC_INC_DIR})
link_directories(${PETSC_LINK_DIR})

# ----------------------------- head file ------------------------------
include_directories(${PROJECT_DIR})

#
#ADD_SUBDIRECTORY( src )
AUX_SOURCE_DIRECTORY(. DIR_SRCS)

add_executable(examplePETScGit ${DIR_SRCS})
target_link_libraries(examplePETScGit libmpi.so)
target_link_libraries(examplePETScGit libpetsc.so)
