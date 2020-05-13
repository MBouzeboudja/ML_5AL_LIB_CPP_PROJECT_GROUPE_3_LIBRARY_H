cmake_minimum_required(VERSION 3.16)
project(ML_5AL_lib_cpp_project_groupe_3)

set(EIGEN_DIR "/home/bouzeboudja/workspace/5AL/ML/eigen-3.3.7/Eigen")
include_directories(${EIGEN_DIR})
set(CMAKE_CXX_STANDARD 20)
add_library(ML_5AL_lib_cpp_project_groupe_3 SHARED library.cpp library.h utils.cpp utils.h)
target_link_libraries (ML_5AL_lib_cpp_project_groupe_3 ${EIGEN_DIR})
