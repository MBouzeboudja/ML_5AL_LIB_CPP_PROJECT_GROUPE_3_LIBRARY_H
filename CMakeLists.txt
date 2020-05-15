cmake_minimum_required(VERSION 3.16)
project(ML_5AL_lib_cpp_project_groupe_3)
set(CMAKE_CXX_STANDARD 20)
add_library(ML_5AL_lib_cpp_project_groupe_3 SHARED library.cpp library.h utils.cpp utils.h)

#Mahdi
set(EIGEN_DIR "D:/Users/Arthur/Travail2020/ML/eigen")
include_directories(${EIGEN_DIR})
target_link_libraries (ML_5AL_lib_cpp_project_groupe_3 ${EIGEN_DIR})


add_subdirectory(csv-parser)
add_executable(appExec main.cpp library.cpp library.h utils.cpp utils.h)
target_link_libraries (appExec ${EIGEN_DIR} csv)