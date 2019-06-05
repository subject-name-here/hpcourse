cmake_minimum_required(VERSION 3.13)
project(hw4)

set(CMAKE_CXX_STANDARD 14)

find_package (OpenCL REQUIRED)

if (${OPENCL_FOUND})
    include_directories (${OpenCL_INCLUDE_DIR})
    add_executable(hw4 main.cpp)
    target_link_libraries (hw4 LINK_PRIVATE ${OpenCL_LIBRARY})
endif()