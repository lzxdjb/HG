cmake_minimum_required(VERSION 3.10)

# Set the project name
project(NewtonMethodProject)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add the newton library
add_library(newton STATIC newton/newton_method.cpp)

# Add the executable
add_executable(NewtonMethod main.cpp)

# Specify the path to Eigen (adjust this path to where Eigen is located on your system)
set(EIGEN3_INCLUDE_DIR "/usr/include/eigen3")

# Include Eigen3 directory
include_directories(${EIGEN3_INCLUDE_DIR})

# Link the executable to the newton library
target_link_libraries(NewtonMethod newton)

# Include directories for newton library
target_include_directories(newton PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/newton)
target_include_directories(NewtonMethod PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/newton)
