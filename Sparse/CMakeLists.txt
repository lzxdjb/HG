cmake_minimum_required(VERSION 3.0.0)
project(TinyMPC VERSION 0.2.0 LANGUAGES CUDA CXX)

# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -ftree-vectorize")

# set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -O3  -Wall")
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3  -Wall")

set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-11.8)


include_directories("/usr/include/eigen3")

add_subdirectory(src)
add_subdirectory(example)




# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -w")
