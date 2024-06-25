add_library(tinympc STATIC
    head.cu
    # codegen.cpp
)

target_include_directories(tinympc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..) 
