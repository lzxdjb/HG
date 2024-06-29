add_executable(main main.cu)


find_library(CUSOLVER_LIBRARY cusolver PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
if(NOT CUSOLVER_LIBRARY)
    message(FATAL_ERROR "cuSOLVER library not found")
endif()

target_link_libraries(main LINK_PUBLIC tinympc ${CUSPARSE_LIBRARY} ${CUSOLVER_LIBRARY})

set_target_properties(main PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)



add_executable(emplate template.cu)

find_library(CUSPARSE_LIBRARY cusparse PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
if(NOT CUSPARSE_LIBRARY)
    message(FATAL_ERROR "cuSPARSE library not found")
endif()

target_link_libraries(emplate LINK_PUBLIC tinympc ${CUSPARSE_LIBRARY})


# # Find the cuSOLVER library
# add_executable(stackoverflow stackoverflow.cu)

# find_library(CUSOLVER_LIBRARY cusolver PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
# if(NOT CUSOLVER_LIBRARY)
#     message(FATAL_ERROR "cuSOLVER library not found")
# endif()

# # Link the libraries
# target_link_libraries(stackoverflow LINK_PUBLIC tinympc ${CUSPARSE_LIBRARY} ${CUSOLVER_LIBRARY})



# add_executable(test test.cu)


# find_library(CUSOLVER_LIBRARY cusolver PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
# if(NOT CUSOLVER_LIBRARY)
#     message(FATAL_ERROR "cuSOLVER library not found")
# endif()

# target_link_libraries(test PRIVATE tinympc ${CUSPARSE_LIBRARY} ${CUSOLVER_LIBRARY})

# set_target_properties(test PROPERTIES
#     CUDA_SEPARABLE_COMPILATION ON
# )


