add_executable(main main.cu)
target_link_libraries(main LINK_PUBLIC tinympc)
# target_compile_options(main PRIVATE -w)  # For GCC/Clang

add_executable(test test.cu)
target_link_libraries(test LINK_PUBLIC tinympc)