# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.25

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/lzx/anaconda3/lib/python3.11/site-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /home/lzx/anaconda3/lib/python3.11/site-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /media/lzx/lzx/HG/OptimizeSparse

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /media/lzx/lzx/HG/OptimizeSparse/build

# Include any dependencies generated for this target.
include src/CMakeFiles/tinympc.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include src/CMakeFiles/tinympc.dir/compiler_depend.make

# Include the progress variables for this target.
include src/CMakeFiles/tinympc.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/tinympc.dir/flags.make

src/CMakeFiles/tinympc.dir/DeviceFunction.cu.o: src/CMakeFiles/tinympc.dir/flags.make
src/CMakeFiles/tinympc.dir/DeviceFunction.cu.o: src/CMakeFiles/tinympc.dir/includes_CUDA.rsp
src/CMakeFiles/tinympc.dir/DeviceFunction.cu.o: /media/lzx/lzx/HG/OptimizeSparse/src/DeviceFunction.cu
src/CMakeFiles/tinympc.dir/DeviceFunction.cu.o: src/CMakeFiles/tinympc.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/lzx/lzx/HG/OptimizeSparse/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object src/CMakeFiles/tinympc.dir/DeviceFunction.cu.o"
	cd /media/lzx/lzx/HG/OptimizeSparse/build/src && /usr/local/cuda-11.8/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT src/CMakeFiles/tinympc.dir/DeviceFunction.cu.o -MF CMakeFiles/tinympc.dir/DeviceFunction.cu.o.d -x cu -c /media/lzx/lzx/HG/OptimizeSparse/src/DeviceFunction.cu -o CMakeFiles/tinympc.dir/DeviceFunction.cu.o

src/CMakeFiles/tinympc.dir/DeviceFunction.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/tinympc.dir/DeviceFunction.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

src/CMakeFiles/tinympc.dir/DeviceFunction.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/tinympc.dir/DeviceFunction.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

src/CMakeFiles/tinympc.dir/head.cu.o: src/CMakeFiles/tinympc.dir/flags.make
src/CMakeFiles/tinympc.dir/head.cu.o: src/CMakeFiles/tinympc.dir/includes_CUDA.rsp
src/CMakeFiles/tinympc.dir/head.cu.o: /media/lzx/lzx/HG/OptimizeSparse/src/head.cu
src/CMakeFiles/tinympc.dir/head.cu.o: src/CMakeFiles/tinympc.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/lzx/lzx/HG/OptimizeSparse/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CUDA object src/CMakeFiles/tinympc.dir/head.cu.o"
	cd /media/lzx/lzx/HG/OptimizeSparse/build/src && /usr/local/cuda-11.8/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT src/CMakeFiles/tinympc.dir/head.cu.o -MF CMakeFiles/tinympc.dir/head.cu.o.d -x cu -c /media/lzx/lzx/HG/OptimizeSparse/src/head.cu -o CMakeFiles/tinympc.dir/head.cu.o

src/CMakeFiles/tinympc.dir/head.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/tinympc.dir/head.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

src/CMakeFiles/tinympc.dir/head.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/tinympc.dir/head.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target tinympc
tinympc_OBJECTS = \
"CMakeFiles/tinympc.dir/DeviceFunction.cu.o" \
"CMakeFiles/tinympc.dir/head.cu.o"

# External object files for target tinympc
tinympc_EXTERNAL_OBJECTS =

src/libtinympc.a: src/CMakeFiles/tinympc.dir/DeviceFunction.cu.o
src/libtinympc.a: src/CMakeFiles/tinympc.dir/head.cu.o
src/libtinympc.a: src/CMakeFiles/tinympc.dir/build.make
src/libtinympc.a: src/CMakeFiles/tinympc.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/media/lzx/lzx/HG/OptimizeSparse/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CUDA static library libtinympc.a"
	cd /media/lzx/lzx/HG/OptimizeSparse/build/src && $(CMAKE_COMMAND) -P CMakeFiles/tinympc.dir/cmake_clean_target.cmake
	cd /media/lzx/lzx/HG/OptimizeSparse/build/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/tinympc.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/tinympc.dir/build: src/libtinympc.a
.PHONY : src/CMakeFiles/tinympc.dir/build

src/CMakeFiles/tinympc.dir/clean:
	cd /media/lzx/lzx/HG/OptimizeSparse/build/src && $(CMAKE_COMMAND) -P CMakeFiles/tinympc.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/tinympc.dir/clean

src/CMakeFiles/tinympc.dir/depend:
	cd /media/lzx/lzx/HG/OptimizeSparse/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /media/lzx/lzx/HG/OptimizeSparse /media/lzx/lzx/HG/OptimizeSparse/src /media/lzx/lzx/HG/OptimizeSparse/build /media/lzx/lzx/HG/OptimizeSparse/build/src /media/lzx/lzx/HG/OptimizeSparse/build/src/CMakeFiles/tinympc.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/tinympc.dir/depend

