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
CMAKE_SOURCE_DIR = /media/lzx/lzx/HG/UpdatedPublished

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /media/lzx/lzx/HG/UpdatedPublished/build

# Include any dependencies generated for this target.
include example/CMakeFiles/main.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include example/CMakeFiles/main.dir/compiler_depend.make

# Include the progress variables for this target.
include example/CMakeFiles/main.dir/progress.make

# Include the compile flags for this target's objects.
include example/CMakeFiles/main.dir/flags.make

example/CMakeFiles/main.dir/main.cu.o: example/CMakeFiles/main.dir/flags.make
example/CMakeFiles/main.dir/main.cu.o: example/CMakeFiles/main.dir/includes_CUDA.rsp
example/CMakeFiles/main.dir/main.cu.o: /media/lzx/lzx/HG/UpdatedPublished/example/main.cu
example/CMakeFiles/main.dir/main.cu.o: example/CMakeFiles/main.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/lzx/lzx/HG/UpdatedPublished/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object example/CMakeFiles/main.dir/main.cu.o"
	cd /media/lzx/lzx/HG/UpdatedPublished/build/example && /usr/local/cuda-11.8/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT example/CMakeFiles/main.dir/main.cu.o -MF CMakeFiles/main.dir/main.cu.o.d -x cu -rdc=true -c /media/lzx/lzx/HG/UpdatedPublished/example/main.cu -o CMakeFiles/main.dir/main.cu.o

example/CMakeFiles/main.dir/main.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/main.dir/main.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

example/CMakeFiles/main.dir/main.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/main.dir/main.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target main
main_OBJECTS = \
"CMakeFiles/main.dir/main.cu.o"

# External object files for target main
main_EXTERNAL_OBJECTS =

example/CMakeFiles/main.dir/cmake_device_link.o: example/CMakeFiles/main.dir/main.cu.o
example/CMakeFiles/main.dir/cmake_device_link.o: example/CMakeFiles/main.dir/build.make
example/CMakeFiles/main.dir/cmake_device_link.o: src/libtinympc.a
example/CMakeFiles/main.dir/cmake_device_link.o: /usr/local/cuda-11.8/lib64/libcusparse.so
example/CMakeFiles/main.dir/cmake_device_link.o: /usr/local/cuda-11.8/lib64/libcusolver.so
example/CMakeFiles/main.dir/cmake_device_link.o: example/CMakeFiles/main.dir/deviceLinkLibs.rsp
example/CMakeFiles/main.dir/cmake_device_link.o: example/CMakeFiles/main.dir/deviceObjects1
example/CMakeFiles/main.dir/cmake_device_link.o: example/CMakeFiles/main.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/media/lzx/lzx/HG/UpdatedPublished/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA device code CMakeFiles/main.dir/cmake_device_link.o"
	cd /media/lzx/lzx/HG/UpdatedPublished/build/example && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/main.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
example/CMakeFiles/main.dir/build: example/CMakeFiles/main.dir/cmake_device_link.o
.PHONY : example/CMakeFiles/main.dir/build

# Object files for target main
main_OBJECTS = \
"CMakeFiles/main.dir/main.cu.o"

# External object files for target main
main_EXTERNAL_OBJECTS =

example/main: example/CMakeFiles/main.dir/main.cu.o
example/main: example/CMakeFiles/main.dir/build.make
example/main: src/libtinympc.a
example/main: /usr/local/cuda-11.8/lib64/libcusparse.so
example/main: /usr/local/cuda-11.8/lib64/libcusolver.so
example/main: example/CMakeFiles/main.dir/cmake_device_link.o
example/main: example/CMakeFiles/main.dir/linkLibs.rsp
example/main: example/CMakeFiles/main.dir/objects1
example/main: example/CMakeFiles/main.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/media/lzx/lzx/HG/UpdatedPublished/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CUDA executable main"
	cd /media/lzx/lzx/HG/UpdatedPublished/build/example && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/main.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
example/CMakeFiles/main.dir/build: example/main
.PHONY : example/CMakeFiles/main.dir/build

example/CMakeFiles/main.dir/clean:
	cd /media/lzx/lzx/HG/UpdatedPublished/build/example && $(CMAKE_COMMAND) -P CMakeFiles/main.dir/cmake_clean.cmake
.PHONY : example/CMakeFiles/main.dir/clean

example/CMakeFiles/main.dir/depend:
	cd /media/lzx/lzx/HG/UpdatedPublished/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /media/lzx/lzx/HG/UpdatedPublished /media/lzx/lzx/HG/UpdatedPublished/example /media/lzx/lzx/HG/UpdatedPublished/build /media/lzx/lzx/HG/UpdatedPublished/build/example /media/lzx/lzx/HG/UpdatedPublished/build/example/CMakeFiles/main.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : example/CMakeFiles/main.dir/depend

