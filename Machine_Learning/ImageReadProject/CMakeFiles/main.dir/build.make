# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.9

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/harry/C++/Machine_Learning/ImageReadProject

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/harry/C++/Machine_Learning/ImageReadProject

# Include any dependencies generated for this target.
include CMakeFiles/main.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/main.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/main.dir/flags.make

CMakeFiles/main.dir/src/Classification.cpp.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/src/Classification.cpp.o: src/Classification.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/harry/C++/Machine_Learning/ImageReadProject/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/main.dir/src/Classification.cpp.o"
	/usr/sbin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/main.dir/src/Classification.cpp.o -c /home/harry/C++/Machine_Learning/ImageReadProject/src/Classification.cpp

CMakeFiles/main.dir/src/Classification.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/main.dir/src/Classification.cpp.i"
	/usr/sbin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/harry/C++/Machine_Learning/ImageReadProject/src/Classification.cpp > CMakeFiles/main.dir/src/Classification.cpp.i

CMakeFiles/main.dir/src/Classification.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/main.dir/src/Classification.cpp.s"
	/usr/sbin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/harry/C++/Machine_Learning/ImageReadProject/src/Classification.cpp -o CMakeFiles/main.dir/src/Classification.cpp.s

CMakeFiles/main.dir/src/Classification.cpp.o.requires:

.PHONY : CMakeFiles/main.dir/src/Classification.cpp.o.requires

CMakeFiles/main.dir/src/Classification.cpp.o.provides: CMakeFiles/main.dir/src/Classification.cpp.o.requires
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/src/Classification.cpp.o.provides.build
.PHONY : CMakeFiles/main.dir/src/Classification.cpp.o.provides

CMakeFiles/main.dir/src/Classification.cpp.o.provides.build: CMakeFiles/main.dir/src/Classification.cpp.o


CMakeFiles/main.dir/src/Source.cpp.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/src/Source.cpp.o: src/Source.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/harry/C++/Machine_Learning/ImageReadProject/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/main.dir/src/Source.cpp.o"
	/usr/sbin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/main.dir/src/Source.cpp.o -c /home/harry/C++/Machine_Learning/ImageReadProject/src/Source.cpp

CMakeFiles/main.dir/src/Source.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/main.dir/src/Source.cpp.i"
	/usr/sbin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/harry/C++/Machine_Learning/ImageReadProject/src/Source.cpp > CMakeFiles/main.dir/src/Source.cpp.i

CMakeFiles/main.dir/src/Source.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/main.dir/src/Source.cpp.s"
	/usr/sbin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/harry/C++/Machine_Learning/ImageReadProject/src/Source.cpp -o CMakeFiles/main.dir/src/Source.cpp.s

CMakeFiles/main.dir/src/Source.cpp.o.requires:

.PHONY : CMakeFiles/main.dir/src/Source.cpp.o.requires

CMakeFiles/main.dir/src/Source.cpp.o.provides: CMakeFiles/main.dir/src/Source.cpp.o.requires
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/src/Source.cpp.o.provides.build
.PHONY : CMakeFiles/main.dir/src/Source.cpp.o.provides

CMakeFiles/main.dir/src/Source.cpp.o.provides.build: CMakeFiles/main.dir/src/Source.cpp.o


CMakeFiles/main.dir/src/mnist_dataset_reader.cpp.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/src/mnist_dataset_reader.cpp.o: src/mnist_dataset_reader.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/harry/C++/Machine_Learning/ImageReadProject/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/main.dir/src/mnist_dataset_reader.cpp.o"
	/usr/sbin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/main.dir/src/mnist_dataset_reader.cpp.o -c /home/harry/C++/Machine_Learning/ImageReadProject/src/mnist_dataset_reader.cpp

CMakeFiles/main.dir/src/mnist_dataset_reader.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/main.dir/src/mnist_dataset_reader.cpp.i"
	/usr/sbin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/harry/C++/Machine_Learning/ImageReadProject/src/mnist_dataset_reader.cpp > CMakeFiles/main.dir/src/mnist_dataset_reader.cpp.i

CMakeFiles/main.dir/src/mnist_dataset_reader.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/main.dir/src/mnist_dataset_reader.cpp.s"
	/usr/sbin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/harry/C++/Machine_Learning/ImageReadProject/src/mnist_dataset_reader.cpp -o CMakeFiles/main.dir/src/mnist_dataset_reader.cpp.s

CMakeFiles/main.dir/src/mnist_dataset_reader.cpp.o.requires:

.PHONY : CMakeFiles/main.dir/src/mnist_dataset_reader.cpp.o.requires

CMakeFiles/main.dir/src/mnist_dataset_reader.cpp.o.provides: CMakeFiles/main.dir/src/mnist_dataset_reader.cpp.o.requires
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/src/mnist_dataset_reader.cpp.o.provides.build
.PHONY : CMakeFiles/main.dir/src/mnist_dataset_reader.cpp.o.provides

CMakeFiles/main.dir/src/mnist_dataset_reader.cpp.o.provides.build: CMakeFiles/main.dir/src/mnist_dataset_reader.cpp.o


# Object files for target main
main_OBJECTS = \
"CMakeFiles/main.dir/src/Classification.cpp.o" \
"CMakeFiles/main.dir/src/Source.cpp.o" \
"CMakeFiles/main.dir/src/mnist_dataset_reader.cpp.o"

# External object files for target main
main_EXTERNAL_OBJECTS =

main: CMakeFiles/main.dir/src/Classification.cpp.o
main: CMakeFiles/main.dir/src/Source.cpp.o
main: CMakeFiles/main.dir/src/mnist_dataset_reader.cpp.o
main: CMakeFiles/main.dir/build.make
main: /usr/lib/libopencv_videostab.so.2.4.13
main: /usr/lib/libopencv_ts.a
main: /usr/lib/libopencv_superres.so.2.4.13
main: /usr/lib/libopencv_stitching.so.2.4.13
main: /usr/lib/libopencv_contrib.so.2.4.13
main: /lib64/libGLU.so
main: /lib64/libGL.so
main: /usr/lib/libopencv_nonfree.so.2.4.13
main: /usr/lib/libopencv_ocl.so.2.4.13
main: /usr/lib/libopencv_gpu.so.2.4.13
main: /usr/lib/libopencv_photo.so.2.4.13
main: /usr/lib/libopencv_objdetect.so.2.4.13
main: /usr/lib/libopencv_legacy.so.2.4.13
main: /usr/lib/libopencv_video.so.2.4.13
main: /usr/lib/libopencv_ml.so.2.4.13
main: /usr/lib/libopencv_calib3d.so.2.4.13
main: /usr/lib/libopencv_features2d.so.2.4.13
main: /usr/lib/libopencv_highgui.so.2.4.13
main: /usr/lib/libopencv_imgproc.so.2.4.13
main: /usr/lib/libopencv_flann.so.2.4.13
main: /usr/lib/libopencv_core.so.2.4.13
main: CMakeFiles/main.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/harry/C++/Machine_Learning/ImageReadProject/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable main"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/main.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/main.dir/build: main

.PHONY : CMakeFiles/main.dir/build

CMakeFiles/main.dir/requires: CMakeFiles/main.dir/src/Classification.cpp.o.requires
CMakeFiles/main.dir/requires: CMakeFiles/main.dir/src/Source.cpp.o.requires
CMakeFiles/main.dir/requires: CMakeFiles/main.dir/src/mnist_dataset_reader.cpp.o.requires

.PHONY : CMakeFiles/main.dir/requires

CMakeFiles/main.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/main.dir/cmake_clean.cmake
.PHONY : CMakeFiles/main.dir/clean

CMakeFiles/main.dir/depend:
	cd /home/harry/C++/Machine_Learning/ImageReadProject && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/harry/C++/Machine_Learning/ImageReadProject /home/harry/C++/Machine_Learning/ImageReadProject /home/harry/C++/Machine_Learning/ImageReadProject /home/harry/C++/Machine_Learning/ImageReadProject /home/harry/C++/Machine_Learning/ImageReadProject/CMakeFiles/main.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/main.dir/depend

