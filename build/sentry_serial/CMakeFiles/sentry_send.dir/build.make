# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

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
CMAKE_SOURCE_DIR = /home/zhang/下载/RoboDetect/src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/zhang/下载/RoboDetect/build

# Include any dependencies generated for this target.
include sentry_serial/CMakeFiles/sentry_send.dir/depend.make

# Include the progress variables for this target.
include sentry_serial/CMakeFiles/sentry_send.dir/progress.make

# Include the compile flags for this target's objects.
include sentry_serial/CMakeFiles/sentry_send.dir/flags.make

sentry_serial/CMakeFiles/sentry_send.dir/src/serial_device.cpp.o: sentry_serial/CMakeFiles/sentry_send.dir/flags.make
sentry_serial/CMakeFiles/sentry_send.dir/src/serial_device.cpp.o: /home/zhang/下载/RoboDetect/src/sentry_serial/src/serial_device.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zhang/下载/RoboDetect/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object sentry_serial/CMakeFiles/sentry_send.dir/src/serial_device.cpp.o"
	cd /home/zhang/下载/RoboDetect/build/sentry_serial && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/sentry_send.dir/src/serial_device.cpp.o -c /home/zhang/下载/RoboDetect/src/sentry_serial/src/serial_device.cpp

sentry_serial/CMakeFiles/sentry_send.dir/src/serial_device.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sentry_send.dir/src/serial_device.cpp.i"
	cd /home/zhang/下载/RoboDetect/build/sentry_serial && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zhang/下载/RoboDetect/src/sentry_serial/src/serial_device.cpp > CMakeFiles/sentry_send.dir/src/serial_device.cpp.i

sentry_serial/CMakeFiles/sentry_send.dir/src/serial_device.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sentry_send.dir/src/serial_device.cpp.s"
	cd /home/zhang/下载/RoboDetect/build/sentry_serial && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zhang/下载/RoboDetect/src/sentry_serial/src/serial_device.cpp -o CMakeFiles/sentry_send.dir/src/serial_device.cpp.s

sentry_serial/CMakeFiles/sentry_send.dir/src/serial_port.cpp.o: sentry_serial/CMakeFiles/sentry_send.dir/flags.make
sentry_serial/CMakeFiles/sentry_send.dir/src/serial_port.cpp.o: /home/zhang/下载/RoboDetect/src/sentry_serial/src/serial_port.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zhang/下载/RoboDetect/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object sentry_serial/CMakeFiles/sentry_send.dir/src/serial_port.cpp.o"
	cd /home/zhang/下载/RoboDetect/build/sentry_serial && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/sentry_send.dir/src/serial_port.cpp.o -c /home/zhang/下载/RoboDetect/src/sentry_serial/src/serial_port.cpp

sentry_serial/CMakeFiles/sentry_send.dir/src/serial_port.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sentry_send.dir/src/serial_port.cpp.i"
	cd /home/zhang/下载/RoboDetect/build/sentry_serial && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zhang/下载/RoboDetect/src/sentry_serial/src/serial_port.cpp > CMakeFiles/sentry_send.dir/src/serial_port.cpp.i

sentry_serial/CMakeFiles/sentry_send.dir/src/serial_port.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sentry_send.dir/src/serial_port.cpp.s"
	cd /home/zhang/下载/RoboDetect/build/sentry_serial && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zhang/下载/RoboDetect/src/sentry_serial/src/serial_port.cpp -o CMakeFiles/sentry_send.dir/src/serial_port.cpp.s

sentry_serial/CMakeFiles/sentry_send.dir/src/crc.cpp.o: sentry_serial/CMakeFiles/sentry_send.dir/flags.make
sentry_serial/CMakeFiles/sentry_send.dir/src/crc.cpp.o: /home/zhang/下载/RoboDetect/src/sentry_serial/src/crc.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zhang/下载/RoboDetect/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object sentry_serial/CMakeFiles/sentry_send.dir/src/crc.cpp.o"
	cd /home/zhang/下载/RoboDetect/build/sentry_serial && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/sentry_send.dir/src/crc.cpp.o -c /home/zhang/下载/RoboDetect/src/sentry_serial/src/crc.cpp

sentry_serial/CMakeFiles/sentry_send.dir/src/crc.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sentry_send.dir/src/crc.cpp.i"
	cd /home/zhang/下载/RoboDetect/build/sentry_serial && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zhang/下载/RoboDetect/src/sentry_serial/src/crc.cpp > CMakeFiles/sentry_send.dir/src/crc.cpp.i

sentry_serial/CMakeFiles/sentry_send.dir/src/crc.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sentry_send.dir/src/crc.cpp.s"
	cd /home/zhang/下载/RoboDetect/build/sentry_serial && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zhang/下载/RoboDetect/src/sentry_serial/src/crc.cpp -o CMakeFiles/sentry_send.dir/src/crc.cpp.s

sentry_serial/CMakeFiles/sentry_send.dir/src/serial_test.cpp.o: sentry_serial/CMakeFiles/sentry_send.dir/flags.make
sentry_serial/CMakeFiles/sentry_send.dir/src/serial_test.cpp.o: /home/zhang/下载/RoboDetect/src/sentry_serial/src/serial_test.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zhang/下载/RoboDetect/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object sentry_serial/CMakeFiles/sentry_send.dir/src/serial_test.cpp.o"
	cd /home/zhang/下载/RoboDetect/build/sentry_serial && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/sentry_send.dir/src/serial_test.cpp.o -c /home/zhang/下载/RoboDetect/src/sentry_serial/src/serial_test.cpp

sentry_serial/CMakeFiles/sentry_send.dir/src/serial_test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sentry_send.dir/src/serial_test.cpp.i"
	cd /home/zhang/下载/RoboDetect/build/sentry_serial && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zhang/下载/RoboDetect/src/sentry_serial/src/serial_test.cpp > CMakeFiles/sentry_send.dir/src/serial_test.cpp.i

sentry_serial/CMakeFiles/sentry_send.dir/src/serial_test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sentry_send.dir/src/serial_test.cpp.s"
	cd /home/zhang/下载/RoboDetect/build/sentry_serial && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zhang/下载/RoboDetect/src/sentry_serial/src/serial_test.cpp -o CMakeFiles/sentry_send.dir/src/serial_test.cpp.s

# Object files for target sentry_send
sentry_send_OBJECTS = \
"CMakeFiles/sentry_send.dir/src/serial_device.cpp.o" \
"CMakeFiles/sentry_send.dir/src/serial_port.cpp.o" \
"CMakeFiles/sentry_send.dir/src/crc.cpp.o" \
"CMakeFiles/sentry_send.dir/src/serial_test.cpp.o"

# External object files for target sentry_send
sentry_send_EXTERNAL_OBJECTS =

/home/zhang/下载/RoboDetect/devel/lib/sentry_serial/sentry_send: sentry_serial/CMakeFiles/sentry_send.dir/src/serial_device.cpp.o
/home/zhang/下载/RoboDetect/devel/lib/sentry_serial/sentry_send: sentry_serial/CMakeFiles/sentry_send.dir/src/serial_port.cpp.o
/home/zhang/下载/RoboDetect/devel/lib/sentry_serial/sentry_send: sentry_serial/CMakeFiles/sentry_send.dir/src/crc.cpp.o
/home/zhang/下载/RoboDetect/devel/lib/sentry_serial/sentry_send: sentry_serial/CMakeFiles/sentry_send.dir/src/serial_test.cpp.o
/home/zhang/下载/RoboDetect/devel/lib/sentry_serial/sentry_send: sentry_serial/CMakeFiles/sentry_send.dir/build.make
/home/zhang/下载/RoboDetect/devel/lib/sentry_serial/sentry_send: /opt/ros/noetic/lib/libserial.so
/home/zhang/下载/RoboDetect/devel/lib/sentry_serial/sentry_send: /opt/ros/noetic/lib/libroscpp.so
/home/zhang/下载/RoboDetect/devel/lib/sentry_serial/sentry_send: /usr/lib/x86_64-linux-gnu/libpthread.so
/home/zhang/下载/RoboDetect/devel/lib/sentry_serial/sentry_send: /usr/lib/x86_64-linux-gnu/libboost_chrono.so.1.71.0
/home/zhang/下载/RoboDetect/devel/lib/sentry_serial/sentry_send: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so.1.71.0
/home/zhang/下载/RoboDetect/devel/lib/sentry_serial/sentry_send: /opt/ros/noetic/lib/librosconsole.so
/home/zhang/下载/RoboDetect/devel/lib/sentry_serial/sentry_send: /opt/ros/noetic/lib/librosconsole_log4cxx.so
/home/zhang/下载/RoboDetect/devel/lib/sentry_serial/sentry_send: /opt/ros/noetic/lib/librosconsole_backend_interface.so
/home/zhang/下载/RoboDetect/devel/lib/sentry_serial/sentry_send: /usr/lib/x86_64-linux-gnu/liblog4cxx.so
/home/zhang/下载/RoboDetect/devel/lib/sentry_serial/sentry_send: /usr/lib/x86_64-linux-gnu/libboost_regex.so.1.71.0
/home/zhang/下载/RoboDetect/devel/lib/sentry_serial/sentry_send: /opt/ros/noetic/lib/libxmlrpcpp.so
/home/zhang/下载/RoboDetect/devel/lib/sentry_serial/sentry_send: /opt/ros/noetic/lib/libroscpp_serialization.so
/home/zhang/下载/RoboDetect/devel/lib/sentry_serial/sentry_send: /opt/ros/noetic/lib/librostime.so
/home/zhang/下载/RoboDetect/devel/lib/sentry_serial/sentry_send: /usr/lib/x86_64-linux-gnu/libboost_date_time.so.1.71.0
/home/zhang/下载/RoboDetect/devel/lib/sentry_serial/sentry_send: /opt/ros/noetic/lib/libcpp_common.so
/home/zhang/下载/RoboDetect/devel/lib/sentry_serial/sentry_send: /usr/lib/x86_64-linux-gnu/libboost_system.so.1.71.0
/home/zhang/下载/RoboDetect/devel/lib/sentry_serial/sentry_send: /usr/lib/x86_64-linux-gnu/libboost_thread.so.1.71.0
/home/zhang/下载/RoboDetect/devel/lib/sentry_serial/sentry_send: /usr/lib/x86_64-linux-gnu/libconsole_bridge.so.0.4
/home/zhang/下载/RoboDetect/devel/lib/sentry_serial/sentry_send: sentry_serial/CMakeFiles/sentry_send.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/zhang/下载/RoboDetect/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX executable /home/zhang/下载/RoboDetect/devel/lib/sentry_serial/sentry_send"
	cd /home/zhang/下载/RoboDetect/build/sentry_serial && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/sentry_send.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
sentry_serial/CMakeFiles/sentry_send.dir/build: /home/zhang/下载/RoboDetect/devel/lib/sentry_serial/sentry_send

.PHONY : sentry_serial/CMakeFiles/sentry_send.dir/build

sentry_serial/CMakeFiles/sentry_send.dir/clean:
	cd /home/zhang/下载/RoboDetect/build/sentry_serial && $(CMAKE_COMMAND) -P CMakeFiles/sentry_send.dir/cmake_clean.cmake
.PHONY : sentry_serial/CMakeFiles/sentry_send.dir/clean

sentry_serial/CMakeFiles/sentry_send.dir/depend:
	cd /home/zhang/下载/RoboDetect/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zhang/下载/RoboDetect/src /home/zhang/下载/RoboDetect/src/sentry_serial /home/zhang/下载/RoboDetect/build /home/zhang/下载/RoboDetect/build/sentry_serial /home/zhang/下载/RoboDetect/build/sentry_serial/CMakeFiles/sentry_send.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : sentry_serial/CMakeFiles/sentry_send.dir/depend

