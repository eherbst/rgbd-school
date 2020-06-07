cmake_minimum_required(VERSION 2.8)
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

# Set the build type.  Options are:
#  Coverage       : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries
set(ROS_BUILD_TYPE RelWithDebInfo)
rosbuild_init()
rosbuild_add_boost_directories()
add_definitions(-std=c++0x -Wall -fPIC) #-mfpmath=sse -mmmx -msse -msse2 -msse3

#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

##################################################################
# libs

##################################################################
# drivers

rosbuild_add_executable(sceneFlowBasedSLAM src/sceneFlowBasedSLAM.cpp)
rosbuild_link_boost(sceneFlowBasedSLAM system filesystem)

rosbuild_add_executable(sceneFlowBasedSLAMFrame2Map src/sceneFlowBasedSLAMFrame2Map.cpp)
rosbuild_link_boost(sceneFlowBasedSLAMFrame2Map system filesystem)

rosbuild_add_executable(dficpBasedSLAM src/dficpBasedSLAM.cpp)
rosbuild_link_boost(dficpBasedSLAM system filesystem)

rosbuild_add_executable(icpBasedVO src/icpBasedVO.cpp)
rosbuild_link_boost(icpBasedVO system filesystem)

rosbuild_add_executable(viewFlowSLAMResultsAsFrames src/viewFlowSLAMResultsAsFrames.cpp)
rosbuild_link_boost(viewFlowSLAMResultsAsFrames system filesystem)
