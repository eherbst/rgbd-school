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
add_definitions(-std=c++0x -Wall -fPIC -mfpmath=sse -mmmx -msse -msse2 -msse3)

#baxter catkin pkgs; TODO what's the proper way to incorporate this in rosbuild?
include_directories(/home/eherbst/proj/baxter/ros_ws/devel/include)

find_package(QtForRGBD REQUIRED)

find_package(OpenNI2 REQUIRED)

find_package(PeterIntelMapping REQUIRED)

FIND_PACKAGE(CUDA)
set(CUDA_NVCC_FLAGS -ccbin /usr/local/bin/gcc44 ${CUDA_NVCC_FLAGS} -arch=sm_20 -Xcompiler -fPIC) #-g -G are needed if using cuda-gdb
add_definitions(-I${CUDA_INCLUDE_DIRS})

#OMPL
include_directories(/home/eherbst/include)
link_directories(/home/eherbst/lib)

#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

##################################################################
# libs

rosbuild_add_library(online_obj_modeling
							src/onlineModeler.cpp
							)
target_link_libraries(online_obj_modeling ${PETER_INTEL_LIBS})
target_link_libraries(online_obj_modeling CGAL)

rosbuild_add_library(active_obj_modeling
							src/baxterContinuousTrajSender.cpp
							src/collisionChecking.cpp
							src/continuousTrajSender.cpp
							src/diffingConnComps.cpp
							src/freeFlyingCameraContinuousTrajSender.cpp
							src/marvinContinuousTrajSender.cpp
							src/onlineActiveModeler.cpp
							src/polyhedronApproximationWithSpheres.cpp #TODO move this to another package
							src/progressiveMeshes/quadrics.cpp
							src/robotMeshSimplification.cpp
							)
target_link_libraries(active_obj_modeling online_obj_modeling)
target_link_libraries(active_obj_modeling /home/eherbst/lib/libompl.so)
target_link_libraries(active_obj_modeling procps) #TODO debugging memory usage 20140407; look for readproc.h

##################################################################
# drivers

#can also do non-active online modeling
rosbuild_add_executable(activeModelingDriver src/activeModelingDriver.cpp)
rosbuild_link_boost(activeModelingDriver system filesystem program_options date_time)
target_link_libraries(activeModelingDriver active_obj_modeling)
target_link_libraries(activeModelingDriver ${QT_LIBRARIES_ALL})

#used in the icra14 submission
rosbuild_add_executable(processHumanSegmDemo src/processHumanSegmDemo.cpp)
rosbuild_link_boost(processHumanSegmDemo system filesystem program_options)
target_link_libraries(processHumanSegmDemo active_obj_modeling ${PETER_INTEL_LIBS})
#set_target_properties(processHumanSegmDemo PROPERTIES LINK_FLAGS -v)

#for making a bkgnd map with human-caused camera motion
#20140123 doesn't quite build, and the recording code is mostly commented out in other classes anyway because I haven't been using it
#rosbuild_add_executable(buildMapOnMarvin src/buildMapOnMarvin.cpp)
#rosbuild_link_boost(buildMapOnMarvin system filesystem program_options)
#target_link_libraries(buildMapOnMarvin active_obj_modeling)

rosbuild_add_executable(highlightSegmentForIcra14Video src/highlightSegmentForIcra14Video.cpp)

rosbuild_add_executable(locateMostMovingAreaInVideo src/locateMostMovingAreaInVideo.cpp)
rosbuild_link_boost(locateMostMovingAreaInVideo program_options)

rosbuild_add_executable(locateMostMovingAreaInVideoZachFlow src/locateMostMovingAreaInVideoZachFlow.cpp)
rosbuild_link_boost(locateMostMovingAreaInVideoZachFlow system filesystem program_options)

##################################################################
# tests

#rosbuild_add_executable(testCHOMP src/tests/testCHOMP.cpp)

#rosbuild_add_executable(testPCLNormals src/tests/testPCLNormals.cpp)
#target_link_libraries(testPCLNormals vtkCommon vtkFiltering vtkRendering)

#fiddling with making output from many threads and pieces of a program less intertwined -- don't quite have something workable yet
#rosbuild_add_executable(testMultipleStreamsMultipleTerminals src/tests/testMultipleStreamsMultipleTerminals.cpp)
#rosbuild_link_boost(testMultipleStreamsMultipleTerminals iostreams)

#20140115 debugging link problems w/ peter-intel
#rosbuild_add_executable(testPIMLinking src/tests/testPIMLinking.cpp)
#target_link_libraries(testPIMLinking ${PETER_INTEL_LIBS})

rosbuild_add_executable(testModelerMemLeak src/tests/testModelerMemLeak.cpp)
target_link_libraries(testModelerMemLeak ${PETER_INTEL_LIBS})
