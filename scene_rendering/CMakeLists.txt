cmake_minimum_required(VERSION 2.4.6)
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

# Set the build type.  Options are:
#  Coverage       : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries
#set(ROS_BUILD_TYPE RelWithDebInfo)
rosbuild_init()
rosbuild_add_boost_directories()

add_definitions(-std=c++0x -Wall -fPIC -mfpmath=sse -mmmx -msse -msse2 -msse3)

find_package(QtForRGBD REQUIRED)

find_package(OpenNI2 REQUIRED)

FIND_PACKAGE(CUDA)
set(CUDA_NVCC_FLAGS -ccbin /usr/local/bin/gcc44 ${CUDA_NVCC_FLAGS} -arch=sm_20 -Xcompiler -fPIC) #-g -G are needed if using cuda-gdb

#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

##################################################################
# libs

cuda_add_library(scene_rendering_cuda
							src/sceneSamplerCUDARenderedTSDF.cu
							src/castRaysIntoSurfels.cu
							src/viewScoringRenderer.cu
							)
target_link_libraries(scene_rendering_cuda rgbd_cuda_util_cu)

rosbuild_add_library(scene_rendering
							src/castRaysIntoSurfels.cpp
							src/pointCloudRenderer.cpp
							src/sceneRenderingCache.cpp
							src/sceneSamplerImages.cpp
							src/sceneSamplerPointCloud.cpp
							src/sceneSamplerRenderedTSDF.cpp
							src/surfelCloudRenderer.cpp
							src/triangulatedMeshRenderer.cpp
							src/viewScoringRenderer.cpp
							src/voxelGridRenderer.cpp
							src/voxelGridSceneInfo.cpp
							)
target_link_libraries(scene_rendering opengl_util GLU GL)
target_link_libraries(scene_rendering vrip_utils)
target_link_libraries(scene_rendering scene_rendering_cuda)
rosbuild_link_boost(scene_rendering system filesystem)

##################################################################
# drivers

#test driver for mesh renderer
rosbuild_add_executable(testglstuff src/tests/testglstuff.cpp)
target_link_libraries(testglstuff scene_rendering)

#test driver for voxel grid renderer
rosbuild_add_executable(testglstuff2 src/tests/testglstuff2.cpp)
target_link_libraries(testglstuff2 scene_rendering)

rosbuild_add_executable(testPointCloudRenderer src/tests/testPointCloudRenderer.cpp)
target_link_libraries(testPointCloudRenderer scene_rendering)
