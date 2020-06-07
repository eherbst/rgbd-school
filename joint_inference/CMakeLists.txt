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

add_definitions(-std=c++0x -Wall -O3 -fPIC) #-mfpmath=sse -mmmx -msse -msse2 -msse3

#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

#rosbuild_add_library(scene_matching_mln
#							src/sceneMatchingMLN.cpp
#							)
#rosbuild_add_library(scene_matching_crf
#							src/sceneMatchingCRF.cpp
#							)

rosbuild_add_executable(mlntests src/mlntests.cpp)
target_link_libraries(mlntests scene_matching_common)
rosbuild_link_boost(mlntests system filesystem)

#rosbuild_add_executable(parseAlchemyOutputClusterSizes src/parseAlchemyOutputClusterSizes.cpp)

#rosbuild_add_executable(visualizeAlchemyResults src/visualizeAlchemyResults.cpp)
#target_link_libraries(visualizeAlchemyResults scene_matching_common)
#rosbuild_link_boost(visualizeAlchemyResults serialization)

#rosbuild_add_executable(visualizeFastinfResults src/visualizeFastinfResults.cpp)
#target_link_libraries(visualizeFastinfResults scene_matching_common)
#rosbuild_link_boost(visualizeFastinfResults serialization)

#rosbuild_add_executable(calculateAlchemyOutputAccuracy src/calculateAlchemyOutputAccuracy.cpp)

#rosbuild_add_executable(reconstructObjsAfterInference src/reconstructObjsAfterInference.cpp)
#target_link_libraries(reconstructObjsAfterInference scene_matching_common)
#rosbuild_link_boost(reconstructObjsAfterInference serialization)

#rosbuild_add_executable(createGTEvidenceForMLNLearning src/createGTEvidenceForMLNLearning.cpp)
#target_link_libraries(createGTEvidenceForMLNLearning scene_matching_common)
