# 父目录下的CMakeLists.txt
cmake_minimum_required(VERSION 3.10.0)
project(TraditionalStitch)

#include_directories(HomoStitch)
#include_directories(MeshStitch)
#include_directories(example)

include_directories(
${CMAKE_CURRENT_SOURCE_DIR}/HomoStitch/include 
${CMAKE_CURRENT_SOURCE_DIR}/HomoStitch/include/Blend 
${CMAKE_CURRENT_SOURCE_DIR}/HomoStitch/include/StitchFilter 
${CMAKE_CURRENT_SOURCE_DIR}/HomoStitch/include/StitichBase 
${CMAKE_CURRENT_SOURCE_DIR}/HomoStitch/include/StitichBase/Cylinder 
${CMAKE_CURRENT_SOURCE_DIR}/HomoStitch/include/StitichBase/Estimate 
${CMAKE_CURRENT_SOURCE_DIR}/HomoStitch/include/StitichBase/Trans 
${CMAKE_CURRENT_SOURCE_DIR}/HomoStitch/common 
${PROJECT_SOURCE_DIR}/3dparty/spdlog-1.8.5/include
 )

add_subdirectory(HomoStitch lib) 
#add_subdirectory(MeshStitch lib) 

SET(SRC_HomoStitch example/HomoStitchTest.cpp)
SET(SRC_MeshStitch example/MeshStitch.cpp)

add_executable(HomoStitch_Test ${SRC_HomoStitch})
target_link_libraries(HomoStitch_Test HomoStitch)

#add_executable(MeshStitch_Test ${SRC_MeshStitch})
#target_link_libraries(MeshStitch_Test MeshStitch)


set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

message(STATUS "This is BINARY dir" ${PROJECT_BINARY_DIR}/lib)      # build所在路径
message(STATUS "This is SOURCE dir" ${PROJECT_SOURCE_DIR})          # 工程路径