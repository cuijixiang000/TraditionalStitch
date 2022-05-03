cmake_minimum_required(VERSION 3.10.0)
#PROJECT(HomoStitch)

#IF(NOT CMAKE_BUILD_TYPE)
  #SET(CMAKE_BUILD_TYPE Release)
#ENDIF()
#MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})


#ADD OPENCV
FIND_PACKAGE(OpenCV 4.0.0 REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIR})
LINK_DIRECTORIES($(OpenCV_LIBS_DIR))

#ADD Eigen
FIND_PACKAGE(Eigen3 REQUIRED)
INCLUDE_DIRECTORIES(${EIGEN3_INCLUDE_DIRS})

find_package(OpenMP)
if (NOT OPENMP_FOUND)
	message(WARNING "OpenMP is not found. Will be slow.")
else()
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
INCLUDE_DIRECTORIES(
${CMAKE_CURRENT_SOURCE_DIR}/include 
${CMAKE_CURRENT_SOURCE_DIR}/include/Blend 
${CMAKE_CURRENT_SOURCE_DIR}/include/StitchFilter 
${CMAKE_CURRENT_SOURCE_DIR}/include/StitichBase 
${CMAKE_CURRENT_SOURCE_DIR}/include/StitichBase/Cylinder 
${CMAKE_CURRENT_SOURCE_DIR}/include/StitichBase/Estimate 
${CMAKE_CURRENT_SOURCE_DIR}/include/StitichBase/Trans 
${CMAKE_CURRENT_SOURCE_DIR}/common 
${PROJECT_SOURCE_DIR}/3dparty/spdlog-1.8.5/include
)
INCLUDE_DIRECTORIES(
${CMAKE_CURRENT_SOURCE_DIR}/src/ 
${CMAKE_CURRENT_SOURCE_DIR}/src/Blend 
${CMAKE_CURRENT_SOURCE_DIR}/src/StitchFilter 
${CMAKE_CURRENT_SOURCE_DIR}/src/StitichBase 
${CMAKE_CURRENT_SOURCE_DIR}/src/StitichBase/Cylinder 
${CMAKE_CURRENT_SOURCE_DIR}/src/StitichBase/Estimate 
${CMAKE_CURRENT_SOURCE_DIR}/src/StitichBase/Trans 
)

file(GLOB SOURCES 
src/Blend/*.cpp 
src/StitchFilter/*.cpp 
src/StitichBase/Cylinder/*.cpp 
src/StitichBase/Estimate/*.cpp 
src/StitichBase/Trans/*.cpp 
src/StitichBase/*.cpp 
include/Blend/*.h 
include/StitchFilter/*.h 
include/StitichBase/Cylinder/*.h 
include/StitichBase/Estimate/*.h 
include/StitichBase/Trans/*.h 
include/StitichBase/*.h 
common/*.h 
common/*.cpp
)

add_library(HomoStitch ${SOURCES})
#add_library(${PROJECT_NAME} ${SOURCES})
TARGET_LINK_LIBRARIES(HomoStitch ${OpenCV_LIBS} -fopenmp)

SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)