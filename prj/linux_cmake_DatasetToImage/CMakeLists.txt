PROJECT(DatasetToImage)
CMAKE_MINIMUM_REQUIRED(VERSION 3.0)

SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c17")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

SET(CMAKE_BUILD_TYPE "Release")
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O2")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O2")

MESSAGE(STATUS "cmake current source dir: ${CMAKE_CURRENT_SOURCE_DIR}")
SET(PATH_TEST_FILES ${CMAKE_CURRENT_SOURCE_DIR}/./../../demo/DatasetToImage)
#SET(PATH_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/./../../src/)
MESSAGE(STATUS "path test files: ${PATH_TEST_FILES}")

FIND_PACKAGE(OpenCV)

# head file search path
INCLUDE_DIRECTORIES(
	${PATH_TEST_FILES}
	${OpenCV_INCLUDE_DIRS}
)

LINK_DIRECTORIES(${OpenCV_LIB_DIR})

# find dep library
#SET(SEETAFACE_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/build CACHE PATH "seetaface library path")
#MESSAGE(STATUS "seetaface library dir: ${SEETAFACE_LIB_DIR}")
#LINK_DIRECTORIES(${SEETAFACE_LIB_DIR})

# recursive query match files :*.cpp
FILE(GLOB_RECURSE TEST_CPP_LIST ${PATH_TEST_FILES}/*.cpp)
#MESSAGE(STATUS "test cpp list: ${TEST_CPP_LIST} ${TEST_C_LIST}")

# build executable program
ADD_EXECUTABLE(DatasetToImage ${TEST_CPP_LIST} ${TEST_C_LIST})
# add dependent library: static and dynamic
TARGET_LINK_LIBRARIES(DatasetToImage ${OpenCV_LIBS} pthread)
