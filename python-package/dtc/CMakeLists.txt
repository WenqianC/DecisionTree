cmake_minimum_required (VERSION 2.6)

cmake_policy(SET CMP0015 NEW)

project (Decision_tree_classifier_lib)

if(UNIX)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=gnu++11")
endif()

# The version number
set (Decision_tree_classifier_lib_VERSION_MAJOR 1)
set (Decision_tree_classifier_lib_VERSION_MINOR 0)

set (USE_SERVER 0)


# add the binary tree to the search path for include files
# so that we will find TutorialConfig.h
include_directories("${PROJECT_BINARY_DIR}")


# link Eigen
if(USE_SERVER)
set(Eigen_DIR /lci/project/raid1/jimmy_chen/lib/eigen_3.2.6/include/eigen3)
include_directories (${Eigen_DIR})
else()
set(Eigen_DIR /Users/jimmy/Code/Eigen_3.3.5)
include_directories (${Eigen_DIR})
endif()


# add directory files for head files?
include_directories (./src ./src/dt_util)

set(TRAIN_SRC ./src/dtc_python.cpp)

# .cpp in src
set(SOURCE_CODE1 ./src/dt_classifier_builder.cpp ./src/dt_classifier.cpp ./src/dtc_tree.cpp ./src/dtc_param.cpp)


# .cpp in dt_util
set(SOURCE_CODE_DT_UTIL ./src/dt_util/dt_random.cpp ./src/dt_util/dt_param_parser ./src/dt_util/dt_util.cpp ./src/dt_util/vnl_random.cxx)

set(SOURCE_CODE ${SOURCE_CODE1} ${SOURCE_CODE_DT_UTIL})

# add the executable
#add_executable(DTC_train ${TRAIN_SRC} ${SOURCE_CODE})
#add_executable(DTC_test ${TEST_SRC} ${SOURCE_CODE})

# add library
add_library(dtc_python SHARED ${TRAIN_SRC} ${SOURCE_CODE})

















