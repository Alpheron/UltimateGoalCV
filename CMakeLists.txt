cmake_minimum_required(VERSION 3.17)
project(UltimateGoalCV)
set(CMAKE_CXX_STANDARD 20)

FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(Eigen3 REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${Eigen3_INCLUDE_DIRS})
add_executable(singleImage ${CMAKE_CURRENT_SOURCE_DIR}/src/ringStack/cpp/singleImage.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/ringStack/cpp/singleImage.h)
add_executable(phoneCam ${CMAKE_CURRENT_SOURCE_DIR}/src/ringStack/cpp/phoneCam.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/ringStack/cpp/phoneCam.h)
TARGET_LINK_LIBRARIES(singleImage ${Eigen3_LIBS})
TARGET_LINK_LIBRARIES(singleImage ${OpenCV_LIBS})
TARGET_LINK_LIBRARIES(phoneCam ${Eigen3_LIBS})
TARGET_LINK_LIBRARIES(phoneCam ${OpenCV_LIBS})