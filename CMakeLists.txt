cmake_minimum_required(VERSION 3.17)
project(UltimateGoalCV)
set(CMAKE_CXX_STANDARD 20)

FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(Eigen3 REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${Eigen3_INCLUDE_DIRS})
add_executable(UltimateGoalCV src/cpp/main.cpp src/cpp/phoneCam.cpp src/cpp/main.h src/cpp/phoneCam.h)

TARGET_LINK_LIBRARIES(UltimateGoalCV ${Eigen3_LIBS})
TARGET_LINK_LIBRARIES(UltimateGoalCV ${OpenCV_LIBS})