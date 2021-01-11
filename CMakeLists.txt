cmake_minimum_required(VERSION 2.6)
project(yolov3)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

set(CUDA_NVCC_PLAGS ${CUDA_NVCC_PLAGS};-std=c++11;-g;-G;-gencode;arch=compute_75;code=sm_75)

include_directories(${PROJECT_SOURCE_DIR}/include)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
include_directories(/home/bhap/Downloads/TensorRT-7.0.0.11/targets/x86_64-linux-gnu/include/)
link_directories(/home/bhap/Downloads/TensorRT-7.0.0.11/targets/x86_64-linux-gnu/lib/)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

cuda_add_library(yololayer SHARED ${PROJECT_SOURCE_DIR}/src/yololayer.cu)
# target_link_libraries(yololayer nvinfer cudart nvonnxparser)

set(OpenCV_DIR /home/bhap/Downloads/opencv-3.2.0/build)
find_package(OpenCV 3 REQUIRED)
message(STATUS "Opencv library status: ")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(yolov3 ${PROJECT_SOURCE_DIR}/src/yolov3.cpp)
target_link_libraries(yolov3 nvinfer)
target_link_libraries(yolov3 cudart)
target_link_libraries(yolov3 nvonnxparser)
target_link_libraries(yolov3 yololayer)
target_link_libraries(yolov3 ${OpenCV_LIBS})

add_definitions(-O2 -pthread)
