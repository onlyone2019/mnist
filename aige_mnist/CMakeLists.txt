cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(mnist-infer)

set(Torch_DIR "/home/wangjie/software/libtorch/share/cmake/Torch/")
find_package(Torch REQUIRED PATHS "/home/wangjie/software/libtorch/share/cmake/Torch/")
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS})

add_library(mnist-infer SHARED ${PROJECT_SOURCE_DIR}/mnist-infer.cpp)
target_link_libraries(mnist-infer ${TORCH_LIBRARIES} ${OpenCV_LIBS})
set_property(TARGET mnist-infer PROPERTY CXX_STANDARD 20)