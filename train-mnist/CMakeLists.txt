cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(mnist)

find_package(Torch REQUIRED PATHS "/home/wangjie/software/libtorch/share/cmake/Torch/")
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS})

add_executable(train
        src/train.cpp
        src/utils.cpp
        )
target_link_libraries(train ${TORCH_LIBRARIES} ${OpenCV_LIBS})
set_property(TARGET train PROPERTY CXX_STANDARD 20)

add_executable(detect
        src/detect.cpp
        src/utils.cpp
        )
target_link_libraries(detect ${TORCH_LIBRARIES} ${OpenCV_LIBS})
set_property(TARGET detect PROPERTY CXX_STANDARD 20)