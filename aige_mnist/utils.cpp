//
// Created by wangjie on 23-4-4.
//
#include "include/utils.h"

torch::Device select_device() {
    // select device
    torch::DeviceType device_type;
    if (torch::cuda::is_available()) {
        std::cout << "CUDA available! Use GPU." << std::endl;
        device_type = torch::kCUDA;
    } else {
        std::cout << "Use CPU." << std::endl;
        device_type = torch::kCPU;
    }
    torch::Device device(device_type);
    return device;
}