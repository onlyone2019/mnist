//
// Created by wangjie on 23-4-7.
//

#ifndef MNIST_MODEL_H
#define MNIST_MODEL_H

#include <torch/nn.h>

struct LeNet : torch::nn::Module {
    LeNet() {
        // Construct and register two Linear submodules.
        LeNet::conv1 = register_module("conv1", torch::nn::Conv2d(1, 6, 5));
        LeNet::conv2 = register_module("conv2", torch::nn::Conv2d(6, 16, 5));
        LeNet::fc1 = register_module("fc1", torch::nn::Linear(16 * 4 * 4, 120));
        LeNet::fc2 = register_module("fc2", torch::nn::Linear(120, 84));
        LeNet::fc3 = register_module("fc3", torch::nn::Linear(84, 10));

        LeNet::pool = register_module("pool", torch::nn::MaxPool2d(2));
        LeNet::relu = register_module("relu", torch::nn::ReLU());
    }

    torch::Tensor forward(torch::Tensor x) {
        x = LeNet::conv1->forward(x);
        x = LeNet::relu->forward(x);
        x = LeNet::pool->forward(x);

        x = LeNet::conv2->forward(x);
        x = LeNet::relu->forward(x);
        x = LeNet::pool->forward(x);

        x = x.view({x.size(0), 16 * 4 * 4});
        x = LeNet::fc1->forward(x);
        x = LeNet::relu->forward(x);
        x = LeNet::fc2->forward(x);
        x = LeNet::relu->forward(x);
        x = LeNet::fc3->forward(x);
        x = torch::log_softmax(x, -1);
        return x;
    }

    // Use one of many "standard library" modules.
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::MaxPool2d pool{nullptr};
    torch::nn::ReLU relu{nullptr};
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};

#endif //MNIST_MODEL_H
