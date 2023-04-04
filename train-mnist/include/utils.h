//
// Created by wangjie on 23-4-4.
//

#ifndef MNIST_UTILS_H
#define MNIST_UTILS_H

#include "model.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <sys/stat.h>
#include <torch/cuda.h>
#include <torch/optim.h>
#include <unistd.h>
#include <vector>

torch::Device select_device();

template<typename DataLoader>
double train(size_t epoch,
             std::shared_ptr<LeNet> model,
             torch::Device device,
             DataLoader &data_loader,
             torch::optim::Optimizer &optimizer,
             size_t dataset_size) {
    // set train mode
    model->train();
    size_t batch_index = 1;
    // Iterate data loader to yield batches from the dataset
    for (auto &batch : data_loader) {
        auto images = batch.data.to(device);
        auto targets = batch.target.to(device);
        optimizer.zero_grad();
        auto output = model->forward(images);
        auto loss = torch::nll_loss(output, targets);
        AT_ASSERT(!std::isnan(loss.template item<float>()));

        // Compute gradients
        loss.backward();
        // Update the parameters
        optimizer.step();

        if (++batch_index % 10 == 0) {
            std::printf("\rTrain Epoch: %ld [%5ld/%5ld] Loss: %.4f",
                        epoch,
                        batch_index * batch.data.size(0),
                        dataset_size,
                        loss.template item<float>());
        }
    }
}

template<typename DataLoader>
std::vector<double> evaluate(std::shared_ptr<LeNet> model,
                             torch::Device device,
                             DataLoader &data_loader,
                             size_t dataset_size) {
    torch::NoGradGuard no_grad;
    model->eval();

    // define value.
    double loss = 0.;
    double accuracy;
    size_t correct = 0;
    std::vector<double> result;

    for (const auto &batch : data_loader) {
        auto data = batch.data.to(device);
        auto targets = batch.target.to(device);
        auto output = model->forward(data);
        loss += torch::nll_loss(output, targets, {}, torch::Reduction::Sum).template item<double>();
        auto pred = output.argmax(1);
        correct += pred.eq(targets).sum().template item<int64_t>();
    }
    loss /= dataset_size;
    accuracy = static_cast<double>(correct) / dataset_size;

    std::printf("\nTest set: Average loss: %.4f | Accuracy: %.3f\n", loss, accuracy);

    result.push_back(loss);
    result.push_back(accuracy);

    return result;
}

#endif //MNIST_UTILS_H
